"""
tests/test_diagnosis.py
-----------------------
Standalone tests for File 6 (the diagnosis path's cheap loop + sufficiency gate).
Run from the project root:

    python -m tests.test_diagnosis

No pytest required. All tests build a ConsultationState and parse an IntakeResult,
so they need pydantic (your env has it).

What these tests PROVE:
    1. When the model is not sufficient, the path asks the model's next question
       and stays in INTERVIEW, recording the differential and chief complaint.
    2. When the model is sufficient AND the top hypothesis clears the confidence
       floor, the path converges (phase -> MILESTONE) and runs the milestone seam.
    3. The confidence floor OVERRIDES a "sufficient" on a weak top hypothesis —
       the path keeps interviewing (using a fallback question if the model gave
       none).
    4. The iteration cap OVERRIDES the model: once hit, the path converges even
       though the model wanted to keep going.
    5. On re-entry with phase already MILESTONE, the path runs the milestone seam
       and makes NO loop call.
    6. Invalid model output (out-of-range confidence) surfaces as LLMError via
       complete_typed.
    7. DiagnosisPath registers and dispatches through the real Router as an
       ordinary peer.

A minimal ScriptedGateway is defined here (rather than importing the project's
shared FakeGateway) so this file is self-contained and makes no assumption about
that fake's constructor. complete_typed is inherited from the base gateway, so the
scripted structured responses flow straight through it into IntakeResult.
"""

from __future__ import annotations

import asyncio
import inspect
import traceback
from typing import Any, AsyncIterator, Sequence

from app.core.enums import DiagnosticPhase, Route
from app.core.state import ConsultationState, PatientRecord, SessionState, VisitState
from app.llm.base import LLMError, LLMGateway, LLMMessage, ModelTier
from app.paths.diagnosis import DiagnosisPath, IntakeResult
from app.paths.diagnosis.path import _FALLBACK_QUESTION
from app.routing import Router, TransitionKind, TurnClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedGateway(LLMGateway):
    """Returns canned structured dicts in order; records each call. complete_typed
    (inherited from LLMGateway) validates these into the caller's schema."""

    def __init__(self, structured: list[dict[str, Any]]) -> None:
        self._structured = list(structured)
        self.calls: list[dict[str, Any]] = []

    async def complete_structured(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None, model: ModelTier | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user, "model": model})
        if not self._structured:
            raise LLMError("ScriptedGateway: no more canned responses")
        return self._structured.pop(0)

    async def complete_text(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None, model: ModelTier | None = None,
    ) -> str:
        return ""

    async def stream_text(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None, model: ModelTier | None = None,
    ) -> AsyncIterator[str]:
        yield ""  # not used by the diagnosis loop; present to satisfy the ABC


class FixedClassifier(TurnClassifier):
    def __init__(self, route: Route) -> None:
        self._route = route

    async def classify(self, *, turn: str, state: ConsultationState, available) -> Route:
        return self._route


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _state(
    *, phase: DiagnosticPhase = DiagnosticPhase.INTAKE,
    iterations: int = 0, active: Route = Route.DIAGNOSIS,
) -> ConsultationState:
    patient = PatientRecord(patient_id="p1", age=34)
    visit = VisitState(visit_id="v1", patient_id="p1")
    session = SessionState(
        session_id="s1", patient_id="p1", visit_id="v1",
        active_route=active, phase=phase, interview_iterations=iterations,
    )
    return ConsultationState(patient=patient, visit=visit, session=session)


def _intake(
    sufficient: bool, *, conf: float = 0.0,
    question: str | None = "What symptoms are you having?",
    condition: str = "Tension headache",
) -> dict[str, Any]:
    return {
        "understanding": "Patient reports a headache.",
        "chief_complaint": "headache",
        "hypotheses": [
            {"condition": condition, "confidence": conf, "supporting": [], "against": []}
        ],
        "sufficient": sufficient,
        "sufficiency_rationale": "test scenario",
        "next_question": question,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_interview_asks_question_when_insufficient() -> None:
    gw = ScriptedGateway([_intake(False, conf=0.3, question="Where is the pain?")])
    path = DiagnosisPath(gw)
    st = _state()

    res = await path.handle(turn="My head hurts", state=st)

    assert res.reply == "Where is the pain?", res.reply
    assert res.transition.kind is TransitionKind.STAY
    assert st.session.phase is DiagnosticPhase.INTERVIEW
    assert st.session.interview_iterations == 1
    assert st.visit.chief_complaint == "headache"
    assert [h.condition for h in st.visit.differential] == ["Tension headache"]
    assert len(gw.calls) == 1
    assert gw.calls[0]["model"] is ModelTier.REASONING  # default tier


async def test_converges_when_sufficient_and_confident() -> None:
    gw = ScriptedGateway([_intake(True, conf=0.82, question=None)])
    path = DiagnosisPath(gw)  # default floor 0.5
    st = _state()

    res = await path.handle(turn="It's a mild tension headache, no other symptoms", state=st)

    assert st.session.phase is DiagnosticPhase.MILESTONE
    assert "assessment" in res.reply.lower(), res.reply
    assert res.transition.kind is TransitionKind.STAY
    # differential still captured on the converging turn
    assert st.visit.differential and st.visit.differential[0].confidence == 0.82


async def test_confidence_floor_blocks_premature_convergence() -> None:
    # Model says sufficient, but the top hypothesis is weak (< floor) -> keep going.
    gw = ScriptedGateway([_intake(True, conf=0.3, question="Any fever or neck stiffness?")])
    path = DiagnosisPath(gw, confidence_floor=0.5)
    st = _state()

    res = await path.handle(turn="not sure", state=st)

    assert st.session.phase is DiagnosticPhase.INTERVIEW, "floor must block convergence"
    assert res.reply == "Any fever or neck stiffness?"


async def test_floor_block_uses_fallback_when_model_gives_no_question() -> None:
    # Sufficient + low confidence + no next_question -> path supplies the fallback.
    gw = ScriptedGateway([_intake(True, conf=0.3, question=None)])
    path = DiagnosisPath(gw, confidence_floor=0.5)
    st = _state()

    res = await path.handle(turn="hmm", state=st)

    assert st.session.phase is DiagnosticPhase.INTERVIEW
    assert res.reply == _FALLBACK_QUESTION


async def test_iteration_cap_forces_convergence() -> None:
    # Model wants to keep going (sufficient=False) but the cap forces the milestone.
    gw = ScriptedGateway([_intake(False, conf=0.2, question="Tell me more")])
    path = DiagnosisPath(gw, max_interview_iterations=1)
    st = _state(iterations=0)

    res = await path.handle(turn="my head hurts a bit", state=st)

    assert st.session.interview_iterations == 1
    assert st.session.phase is DiagnosticPhase.MILESTONE, "cap must force convergence"
    assert "assessment" in res.reply.lower()


async def test_milestone_seam_on_reentry_makes_no_loop_call() -> None:
    gw = ScriptedGateway([])  # must NOT be consulted
    path = DiagnosisPath(gw)
    st = _state(phase=DiagnosticPhase.MILESTONE)

    res = await path.handle(turn="here is my X-ray result", state=st)

    assert "assessment" in res.reply.lower()
    assert gw.calls == [], "milestone re-entry must not run the interview loop"


async def test_invalid_confidence_surfaces_as_llm_error() -> None:
    gw = ScriptedGateway([_intake(False, conf=1.5)])  # confidence out of [0,1]
    path = DiagnosisPath(gw)
    st = _state()
    try:
        await path.handle(turn="...", state=st)
        raise AssertionError("expected LLMError for an out-of-range confidence")
    except LLMError:
        pass


async def test_registers_and_dispatches_as_router_peer() -> None:
    gw = ScriptedGateway([_intake(False, conf=0.3, question="Where exactly?")])
    router = Router(FixedClassifier(Route.DIAGNOSIS))
    router.register(DiagnosisPath(gw))  # registers exactly like any peer

    st = _state(active=Route.UNKNOWN)  # force a fresh classification
    res = await router.route_turn(turn="head hurts", state=st)

    assert res.reply == "Where exactly?"
    assert st.session.active_route is Route.DIAGNOSIS


def test_intake_result_schema_roundtrip() -> None:
    ir = IntakeResult.model_validate(_intake(False, conf=0.4))
    assert ir.sufficient is False
    assert ir.chief_complaint == "headache"
    assert ir.hypotheses[0].confidence == 0.4
    # the nested entries are real domain Hypothesis instances
    from app.core.state import Hypothesis
    assert isinstance(ir.hypotheses[0], Hypothesis)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_interview_asks_question_when_insufficient,
    test_converges_when_sufficient_and_confident,
    test_confidence_floor_blocks_premature_convergence,
    test_floor_block_uses_fallback_when_model_gives_no_question,
    test_iteration_cap_forces_convergence,
    test_milestone_seam_on_reentry_makes_no_loop_call,
    test_invalid_confidence_surfaces_as_llm_error,
    test_registers_and_dispatches_as_router_peer,
    test_intake_result_schema_roundtrip,
]


def main() -> None:
    passed = failed = 0
    for test in ALL_TESTS:
        try:
            result = test()
            if inspect.iscoroutine(result):
                asyncio.run(result)
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - the runner reports everything
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

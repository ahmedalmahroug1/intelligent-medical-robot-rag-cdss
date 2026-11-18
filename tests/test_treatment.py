"""
tests/test_treatment.py
-----------------------
Standalone tests for the TREATMENT path. Run from the project root:

    python -m tests.test_treatment   (or: PYTHONPATH=. python tests/test_treatment.py)

What these prove (all offline, against a fake gateway):
    1. Given a known condition, the path returns management guidance and stays.
    2. The DETERMINISTIC safety overlay flags a proposed medication the patient is
       allergic to — independently of whatever the model said.
    3. It flags duplication against an active medication.
    4. With no resolvable condition it asks for clarification (does not guess).
    5. If the patient is describing new symptoms, it hands back to DIAGNOSIS.
    6. It uses the converged working-diagnosis condition from the blackboard.
"""

from __future__ import annotations

import asyncio
import inspect
import traceback
from typing import Any, AsyncIterator, Sequence

from app.core.enums import Route, Sex, RiskTier
from app.core.state import (
    Allergy,
    ConsultationState,
    Medication,
    PatientRecord,
    SessionState,
    VisitState,
    WorkingDiagnosis,
)
from app.llm.base import LLMGateway, LLMMessage, ModelTier
from app.routing.contract import TransitionKind
from app.paths.treatment import TreatmentPath


class FakeGateway(LLMGateway):
    """Returns a single canned structured dict for every structured call."""

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.calls = 0

    async def complete_structured(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None, model: ModelTier | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        self.last_user = user
        return self._response

    async def complete_text(self, *, system, user, history=None, model=None) -> str:
        return ""

    async def stream_text(self, *, system, user, history=None, model=None) -> AsyncIterator[str]:
        yield ""


def _state(
    *,
    allergies: list[Allergy] | None = None,
    medications: list[Medication] | None = None,
    working_condition: str | None = None,
) -> ConsultationState:
    patient = PatientRecord(
        patient_id="p1", age=40, sex=Sex.MALE,
        allergies=allergies or [], medications=medications or [],
    )
    visit = VisitState(visit_id="v1", patient_id="p1")
    if working_condition:
        visit.working_diagnosis = WorkingDiagnosis(
            condition=working_condition, confidence=0.8, risk_tier=RiskTier.MODERATE,
            reasoning="…",
        )
    session = SessionState(session_id="s1", patient_id="p1", visit_id="v1",
                           active_route=Route.TREATMENT)
    return ConsultationState(patient=patient, visit=visit, session=session)


def _guidance(**overrides: Any) -> dict[str, Any]:
    base = {
        "condition": "migraine",
        "summary": "Manage with rest and trigger avoidance.",
        "self_care": ["sleep regularly", "stay hydrated"],
        "medication_options": [
            {"name": "Ibuprofen", "purpose": "pain relief", "caution": "take with food"},
        ],
        "red_flags": ["sudden worst-ever headache"],
        "clinician_note": "Consider prophylaxis if frequent.",
        "new_symptoms": False,
    }
    base.update(overrides)
    return base


# ── Tests ────────────────────────────────────────────────────────────────────

async def test_gives_guidance_and_stays() -> None:
    gw = FakeGateway(_guidance())
    res = await TreatmentPath(gw).handle(turn="how do I manage my migraines?", state=_state())
    assert "migraine" in res.reply.lower()
    assert "Ibuprofen" in res.reply
    assert "clinician" in res.reply.lower()           # disclaimer present
    assert res.transition.kind is TransitionKind.STAY


async def test_allergy_overlay_fires() -> None:
    # Model proposes ibuprofen; patient is allergic to it → overlay must flag it,
    # regardless of the model not having caught it.
    gw = FakeGateway(_guidance())
    st = _state(allergies=[Allergy(substance="ibuprofen", reaction="hives")])
    res = await TreatmentPath(gw).handle(turn="treat my migraine", state=st)
    assert "allergy" in res.reply.lower()
    assert "do not take" in res.reply.lower()


async def test_active_med_duplication_flag() -> None:
    gw = FakeGateway(_guidance())
    st = _state(medications=[Medication(name="ibuprofen", active=True)])
    res = await TreatmentPath(gw).handle(turn="treat my migraine", state=st)
    assert "already takes" in res.reply.lower()


async def test_clarifies_when_no_condition() -> None:
    gw = FakeGateway(_guidance(condition=""))
    res = await TreatmentPath(gw).handle(turn="help me", state=_state())
    assert "which condition" in res.reply.lower()
    assert gw.calls == 1


async def test_new_symptoms_hands_off_to_diagnosis() -> None:
    gw = FakeGateway(_guidance(condition="", new_symptoms=True))
    res = await TreatmentPath(gw).handle(turn="I suddenly have chest pain", state=_state())
    assert res.transition.kind is TransitionKind.SWITCH
    assert res.transition.route is Route.DIAGNOSIS


async def test_uses_working_diagnosis_condition() -> None:
    # Model returns an empty condition, but the blackboard has a converged dx →
    # the path proceeds using that condition rather than asking to clarify.
    gw = FakeGateway(_guidance(condition=""))
    st = _state(working_condition="type 2 diabetes")
    res = await TreatmentPath(gw).handle(turn="how do I manage it?", state=st)
    assert "type 2 diabetes" in res.reply.lower()
    assert res.transition.kind is TransitionKind.STAY


# ── Runner ───────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_gives_guidance_and_stays,
    test_allergy_overlay_fires,
    test_active_med_duplication_flag,
    test_clarifies_when_no_condition,
    test_new_symptoms_hands_off_to_diagnosis,
    test_uses_working_diagnosis_condition,
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
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""
tests/test_service.py
---------------------
Integration tests for File 8 (the ConsultationService orchestrator). Run from the
project root:

    python -m tests.test_service

Needs pydantic + aiosqlite (your env has both). These are end-to-end against a
real in-memory database, a real SafetyGuard, and a real Router — only the gateway
and the path are test doubles. They assert the contracts that make the system
safe when fully assembled:

    - a new session is created and persisted (bootstrap lives here, not in the store);
    - an existing patient's history is preserved across a new visit;
    - a routine turn routes to a path, and the transcript is recorded in order;
    - an EMERGENCY turn bypasses the router entirely (the path is never called)
      and returns escalation advice, with the safety result persisted;
    - escalation does not disturb the path's phase;
    - turns accumulate and persist across reloads;
    - sending to an unknown session raises.
"""

from __future__ import annotations

import asyncio
import inspect
import traceback
from typing import Any, AsyncIterator, Sequence

from app.core.enums import MessageRole, Route, TriageLevel
from app.core.state import ConsultationState, PatientRecord
from app.llm.base import LLMGateway, LLMMessage, ModelTier
from app.orchestration import ConsultationError, ConsultationService
from app.persistence.database import Database
from app.persistence.store import ConsultationStore
from app.routing import Path, PathResult, Router, Transition, TurnClassifier
from app.safety.guard import SafetyGuard


# ─────────────────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────────────────

class RoutineGateway(LLMGateway):
    """The only LLM caller exercised here is the guard; return a routine verdict.
    (Deterministic rules still force EMERGENCY on red-flag text, regardless.)"""

    async def complete_structured(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None, model: ModelTier | None = None,
    ) -> dict[str, Any]:
        return {"triage_level": "routine", "red_flags": [], "advice": None}

    async def complete_text(self, *, system, user, history=None, model=None) -> str:
        return ""

    async def stream_text(self, *, system, user, history=None, model=None) -> AsyncIterator[str]:
        yield ""


class FixedClassifier(TurnClassifier):
    def __init__(self, route: Route) -> None:
        self._route = route

    async def classify(self, *, turn, state, available) -> Route:
        return self._route


class RecordingPath(Path):
    route = Route.DIAGNOSIS

    def __init__(self) -> None:
        self.turns: list[str] = []

    async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
        self.turns.append(turn)
        return PathResult(reply=f"reply to: {turn}", transition=Transition.stay())


async def _make_service() -> tuple[ConsultationService, ConsultationStore, RecordingPath, Database]:
    db = Database(":memory:")
    await db.connect()
    store = ConsultationStore(db)
    gateway = RoutineGateway()
    guard = SafetyGuard(gateway)
    router = Router(FixedClassifier(Route.DIAGNOSIS))
    path = RecordingPath()
    router.register(path)
    service = ConsultationService(store=store, guard=guard, router=router)
    return service, store, path, db


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_start_creates_persisted_session() -> None:
    svc, store, path, db = await _make_service()
    try:
        state = await svc.start()
        sid = state.session.session_id
        assert state.patient.patient_id and state.visit.visit_id and sid
        loaded = await store.load(sid)
        assert loaded is not None
        assert loaded.session.session_id == sid
        assert loaded.visit.visit_id == state.visit.visit_id
        assert loaded.patient.patient_id == state.patient.patient_id
    finally:
        await db.close()


async def test_start_preserves_existing_patient_history() -> None:
    svc, store, path, db = await _make_service()
    try:
        await store.patients.save(
            PatientRecord(patient_id="pat-1", chronic_conditions=["asthma"])
        )
        state = await svc.start(patient_id="pat-1")
        assert state.patient.chronic_conditions == ["asthma"]
        assert state.visit.patient_id == "pat-1"
        assert state.session.patient_id == "pat-1"
    finally:
        await db.close()


async def test_routine_turn_routes_and_records_transcript() -> None:
    svc, store, path, db = await _make_service()
    try:
        state = await svc.start()
        sid = state.session.session_id

        outcome = await svc.send(session_id=sid, message="my head hurts")

        assert outcome.reply == "reply to: my head hurts"
        assert outcome.escalated is False
        assert outcome.triage_level is TriageLevel.ROUTINE
        assert outcome.active_route is Route.DIAGNOSIS
        assert path.turns == ["my head hurts"]

        loaded = await store.load(sid)
        roles = [(m.role, m.content) for m in loaded.session.transcript]
        assert roles == [
            (MessageRole.PATIENT, "my head hurts"),
            (MessageRole.DOCTOR_AI, "reply to: my head hurts"),
        ]
        # the reply message is tagged with the handling route
        assert loaded.session.transcript[-1].route is Route.DIAGNOSIS
    finally:
        await db.close()


async def test_emergency_bypasses_router() -> None:
    svc, store, path, db = await _make_service()
    try:
        state = await svc.start()
        sid = state.session.session_id

        outcome = await svc.send(
            session_id=sid, message="I have crushing chest pain radiating to my arm"
        )

        assert outcome.escalated is True
        assert outcome.triage_level is TriageLevel.EMERGENCY
        assert path.turns == [], "router/path MUST be bypassed on an emergency"
        assert "emergency" in outcome.reply.lower()

        loaded = await store.load(sid)
        assert loaded.session.last_safety is not None
        assert loaded.session.last_safety.triage_level is TriageLevel.EMERGENCY
        # the patient turn + escalation reply are still recorded
        assert [m.role for m in loaded.session.transcript] == [
            MessageRole.PATIENT, MessageRole.DOCTOR_AI
        ]
    finally:
        await db.close()


async def test_escalation_does_not_disturb_phase() -> None:
    svc, store, path, db = await _make_service()
    try:
        state = await svc.start()
        sid = state.session.session_id

        await svc.send(session_id=sid, message="hello")
        before = (await store.load(sid)).session.phase

        await svc.send(session_id=sid, message="sudden worst headache of my life")
        after = (await store.load(sid)).session.phase

        assert before == after
    finally:
        await db.close()


async def test_multi_turn_accumulates_and_persists() -> None:
    svc, store, path, db = await _make_service()
    try:
        state = await svc.start()
        sid = state.session.session_id

        await svc.send(session_id=sid, message="first")
        await svc.send(session_id=sid, message="second")

        loaded = await store.load(sid)
        contents = [m.content for m in loaded.session.transcript]
        assert contents == ["first", "reply to: first", "second", "reply to: second"]
        assert path.turns == ["first", "second"]
    finally:
        await db.close()


async def test_unknown_session_raises() -> None:
    svc, store, path, db = await _make_service()
    try:
        try:
            await svc.send(session_id="does-not-exist", message="hi")
            raise AssertionError("expected ConsultationError for unknown session")
        except ConsultationError:
            pass
    finally:
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_start_creates_persisted_session,
    test_start_preserves_existing_patient_history,
    test_routine_turn_routes_and_records_transcript,
    test_emergency_bypasses_router,
    test_escalation_does_not_disturb_phase,
    test_multi_turn_accumulates_and_persists,
    test_unknown_session_raises,
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

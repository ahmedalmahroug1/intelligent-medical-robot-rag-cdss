"""
app/orchestration/service.py
----------------------------
The ConsultationService: the per-turn orchestrator. It is the single place where
the safety and routing contracts established across the lower layers are actually
ASSEMBLED, so no individual client (web, robot, tests) can wire them wrongly.

It is deliberately client-agnostic. The web frontend and the robot are both just
callers of `start()` and `send()`; everything below this line operates on text as
the single source of truth. The FastAPI + SSE layer (File 8b) is a thin transport
wrapper over this service — it adds streaming and HTTP, no orchestration logic.

The per-turn flow `send()` enforces, in order:

    1. Load the blackboard for the session.
    2. Run the SAFETY GUARD first, on the current turn — before the router, and
       with the turn intentionally NOT yet in the transcript (the guard's and
       classifier's contract: they see prior history; the current message is
       passed explicitly).
    3. If the guard returns a red-flag triage (urgent or emergency), BYPASS the
       router entirely and return the escalation advice. Emergencies override
       everything; the router never even runs.
    4. Otherwise route the turn to a path (the router applies the path's
       transition internally — the service must not re-apply it).
    5. The orchestrator OWNS the transcript: append the patient turn and the
       reply now, after guard + routing have run on prior history.
    6. Persist all three tiers atomically.

Session bootstrap (`start()`) lives here, not in persistence: the store only
loads and saves, by design, so creating records and assigning IDs is
orchestration. `start()` reuses an existing patient's permanent record when a
patient id is supplied, so history carries across visits.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.enums import DiagnosticPhase, MessageRole, Route, TriageLevel
from app.core.state import ConsultationState, PatientRecord, SessionState, VisitState

if TYPE_CHECKING:  # injected collaborators; type-only to keep imports light
    from app.routing.router import Router
    from app.persistence.store import ConsultationStore
    from app.safety.guard import SafetyGuard

logger = logging.getLogger(__name__)


# Fallback escalation text if the guard somehow flags a red flag without advice.
_GENERIC_ESCALATION = (
    "Based on what you've described, please seek medical care now. If this is or "
    "may be an emergency, contact your local emergency services immediately."
)


class ConsultationError(RuntimeError):
    """Raised for orchestration problems, e.g. sending to an unknown session."""


@dataclass(frozen=True)
class TurnOutcome:
    """The result of one turn: enough for any client to render the reply and know
    what happened (whether it escalated, which path is active, the phase)."""
    reply: str
    session_id: str
    triage_level: TriageLevel
    escalated: bool
    active_route: Route
    phase: DiagnosticPhase
    red_flags: tuple[str, ...] = ()


def _new_id() -> str:
    return uuid.uuid4().hex


class ConsultationService:
    """Bootstraps sessions and processes turns, assembling guard + router + store."""

    def __init__(
        self,
        *,
        store: ConsultationStore,
        guard: SafetyGuard,
        router: Router,
        id_factory: Callable[[], str] = _new_id,
        completion_observer: Callable[[ConsultationState], None] | None = None,
    ) -> None:
        self._store = store
        self._guard = guard
        self._router = router
        self._new_id = id_factory
        self._completion_observer = completion_observer
        self._notified_visits: set[str] = set()

    def set_completion_observer(
        self, observer: Callable[[ConsultationState], None] | None,
    ) -> None:
        """Attach a non-blocking post-visit observer at the composition root.

        The observer receives a deep snapshot only after the state is persisted.
        It must schedule its own background work and return immediately.
        """
        self._completion_observer = observer

    async def start(self, *, patient_id: str | None = None) -> ConsultationState:
        """
        Begin a new consultation: a fresh visit and session, under either an
        existing patient (history preserved) or a new one. Persists and returns
        the blackboard so the caller has the new session id.
        """
        if patient_id is not None:
            patient = await self._store.patients.get(patient_id)
            if patient is None:
                patient = PatientRecord(patient_id=patient_id)
        else:
            patient = PatientRecord(patient_id=self._new_id())

        visit = VisitState(visit_id=self._new_id(), patient_id=patient.patient_id)
        session = SessionState(
            session_id=self._new_id(),
            patient_id=patient.patient_id,
            visit_id=visit.visit_id,
        )
        state = ConsultationState(patient=patient, visit=visit, session=session)
        await self._store.save(state)
        logger.info(
            "Started consultation session=%s visit=%s patient=%s",
            session.session_id, visit.visit_id, patient.patient_id,
        )
        return state

    async def send(self, *, session_id: str, message: str) -> TurnOutcome:
        """Process one patient turn end to end."""
        state = await self._store.load(session_id)
        if state is None:
            raise ConsultationError(f"Unknown session '{session_id}'.")

        # 1. Safety guard FIRST, on the current turn, before the router. The turn
        #    is not yet in the transcript (guard/classifier contract); it is
        #    appended after processing, in step 3.
        safety = await self._guard.check(message=message, state=state)
        state.session.last_safety = safety

        # 2. Any red-flag triage interrupts: bypass the router, return escalation
        #    advice. The path's phase is left untouched, so a later routine turn
        #    resumes the consultation where it paused.
        if self._is_escalation(safety.triage_level):
            reply = safety.advice or _GENERIC_ESCALATION
        else:
            result = await self._router.route_turn(turn=message, state=state)
            reply = result.reply
        route = state.session.active_route

        # 3. The orchestrator owns the transcript.
        state.session.add_message(MessageRole.PATIENT, message)
        state.session.add_message(MessageRole.DOCTOR_AI, reply, route=route)

        # 4. Persist all three tiers atomically.
        await self._store.save(state)

        # 5. A completed diagnostic visit may trigger background knowledge-gap
        #    curation. It is notified once and is outside the patient reply path.
        if (
            state.session.phase is DiagnosticPhase.COMPLETE
            and self._completion_observer is not None
            and state.visit.visit_id not in self._notified_visits
        ):
            self._notified_visits.add(state.visit.visit_id)
            try:
                self._completion_observer(state.model_copy(deep=True))
            except Exception:  # noqa: BLE001 - curation must never fail a consultation
                logger.exception(
                    "Completion observer failed for visit=%s", state.visit.visit_id
                )

        return TurnOutcome(
            reply=reply,
            session_id=session_id,
            triage_level=safety.triage_level,
            escalated=self._is_escalation(safety.triage_level),
            active_route=route,
            phase=state.session.phase,
            red_flags=tuple(safety.red_flags),
        )

    @staticmethod
    def _is_escalation(level: TriageLevel) -> bool:
        # Both urgent and emergency interrupt the consultation and surface care
        # advice. This is the cautious choice: when a red flag fires, telling the
        # patient to seek care takes priority over continuing the interview. A
        # deployment that prefers "urgent continues with a notice" would relax
        # this to `level is TriageLevel.EMERGENCY`.
        return level in (TriageLevel.EMERGENCY, TriageLevel.URGENT)

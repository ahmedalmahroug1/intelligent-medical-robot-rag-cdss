"""
app/persistence/store.py
------------------------
A thin facade over the three repositories for the most common access pattern:
load the whole blackboard at the start of a turn, save it at the end.

Why this exists (and why it's thin):
    The orchestrator's core loop is "load all three tiers → run agents → save all
    three tiers." Without a facade, every caller would re-implement that 3-step
    load/save with the correct foreign-key ordering. The facade puts that ordering
    in exactly one place. It is intentionally minimal — it composes the
    repositories through their public API and adds nothing else. The repositories
    remain individually accessible (`store.patients`, `store.visits`,
    `store.sessions`) for callers that only need one tier (e.g. the doctor-
    approval path touches only visits).

It does NOT create new records or assign IDs — that is orchestration logic that
belongs to the path/orchestrator layer. This facade only loads and saves.
"""

from __future__ import annotations

from app.core.state import ConsultationState
from app.persistence.database import Database
from app.persistence.repositories import (
    PatientRepository,
    SessionRepository,
    VisitRepository,
)


class PersistenceError(RuntimeError):
    """Raised when stored data violates an invariant we rely on."""


class ConsultationStore:
    """Loads and saves a full ConsultationState (all three tiers)."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self.patients = PatientRepository(db)
        self.visits = VisitRepository(db)
        self.sessions = SessionRepository(db)

    async def load(self, session_id: str) -> ConsultationState | None:
        """
        Reconstruct the blackboard from a session id.

        Returns None if the session does not exist. Raises PersistenceError if
        the session exists but its visit or patient is missing — that means
        referential integrity was violated, which should never happen while
        foreign-key enforcement is on, and we want it to fail loudly if it does.
        """
        session = await self.sessions.get(session_id)
        if session is None:
            return None

        visit = await self.visits.get(session.visit_id)
        patient = await self.patients.get(session.patient_id)
        if visit is None or patient is None:
            raise PersistenceError(
                f"Session '{session_id}' references missing "
                f"visit='{session.visit_id}' or patient='{session.patient_id}'."
            )

        return ConsultationState(patient=patient, visit=visit, session=session)

    async def save(self, state: ConsultationState) -> None:
        """
        Persist all three tiers atomically.

        The three writes run inside a single transaction: either all three are
        committed or, if any one fails, none are — the blackboard is never left
        partially persisted. This matters because the three tiers form one
        logical update per turn (a transcript message, the differential it drove,
        and the advanced phase), and a half-saved turn would break the audit
        chain between what was said and what was concluded.

        Order still matters for foreign keys (parents before children), and each
        write is an idempotent upsert, so re-saving an unchanged tier every turn
        remains harmless.
        """
        async with self._db.transaction():
            await self.patients.save(state.patient)
            await self.visits.save(state.visit)
            await self.sessions.save(state.session)

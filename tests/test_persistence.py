"""
tests/test_persistence.py
-------------------------
Tests for the persistence layer — File 2.

What these prove, and why each matters:
    1. Each tier saves and loads back equal to the original (lossless storage).
    2. Missing keys return None (not an error) — callers can branch on absence.
    3. ConsultationStore.save → load reconstructs the whole blackboard exactly
       → this is the load-at-turn-start / save-at-turn-end path the orchestrator
         depends on.
    4. list_for_patient is scoped and ordered → the follow-up / history path
       sees only that patient's visits, most recent first.
    5. Foreign-key enforcement rejects an orphaned visit → referential integrity
       (a session can never point at a patient/visit that doesn't exist).
    6. Saving the same key twice updates, never duplicates → idempotent upsert,
       which is what makes save-every-turn safe.
    7. Mutating then re-saving persists the change → the blackboard genuinely
       remembers across turns/visits.

Run:
    pip install -r requirements.txt          # pydantic + aiosqlite
    python -m tests.test_persistence         # standalone async runner
    # (pytest users: needs pytest-asyncio; the standalone runner needs nothing)

Tests use an in-memory database (":memory:") opened fresh per test, so they are
fully isolated and leave no files behind.
"""

from __future__ import annotations

import asyncio

from app.core.enums import DiagnosticPhase, MessageRole, Route, Sex
from app.core.state import (
    ConsultationState,
    Hypothesis,
    Medication,
    PatientRecord,
    SessionState,
    VisitState,
)
from app.persistence.database import Database
from app.persistence.store import ConsultationStore, PersistenceError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _fresh() -> tuple[Database, ConsultationStore]:
    """A connected in-memory database and a store over it."""
    db = Database(":memory:")
    await db.connect()
    return db, ConsultationStore(db)


def _patient(pid: str = "PT-001") -> PatientRecord:
    return PatientRecord(
        patient_id=pid,
        age=67,
        sex=Sex.MALE,
        chronic_conditions=["hypertension"],
        medications=[Medication(name="lisinopril", dose="10mg")],
    )


def _visit(vid: str = "V-001", pid: str = "PT-001") -> VisitState:
    return VisitState(
        visit_id=vid,
        patient_id=pid,
        route=Route.DIAGNOSIS,
        chief_complaint="chest pain",
    )


def _session(sid: str = "S-001", vid: str = "V-001", pid: str = "PT-001") -> SessionState:
    return SessionState(session_id=sid, patient_id=pid, visit_id=vid)


def _consultation() -> ConsultationState:
    return ConsultationState(patient=_patient(), visit=_visit(), session=_session())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-tier round-trips
# ─────────────────────────────────────────────────────────────────────────────

async def test_patient_save_and_get_round_trip():
    db, store = await _fresh()
    try:
        original = _patient()
        await store.patients.save(original)
        loaded = await store.patients.get("PT-001")
        assert loaded == original
    finally:
        await db.close()


async def test_visit_save_and_get_round_trip():
    db, store = await _fresh()
    try:
        await store.patients.save(_patient())          # FK parent first
        original = _visit()
        await store.visits.save(original)
        loaded = await store.visits.get("V-001")
        assert loaded == original
    finally:
        await db.close()


async def test_session_save_and_get_round_trip():
    db, store = await _fresh()
    try:
        await store.patients.save(_patient())
        await store.visits.save(_visit())
        original = _session()
        await store.sessions.save(original)
        loaded = await store.sessions.get("S-001")
        assert loaded == original
    finally:
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Absence returns None
# ─────────────────────────────────────────────────────────────────────────────

async def test_get_missing_returns_none():
    db, store = await _fresh()
    try:
        assert await store.patients.get("nope") is None
        assert await store.visits.get("nope") is None
        assert await store.sessions.get("nope") is None
        assert await store.patients.exists("nope") is False
    finally:
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Whole-blackboard save → load
# ─────────────────────────────────────────────────────────────────────────────

async def test_consultation_store_save_and_load():
    db, store = await _fresh()
    try:
        original = _consultation()
        await store.save(original)
        loaded = await store.load("S-001")
        assert loaded is not None
        assert loaded == original
    finally:
        await db.close()


async def test_consultation_load_missing_session_returns_none():
    db, store = await _fresh()
    try:
        assert await store.load("no-such-session") is None
    finally:
        await db.close()


async def test_consultation_load_with_broken_integrity_raises():
    # Save a session whose visit/patient exist, then prove that if a session
    # somehow references a missing parent, load() fails loudly. We simulate the
    # broken state by saving a session that points at a non-existent patient —
    # which the FK constraint itself should already block, so this also doubles
    # as a guard that we never silently return half a blackboard.
    db, store = await _fresh()
    try:
        await store.patients.save(_patient())
        await store.visits.save(_visit())
        await store.sessions.save(_session())
        # Tamper: directly delete the patient row out from under the session.
        await db.connection.execute("PRAGMA foreign_keys=OFF")
        await db.connection.execute("DELETE FROM patients WHERE patient_id = 'PT-001'")
        await db.connection.commit()

        raised = False
        try:
            await store.load("S-001")
        except PersistenceError:
            raised = True
        assert raised, "load() should raise PersistenceError on broken integrity"
    finally:
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scoped, ordered listing
# ─────────────────────────────────────────────────────────────────────────────

async def test_list_for_patient_is_scoped_and_ordered():
    db, store = await _fresh()
    try:
        await store.patients.save(_patient("PT-001"))
        await store.patients.save(_patient("PT-002"))

        # Two visits for PT-001 with distinct start times, one for PT-002.
        v_old = _visit("V-old", "PT-001")
        v_new = _visit("V-new", "PT-001")
        # Force a clear ordering by nudging started_at.
        v_new.started_at = v_old.started_at.replace(year=v_old.started_at.year + 1)
        other = _visit("V-other", "PT-002")

        await store.visits.save(v_old)
        await store.visits.save(v_new)
        await store.visits.save(other)

        visits = await store.visits.list_for_patient("PT-001")
        ids = [v.visit_id for v in visits]
        assert ids == ["V-new", "V-old"]        # scoped to PT-001, newest first
        assert "V-other" not in ids
    finally:
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Foreign-key enforcement
# ─────────────────────────────────────────────────────────────────────────────

async def test_orphan_visit_is_rejected():
    db, store = await _fresh()
    try:
        raised = False
        try:
            await store.visits.save(_visit("V-x", "GHOST-PATIENT"))  # no such patient
        except Exception:
            raised = True
        assert raised, "saving a visit for a non-existent patient should fail (FK)"
    finally:
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Upsert (no duplicates)
# ─────────────────────────────────────────────────────────────────────────────

async def test_save_twice_updates_not_duplicates():
    db, store = await _fresh()
    try:
        p = _patient()
        await store.patients.save(p)
        p.age = 68                       # change something
        await store.patients.save(p)     # same key again

        loaded = await store.patients.get("PT-001")
        assert loaded.age == 68

        cursor = await db.connection.execute("SELECT COUNT(*) AS n FROM patients")
        row = await cursor.fetchone()
        await cursor.close()
        assert row["n"] == 1             # updated in place, not duplicated
    finally:
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Mutation persists across save/reload (memory really works)
# ─────────────────────────────────────────────────────────────────────────────

async def test_mutation_persists_across_reload():
    db, store = await _fresh()
    try:
        c = _consultation()
        await store.save(c)

        # Simulate a turn: advance phase, add a message, grow the differential.
        c.session.phase = DiagnosticPhase.INTERVIEW
        c.session.interview_iterations = 1
        c.session.add_message(MessageRole.PATIENT, "it started this morning", Route.DIAGNOSIS)
        c.visit.differential = [Hypothesis(condition="angina", confidence=0.5)]
        await store.save(c)

        reloaded = await store.load("S-001")
        assert reloaded is not None
        assert reloaded.session.phase is DiagnosticPhase.INTERVIEW
        assert reloaded.session.interview_iterations == 1
        assert len(reloaded.session.transcript) == 1
        assert reloaded.visit.differential[0].condition == "angina"
    finally:
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Transaction semantics (atomicity)
# ─────────────────────────────────────────────────────────────────────────────

async def test_transaction_commits_on_success():
    db, store = await _fresh()
    try:
        async with db.transaction():
            await store.patients.save(_patient())
        # After the block exits cleanly, the write is durable.
        assert await store.patients.get("PT-001") is not None
    finally:
        await db.close()


async def test_transaction_rolls_back_on_error():
    db, store = await _fresh()
    try:
        raised = False
        try:
            async with db.transaction():
                await store.patients.save(_patient())   # staged, not yet committed
                raise RuntimeError("boom")              # forces a rollback
        except RuntimeError:
            raised = True
        assert raised
        # The staged write was undone — nothing persisted.
        assert await store.patients.get("PT-001") is None
    finally:
        await db.close()


async def test_facade_save_is_atomic_on_failure():
    # This is the exact concern: if a later tier's save fails, the earlier tiers
    # must not remain persisted. We force the session save to fail mid-facade and
    # assert that patient and visit were rolled back.
    db, store = await _fresh()
    try:
        async def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated session-save failure")

        store.sessions.save = _boom  # type: ignore[method-assign]

        raised = False
        try:
            await store.save(_consultation())
        except RuntimeError:
            raised = True
        assert raised

        # Neither parent tier should survive the failed transaction.
        assert await store.patients.get("PT-001") is None, "patient should have rolled back"
        assert await store.visits.get("V-001") is None, "visit should have rolled back"
    finally:
        await db.close()


async def test_nested_transaction_defers_commit_to_outer():
    # An inner transaction() must not commit early; the outer block owns it.
    db, store = await _fresh()
    try:
        raised = False
        try:
            async with db.transaction():
                await store.patients.save(_patient())
                async with db.transaction():            # nested — joins outer
                    await store.patients.save(_patient("PT-002"))
                raise RuntimeError("boom after nested block")
        except RuntimeError:
            raised = True
        assert raised
        # Because the inner block deferred to the outer, BOTH writes roll back.
        assert await store.patients.get("PT-001") is None
        assert await store.patients.get("PT-002") is None
    finally:
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone async runner (so `python -m tests.test_persistence` needs no pytest)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            asyncio.run(t())
        except Exception:
            failed += 1
            print(f"[FAIL]  {t.__name__}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[PASS]  {t.__name__}")

    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)

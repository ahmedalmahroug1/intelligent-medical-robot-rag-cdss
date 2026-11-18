"""
app/persistence/repositories.py
-------------------------------
One repository per state tier. Each maps its Pydantic model to/from its table
and exposes only the operations that tier actually needs.

Why three focused repositories (not one big DAO):
    Each tier has a different lifetime and different query needs. Keeping them
    separate means the doctor-approval path can depend on VisitRepository alone,
    a history view on PatientRepository alone, and so on — high cohesion, low
    coupling. The shared SQL mechanics live in two small module-level helpers so
    there is no copy-pasted upsert logic and no inheritance hierarchy to reason
    about (plain functions over a base class — simpler, and enough here).

Serialization:
    Save  → model.model_dump_json() into the `data` column (+ extracted columns).
    Load  → Model.model_validate_json(row["data"]).
    This reuses exactly the round-trip proven in the File 1 tests.

Note on SQL string formatting:
    Table and primary-key names are interpolated into SQL with f-strings. This is
    safe because they are internal constants defined in this module, never user
    input. All actual values are passed as bound parameters (?), never formatted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiosqlite

from app.core.state import PatientRecord, SessionState, VisitState
from app.persistence.database import Database


def _utcnow_iso() -> str:
    """DB write timestamp as an ISO 8601 string (sorts lexicographically)."""
    return datetime.now(timezone.utc).isoformat()


async def _upsert(
    db: Database,
    table: str,
    pk: str,
    row: dict[str, Any],
) -> None:
    """
    Insert a row, or update it in place if the primary key already exists.

    `row` maps column name → value and must include the primary-key column.
    Idempotent: saving the same key twice updates rather than duplicating.

    Commits only when NOT inside a `db.transaction()` block. A standalone save
    is durable on its own; a save running inside a transaction defers its commit
    to the transaction boundary so the whole block is atomic.
    """
    cols = list(row.keys())
    col_list = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != pk)
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT({pk}) DO UPDATE SET {updates}"
    )
    await db.connection.execute(sql, [row[c] for c in cols])
    if not db.in_transaction:
        await db.connection.commit()


async def _fetch_one(
    conn: aiosqlite.Connection,
    table: str,
    pk: str,
    key: str,
) -> aiosqlite.Row | None:
    """Fetch a single row by primary key, or None if it does not exist."""
    cursor = await conn.execute(f"SELECT * FROM {table} WHERE {pk} = ?", (key,))
    row = await cursor.fetchone()
    await cursor.close()
    return row


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — PatientRecord
# ─────────────────────────────────────────────────────────────────────────────

class PatientRepository:
    """Persistence for the permanent patient record."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, record: PatientRecord) -> None:
        await _upsert(
            self._db,
            table="patients",
            pk="patient_id",
            row={
                "patient_id": record.patient_id,
                "data": record.model_dump_json(),
                "saved_at": _utcnow_iso(),
            },
        )

    async def get(self, patient_id: str) -> PatientRecord | None:
        row = await _fetch_one(self._db.connection, "patients", "patient_id", patient_id)
        if row is None:
            return None
        return PatientRecord.model_validate_json(row["data"])

    async def exists(self, patient_id: str) -> bool:
        row = await _fetch_one(self._db.connection, "patients", "patient_id", patient_id)
        return row is not None

    async def list_all(self) -> list[PatientRecord]:
        """Every patient record, most recently saved first. Used by the doctor dashboard."""
        cursor = await self._db.connection.execute(
            "SELECT data FROM patients ORDER BY saved_at DESC")
        rows = await cursor.fetchall()
        await cursor.close()
        return [PatientRecord.model_validate_json(r["data"]) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — VisitState
# ─────────────────────────────────────────────────────────────────────────────

class VisitRepository:
    """Persistence for per-encounter visit state."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, visit: VisitState) -> None:
        await _upsert(
            self._db,
            table="visits",
            pk="visit_id",
            row={
                "visit_id": visit.visit_id,
                "patient_id": visit.patient_id,
                "started_at": visit.started_at.isoformat(),
                "saved_at": _utcnow_iso(),
                "data": visit.model_dump_json(),
            },
        )

    async def get(self, visit_id: str) -> VisitState | None:
        row = await _fetch_one(self._db.connection, "visits", "visit_id", visit_id)
        if row is None:
            return None
        return VisitState.model_validate_json(row["data"])

    async def list_for_patient(self, patient_id: str) -> list[VisitState]:
        """
        All visits for a patient, most recent first.

        Used by the follow-up path and the longitudinal history view to see what
        happened on previous encounters (e.g. a patient returning with a lab
        result the system asked for).
        """
        conn = self._db.connection
        cursor = await conn.execute(
            "SELECT data FROM visits WHERE patient_id = ? ORDER BY started_at DESC",
            (patient_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [VisitState.model_validate_json(r["data"]) for r in rows]

    async def list_all(self, *, limit: int = 500) -> list[VisitState]:
        """Every visit, most recent first (capped). The doctor dashboard's review queue."""
        cursor = await self._db.connection.execute(
            "SELECT data FROM visits ORDER BY started_at DESC LIMIT ?", (limit,))
        rows = await cursor.fetchall()
        await cursor.close()
        return [VisitState.model_validate_json(r["data"]) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — SessionState
# ─────────────────────────────────────────────────────────────────────────────

class SessionRepository:
    """Persistence for per-conversation session state."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, session: SessionState) -> None:
        await _upsert(
            self._db,
            table="sessions",
            pk="session_id",
            row={
                "session_id": session.session_id,
                "visit_id": session.visit_id,
                "patient_id": session.patient_id,
                "started_at": session.started_at.isoformat(),
                "saved_at": _utcnow_iso(),
                "data": session.model_dump_json(),
            },
        )

    async def get(self, session_id: str) -> SessionState | None:
        row = await _fetch_one(self._db.connection, "sessions", "session_id", session_id)
        if row is None:
            return None
        return SessionState.model_validate_json(row["data"])

    async def list_for_visit(self, visit_id: str) -> list[SessionState]:
        """All chat sessions belonging to a visit, most recent first."""
        conn = self._db.connection
        cursor = await conn.execute(
            "SELECT data FROM sessions WHERE visit_id = ? ORDER BY started_at DESC",
            (visit_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [SessionState.model_validate_json(r["data"]) for r in rows]

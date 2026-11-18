"""
app/persistence/database.py
---------------------------
Owns the single aiosqlite connection and the schema.

Responsibilities (and nothing else):
    - open / close the connection,
    - apply the pragmas we depend on,
    - create the schema if it does not exist,
    - provide a transaction boundary so multi-row writes are atomic.

It deliberately knows nothing about the domain models. Mapping models to rows
lives in the repositories; this class is pure connection + DDL. That keeps the
coupling one-directional: repositories depend on Database, Database depends on
nobody.

Storage strategy:
    Each of the three state tiers is stored as one row holding a JSON document
    (the losslessly-serialised Pydantic model) plus a few EXTRACTED columns used
    only for lookups and ordering (foreign keys + a timestamp). We do not
    normalise the nested structures into separate tables: the access patterns
    are key-based, the documents are deeply nested and still evolving, and we
    already have a tested JSON round-trip. This gives lossless storage with
    minimal coupling between the DB schema and the domain model.

Pragmas:
    - journal_mode=WAL  → better read/write concurrency (no-op for :memory:).
    - foreign_keys=ON   → enforce referential integrity (session → visit →
                          patient). Must be set per-connection.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite


# Schema is idempotent (IF NOT EXISTS). For a student project this is sufficient;
# a production system would use versioned migrations instead — noted as future
# work rather than built now.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    data       TEXT NOT NULL,          -- full PatientRecord as JSON
    saved_at   TEXT NOT NULL           -- DB write time (ISO 8601)
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id   TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    started_at TEXT NOT NULL,          -- from the model, for chronological order
    saved_at   TEXT NOT NULL,
    data       TEXT NOT NULL,          -- full VisitState as JSON
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
CREATE INDEX IF NOT EXISTS idx_visits_patient ON visits(patient_id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    visit_id   TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    saved_at   TEXT NOT NULL,
    data       TEXT NOT NULL,          -- full SessionState as JSON
    FOREIGN KEY (visit_id)   REFERENCES visits(visit_id),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_visit ON sessions(visit_id);
"""


class Database:
    """
    Thin wrapper around one aiosqlite connection.

    Lifecycle:
        db = Database("medical_ai.db")
        await db.connect()      # opens connection, sets pragmas, creates schema
        ...                     # repositories use db.connection
        await db.close()

    In the API layer this will be driven by FastAPI's startup/shutdown lifespan.
    A single shared connection is sufficient for this system's load; aiosqlite
    serialises operations on it safely.
    """

    def __init__(self, path: str = "medical_ai.db") -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        # True while a transaction() block is active. Repositories consult this
        # to decide whether to commit themselves: a standalone save commits, but
        # a save running inside a transaction leaves the commit to the boundary.
        self._in_transaction = False

    @property
    def connection(self) -> aiosqlite.Connection:
        """The live connection. Raises if connect() has not been called."""
        if self._conn is None:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self._conn

    @property
    def in_transaction(self) -> bool:
        """Whether a transaction() block is currently active on this database."""
        return self._in_transaction

    async def connect(self) -> None:
        """Open the connection, apply pragmas, and ensure the schema exists."""
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row          # dict-like row access
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(_SCHEMA_SQL)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the connection if open. Safe to call more than once."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """
        Run a block of writes as one atomic unit: commit on success, roll back
        on any exception.

        Usage:
            async with db.transaction():
                await repo_a.save(...)
                await repo_b.save(...)
            # both committed together, or neither if an error was raised

        Behaviour:
            - Sets `in_transaction` for the duration, so repository saves inside
              the block do NOT commit individually — the single commit here is
              what makes the whole block atomic.
            - Nested use is safe: an inner transaction() simply joins the outer
              one and lets the outermost block own the commit/rollback. (SQLite
              does not support truly nested transactions on one connection; this
              avoids a premature inner commit.)

        Note on concurrency: this provides atomicity, not write concurrency. The
        system uses a single shared connection, so write transactions run
        sequentially. Supporting concurrent write transactions would require a
        connection-per-transaction model — deliberately out of scope here.
        """
        conn = self.connection

        # Already inside a transaction → just participate; the outer block commits.
        if self._in_transaction:
            yield conn
            return

        self._in_transaction = True
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            self._in_transaction = False

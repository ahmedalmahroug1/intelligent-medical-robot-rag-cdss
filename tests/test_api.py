"""
tests/test_api.py
-----------------
Tests for File 8b (the FastAPI + SSE layer). Run from the project root:

    python -m tests.test_api

Needs fastapi + httpx + sse-starlette + pydantic + aiosqlite (your env has them).
These drive the real routes through httpx's in-process ASGI transport (no server,
no network) against a bare app with a FAKE service injected — so no OpenAI key is
needed. The routes module is imported directly; the OpenAI-backed app.py is not.

What they prove:
    - POST /consultations starts a session and returns its ids;
    - POST /consultations/{id}/messages routes a turn and returns the reply;
    - an emergency message surfaces escalated=True through the HTTP boundary
      (the guard's router-bypass is visible end to end);
    - an unknown session returns 404; an empty message returns 422;
    - the SSE endpoint emits start/phase/message/done events carrying the reply.
"""

from __future__ import annotations

import asyncio
import inspect
import traceback
from typing import Any, AsyncIterator, Sequence

import httpx
from fastapi import FastAPI

from app.api.routes import router
from app.core.enums import Route
from app.core.state import ConsultationState
from app.llm.base import LLMGateway, LLMMessage, ModelTier
from app.orchestration.service import ConsultationService
from app.persistence.database import Database
from app.persistence.store import ConsultationStore
from app.routing import Path, PathResult, Router, Transition, TurnClassifier
from app.safety.guard import SafetyGuard


# ─────────────────────────────────────────────────────────────────────────────
# Test doubles + app wiring
# ─────────────────────────────────────────────────────────────────────────────

class RoutineGateway(LLMGateway):
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
    async def classify(self, *, turn, state, available) -> Route:
        return Route.DIAGNOSIS


class RecordingPath(Path):
    route = Route.DIAGNOSIS

    def __init__(self) -> None:
        self.turns: list[str] = []

    async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
        self.turns.append(turn)
        return PathResult(reply=f"reply to: {turn}", transition=Transition.stay())


async def _app_and_db() -> tuple[FastAPI, Database, RecordingPath]:
    db = Database(":memory:")
    await db.connect()
    gateway = RoutineGateway()
    path = RecordingPath()
    router_ = Router(FixedClassifier())
    router_.register(path)
    service = ConsultationService(
        store=ConsultationStore(db), guard=SafetyGuard(gateway), router=router_
    )
    app = FastAPI()
    app.state.service = service          # inject directly; no lifespan needed
    app.include_router(router)
    return app, db, path


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _start(client: httpx.AsyncClient) -> str:
    resp = await client.post("/consultations", json={})
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_start_returns_ids() -> None:
    app, db, _ = await _app_and_db()
    client = _client(app)
    try:
        resp = await client.post("/consultations", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] and body["visit_id"] and body["patient_id"]
    finally:
        await client.aclose()
        await db.close()


async def test_post_message_routes_and_replies() -> None:
    app, db, path = await _app_and_db()
    client = _client(app)
    try:
        sid = await _start(client)
        resp = await client.post(f"/consultations/{sid}/messages", json={"message": "my head hurts"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["reply"] == "reply to: my head hurts"
        assert body["escalated"] is False
        assert body["active_route"] == "diagnosis"
        assert path.turns == ["my head hurts"]
    finally:
        await client.aclose()
        await db.close()


async def test_emergency_escalates_through_http() -> None:
    app, db, path = await _app_and_db()
    client = _client(app)
    try:
        sid = await _start(client)
        resp = await client.post(
            f"/consultations/{sid}/messages",
            json={"message": "crushing chest pain spreading to my arm"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["escalated"] is True
        assert body["triage_level"] == "emergency"
        assert path.turns == [], "router/path must be bypassed on an emergency"
        assert "emergency" in body["reply"].lower()
    finally:
        await client.aclose()
        await db.close()


async def test_unknown_session_returns_404() -> None:
    app, db, _ = await _app_and_db()
    client = _client(app)
    try:
        resp = await client.post("/consultations/nope/messages", json={"message": "hi"})
        assert resp.status_code == 404, resp.text
    finally:
        await client.aclose()
        await db.close()


async def test_empty_message_returns_422() -> None:
    app, db, _ = await _app_and_db()
    client = _client(app)
    try:
        sid = await _start(client)
        resp = await client.post(f"/consultations/{sid}/messages", json={"message": ""})
        assert resp.status_code == 422, resp.text
    finally:
        await client.aclose()
        await db.close()


async def test_stream_emits_events_with_reply() -> None:
    app, db, _ = await _app_and_db()
    client = _client(app)
    try:
        sid = await _start(client)
        async with client.stream(
            "POST", f"/consultations/{sid}/messages/stream", json={"message": "hello there"}
        ) as resp:
            assert resp.status_code == 200, resp.text
            body = (await resp.aread()).decode()
        # discrete SSE events, with the reply carried in the message event
        assert "event: start" in body
        assert "event: phase" in body
        assert "event: message" in body
        assert "event: done" in body
        assert "reply to: hello there" in body
    finally:
        await client.aclose()
        await db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_start_returns_ids,
    test_post_message_routes_and_replies,
    test_emergency_escalates_through_http,
    test_unknown_session_returns_404,
    test_empty_message_returns_422,
    test_stream_emits_events_with_reply,
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

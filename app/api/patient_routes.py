"""
app/api/patient_routes.py
-------------------------
The REMOTE-PATIENT web channel — the Concierge over HTTP, so a patient on a website (independent
of the robot) gets the SAME bilingual conversation owner as the robot, and everything they say is
saved to the same database the doctor reviews.

Patients sign in with a username + password (the web equivalent of the robot's face recognition):
`register` creates an account bound to a stable `patient_id`; `login` returns a token. Starting a
consultation requires that token, so the conversation is tied to a known patient whose record is
loaded and preloaded into the Concierge (it greets them back and does not re-ask known facts).

The Concierge agent is stateful in memory across a conversation, so a small in-memory session
registry keeps one `ConciergeAgent` per active session. When the interview completes, the visit is
persisted through the SAME save wire the robot uses (`persist_concierge_visit`). The conversation
never blocks on doctor approval.

Demo note: the session registry and login tokens are in-memory (not bounded, not shared across
processes, lost on restart); a real deployment needs persistent sessions and TLS.
"""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.concierge import prompts
from app.concierge.accounts import PatientAccountRegistry
from app.concierge.agent import ConciergeAgent
from app.concierge.identity import IdentityRegistry
from app.concierge.reasoning import ReasoningFacade
from app.concierge.schemas import ClinicalPicture
from app.concierge.visit_writer import persist_concierge_visit
from app.core.enums import Sex
from app.core.state import PatientRecord
from app.llm.base import LLMGateway
from app.persistence.store import ConsultationStore

if TYPE_CHECKING:
    from app.knowledge.agent import KnowledgeCuratorAgent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/patient", tags=["patient"])


def _picture_from_record(record: PatientRecord) -> ClinicalPicture:
    """Seed a ClinicalPicture from a known record, so the engine is not re-asked what we know
    (mirrors robot/run_robot.py:preload_picture)."""
    pic = ClinicalPicture()
    pic.age = record.age
    if record.sex in (Sex.MALE, Sex.FEMALE):
        pic.sex = record.sex.value
    pic.medications = [m.name for m in record.medications if m.name]
    pic.allergies = [a.substance for a in record.allergies if a.substance]
    pic.chronic_conditions = list(record.chronic_conditions)
    return pic


class ConciergeRuntime:
    """Shared Concierge collaborators + patient accounts + per-session agents for the web channel."""

    def __init__(
        self, *, facade: ReasoningFacade, gateway: LLMGateway, store: ConsultationStore,
        accounts: PatientAccountRegistry | None = None,
        identity: IdentityRegistry | None = None,
        knowledge_agent: KnowledgeCuratorAgent | None = None,
        safety_check: Callable[[str, ClinicalPicture], Awaitable[str | None]] | None = None,
    ) -> None:
        self._facade = facade
        self._gateway = gateway
        self._store = store
        self._accounts = accounts or PatientAccountRegistry()
        self._identity = identity or IdentityRegistry()
        self._knowledge_agent = knowledge_agent
        self._safety_check = safety_check
        self._agents: dict[str, ConciergeAgent] = {}
        self._tokens: dict[str, tuple[str, str]] = {}     # token -> (patient_id, name)

    def set_knowledge_agent(self, agent: KnowledgeCuratorAgent | None) -> None:
        self._knowledge_agent = agent

    # ── auth ──────────────────────────────────────────────────────────────────
    def _issue_token(self, patient_id: str, name: str) -> dict:
        token = uuid.uuid4().hex
        self._tokens[token] = (patient_id, name)
        return {"token": token, "patient_id": patient_id, "name": name}

    async def register(self, username: str, password: str, name: str) -> dict:
        """Create an account (+ its PatientRecord) and return a session token. Raises ValueError
        if the username is taken or the input is empty."""
        patient_id = self._accounts.register(username, password, name)
        if await self._store.patients.get(patient_id) is None:
            await self._store.patients.save(PatientRecord(patient_id=patient_id))
        return self._issue_token(patient_id, name or username)

    def login(self, username: str, password: str) -> dict | None:
        patient_id = self._accounts.authenticate(username, password)
        if patient_id is None:
            return None
        return self._issue_token(patient_id, self._accounts.name_for_username(username))

    # ── conversation ──────────────────────────────────────────────────────────
    async def start(self, token: str) -> dict:
        """Open a session for the authenticated patient: load their record, preload the Concierge,
        and greet them (back, if they have a history)."""
        who = self._tokens.get(token)
        if who is None:
            raise PermissionError("invalid or expired token")
        patient_id, name = who
        record = await self._store.patients.get(patient_id)
        picture = _picture_from_record(record) if record else None
        # A returning patient continues in the language SAVED on their record, locked from the
        # first word; a patient with no saved language is ASKED (bilingually) and their answer
        # locks it for this and every later visit.
        saved_language = (record.preferred_language if record else "") or None
        agent = ConciergeAgent(
            facade=self._facade,
            gateway=self._gateway,
            identity=self._identity,
            safety_check=self._safety_check,
        )
        agent.preload(name=name, patient_id=patient_id, picture=picture,
                      language=saved_language)
        session_id = uuid.uuid4().hex
        self._agents[session_id] = agent
        returning = bool(record and record.visit_history)
        opening = (prompts.opening(name, saved_language, returning=returning)
                   if saved_language else prompts.ask_language(name))
        return {"session_id": session_id, "reply": opening}

    async def send(self, session_id: str, message: str) -> tuple[str, bool, str | None]:
        """One turn. On completion, persist the full visit for the doctor (write-only, never
        blocks). Returns (reply, complete, visit_id-if-just-saved)."""
        agent = self._agents.get(session_id)
        if agent is None:
            raise KeyError(session_id)
        reply = await agent.respond(message)
        snap = agent.snapshot()
        visit_id = None
        if snap.complete:
            visit_id = await persist_concierge_visit(
                snap, store=self._store, facade=self._facade)
            if visit_id and self._knowledge_agent is not None:
                self._knowledge_agent.submit_visit(self._store, visit_id)
        return reply, snap.complete, visit_id

    def send_stream(self, session_id: str, message: str) -> AsyncIterator[str]:
        """One turn, STREAMED: newline-delimited JSON — `{"delta": ...}` per generated chunk,
        then one final `{"done": true, "complete": ..., "visit_id": ...}`. The patient reads the
        reply as it is generated instead of waiting for the whole turn (the web latency fix).
        Persistence happens exactly as in `send`, after the reply finished. Raises KeyError for
        an unknown session BEFORE any byte is streamed, so the route can still answer 404."""
        agent = self._agents.get(session_id)
        if agent is None:
            raise KeyError(session_id)

        async def gen() -> AsyncIterator[str]:
            async for chunk in agent.respond_stream(message):
                yield json.dumps({"delta": chunk}, ensure_ascii=False) + "\n"
            snap = agent.snapshot()
            visit_id = None
            if snap.complete:
                visit_id = await persist_concierge_visit(
                    snap, store=self._store, facade=self._facade)
                if visit_id and self._knowledge_agent is not None:
                    self._knowledge_agent.submit_visit(self._store, visit_id)
            yield json.dumps(
                {"done": True, "complete": snap.complete, "visit_id": visit_id},
                ensure_ascii=False) + "\n"

        return gen()


def _runtime(request: Request) -> ConciergeRuntime:
    rt = getattr(request.app.state, "concierge", None)
    if rt is None:
        raise HTTPException(status_code=503, detail="Concierge runtime not initialised.")
    return rt


class AuthRequest(BaseModel):
    username: str
    password: str
    name: str | None = None


class MessageRequest(BaseModel):
    message: str


@router.post("/register")
async def register(body: AuthRequest, request: Request) -> dict:
    try:
        return await _runtime(request).register(
            body.username, body.password, body.name or body.username)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/login")
async def login(body: AuthRequest, request: Request) -> dict:
    result = _runtime(request).login(body.username, body.password)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return result


@router.post("/consultations")
async def start_consultation(
    request: Request, x_patient_token: str | None = Header(default=None),
) -> dict:
    if not x_patient_token:
        raise HTTPException(status_code=401, detail="Missing patient token; please log in.")
    try:
        return await _runtime(request).start(x_patient_token)
    except PermissionError:
        raise HTTPException(status_code=401, detail="Invalid or expired session; please log in.")


@router.post("/consultations/{session_id}/messages")
async def post_message(session_id: str, body: MessageRequest, request: Request) -> dict:
    try:
        reply, complete, visit_id = await _runtime(request).send(session_id, body.message)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.")
    return {"reply": reply, "complete": complete, "visit_id": visit_id}


@router.post("/consultations/{session_id}/messages/stream")
async def post_message_stream(session_id: str, body: MessageRequest, request: Request):
    """Streaming variant of `post_message` (NDJSON). The non-streaming endpoint stays: the
    tests, the robot channel, and any older client keep working unchanged."""
    try:
        stream = _runtime(request).send_stream(session_id, body.message)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.")
    return StreamingResponse(stream, media_type="application/x-ndjson")

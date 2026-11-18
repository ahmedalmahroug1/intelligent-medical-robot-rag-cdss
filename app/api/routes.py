"""
app/api/routes.py
-----------------
The HTTP endpoints. Every handler is a thin delegate to the ConsultationService
on `app.state.service` — no orchestration logic lives here, so the routes can be
mounted on a real app (with the lifespan that builds the OpenAI-backed service)
or on a bare app with a fake service injected, which is how they are tested.

Endpoints:
    POST /consultations                          → start a consultation
    POST /consultations/{id}/messages            → one turn (JSON response)
    POST /consultations/{id}/messages/stream     → one turn (SSE event stream)

On streaming and the structured-call reality: the per-turn reply is computed by
the service in one shot (the interview call returns structured JSON, which cannot
be streamed as patient text — see the gateway notes). So the SSE stream emits
discrete progress/result EVENTS (start → [safety] → phase → message → done)
rather than token-by-token text. It still opens the connection immediately and
keeps the client informed; finer-grained progress (e.g. per milestone node) would
require the service/milestone to expose progress hooks, which is future work.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    MessageRequest,
    MessageResponse,
    StartRequest,
    StartResponse,
)
from app.orchestration.service import (
    ConsultationError,
    ConsultationService,
    TurnOutcome,
)

router = APIRouter()


def _service(request: Request) -> ConsultationService:
    service = getattr(request.app.state, "service", None)
    if service is None:  # lifespan did not run / wiring missing
        raise HTTPException(status_code=503, detail="Service not initialised.")
    return service


def _to_response(outcome: TurnOutcome) -> MessageResponse:
    return MessageResponse(
        reply=outcome.reply,
        session_id=outcome.session_id,
        triage_level=outcome.triage_level.value,
        escalated=outcome.escalated,
        active_route=outcome.active_route.value,
        phase=outcome.phase.value,
        red_flags=list(outcome.red_flags),
    )


@router.post("/consultations", response_model=StartResponse)
async def start_consultation(body: StartRequest, request: Request) -> StartResponse:
    state = await _service(request).start(patient_id=body.patient_id)
    return StartResponse(
        session_id=state.session.session_id,
        visit_id=state.visit.visit_id,
        patient_id=state.patient.patient_id,
    )


@router.post("/consultations/{session_id}/messages", response_model=MessageResponse)
async def post_message(
    session_id: str, body: MessageRequest, request: Request
) -> MessageResponse:
    try:
        outcome = await _service(request).send(session_id=session_id, message=body.message)
    except ConsultationError:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.")
    return _to_response(outcome)


@router.post("/consultations/{session_id}/messages/stream")
async def stream_message(session_id: str, body: MessageRequest, request: Request):
    service = _service(request)

    async def events():
        yield {"event": "start", "data": json.dumps({"session_id": session_id})}
        try:
            outcome = await service.send(session_id=session_id, message=body.message)
        except ConsultationError:
            yield {"event": "error", "data": json.dumps({"detail": "unknown session"})}
            return
        if outcome.escalated:
            yield {
                "event": "safety",
                "data": json.dumps({
                    "triage_level": outcome.triage_level.value,
                    "red_flags": list(outcome.red_flags),
                }),
            }
        yield {
            "event": "phase",
            "data": json.dumps({
                "phase": outcome.phase.value,
                "active_route": outcome.active_route.value,
            }),
        }
        yield {"event": "message", "data": json.dumps({"reply": outcome.reply})}
        yield {"event": "done", "data": json.dumps({"escalated": outcome.escalated})}

    return EventSourceResponse(events())

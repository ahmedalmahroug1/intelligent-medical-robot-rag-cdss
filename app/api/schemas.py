"""
app/api/schemas.py
------------------
Request/response models for the HTTP boundary. These are intentionally separate
from the domain/blackboard models: the wire format is a transport concern and
should be free to differ from the internal state (which is richer and mutable).
Enum-valued fields are exposed as plain strings here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    """Begin a consultation. Supply an existing patient_id to continue that
    patient's history into a new visit; omit it to create a new patient."""
    patient_id: str | None = None


class StartResponse(BaseModel):
    session_id: str
    visit_id: str
    patient_id: str


class MessageRequest(BaseModel):
    message: str = Field(min_length=1)


class MessageResponse(BaseModel):
    reply: str
    session_id: str
    triage_level: str
    escalated: bool
    active_route: str
    phase: str
    red_flags: list[str] = Field(default_factory=list)

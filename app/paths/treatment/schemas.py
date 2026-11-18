"""
app/paths/treatment/schemas.py
------------------------------
The structured-output contract for the TREATMENT path's single LLM call.

`TreatmentResult` is a transient parse target (a plain BaseModel, like the
diagnosis path's `IntakeResult`): it is constructed from the model's JSON each
turn, read by the path to compose a deterministic, safety-checked reply, and never
persisted as-is. All fields default, so an omitted key parses to a safe value
(notably `condition=""`, which makes the path ask for clarification rather than
guess, and `new_symptoms=False`, so a missing flag never forces a route switch).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProposedMedication(BaseModel):
    """One medication OPTION the guidance mentions. `name` is what the path's
    deterministic safety overlay cross-checks against the patient's allergies and
    active medications — never trusting the model to have caught a conflict."""
    name: str = ""
    purpose: str = ""
    caution: str = ""


class TreatmentResult(BaseModel):
    """Management guidance for an already-known condition. Not a prescription."""
    condition: str = ""                  # empty → the path asks which condition
    summary: str = ""                    # one-paragraph management overview
    self_care: list[str] = Field(default_factory=list)
    medication_options: list[ProposedMedication] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)   # when to seek urgent care
    clinician_note: str = ""
    # The model's read on whether the patient is actually describing NEW symptoms
    # (a diagnosis question), in which case the path hands back to the router.
    new_symptoms: bool = False

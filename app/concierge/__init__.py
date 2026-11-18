"""
app/concierge
-------------
The bilingual (Arabic/English) front-desk "concierge" layer that sits ABOVE the CDSS as the
single patient-facing persona: it identifies the patient, extracts a structured intake,
enriches the patient record, relays to the guard-protected CDSS, and translates the reply.
Informational reception only — it never diagnoses, advises, or reassures.

Public surface:
    ConciergeAgent     — the front-desk agent (wraps a ConsultationService)
    IntakeExtraction   — the structured intake contract
    IdentityRegistry   — name -> patient_id registry (Stage 0)
"""

from app.concierge.agent import ConciergeAgent
from app.concierge.identity import IdentityRegistry
from app.concierge.reasoning import ReasoningFacade
from app.concierge.safety import ConciergeSafety
from app.concierge.schemas import AssessResult, ClinicalPicture, IntakeExtraction

__all__ = [
    "ConciergeAgent", "ReasoningFacade", "ConciergeSafety", "IdentityRegistry",
    "IntakeExtraction", "ClinicalPicture", "AssessResult",
]

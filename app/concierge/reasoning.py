"""
app/concierge/reasoning.py
--------------------------
`ReasoningFacade` — conversation-free access to the CDSS medical reasoning. The concierge OWNS
the conversation; this facade is how it CONSULTS the CDSS as a reasoning engine. It does NOT run
the SafetyGuard, the router, or `service.send()` — it calls the model directly with the structured
clinical picture and REUSES the SAME prompt + reasoning call the DiagnosisPath uses
(`reason_over_context` + `DIAGNOSIS_SYSTEM_PROMPT`), so the differential, the suggested question,
and the sufficiency judgement come from the CDSS's SINGLE reasoning source — no second prompt, no
duplication, no new medical logic.

Stage A exposes `assess`. Stage B will add `recommend_tests`, `interpret_labs`, and `finalize`
(which runs the existing `Milestone`) — pure wiring of capabilities that already exist.
"""
from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from app.concierge.schemas import AssessResult, ClinicalPicture, FinalizeResult
from app.config import settings
from app.core.enums import MessageRole, ReportStatus, Sex
from app.core.state import (
    Allergy, ConsultationState, Medication, PatientRecord, Prescription, SessionState,
    VisitState,
)
from app.llm.base import LLMError, LLMGateway, ModelTier


class _RxItem(BaseModel):
    medication: str = ""
    dose: str = ""
    frequency: str = ""
    rationale: str = ""


class _RxResult(BaseModel):
    """Parse target for the prescription step (transient, like IntakeResult)."""
    prescriptions: list[_RxItem] = Field(default_factory=list)


_PRESCRIBE_SYS = (
    "You propose INITIAL medications for a working diagnosis in a decision-support system. "
    "Every item is PENDING a clinician's approval and has no authority until a doctor signs it. "
    "Given the working diagnosis and the patient's record (age, sex, current medications, "
    "allergies, chronic conditions), propose the few first-line medications, each with a dose, a "
    "frequency, and a one-line rationale. Be conservative; if no medication is clearly warranted, "
    "return an empty list. Return ONLY JSON: "
    '{"prescriptions":[{"medication":"...","dose":"...","frequency":"...","rationale":"..."}]}'
)


def _render_rx(p: ClinicalPicture, condition: str) -> str:
    lines = [f"Working diagnosis: {condition}"]
    if p.age is not None:
        lines.append(f"Age: {p.age}")
    if p.sex:
        lines.append(f"Sex: {p.sex}")
    if p.medications:
        lines.append("Current medications: " + ", ".join(p.medications))
    if p.allergies:
        lines.append("Allergies: " + ", ".join(p.allergies))
    if p.chronic_conditions:
        lines.append("Chronic conditions: " + ", ".join(p.chronic_conditions))
    return "\n".join(lines) + "\n\nPropose initial medications (each pending doctor approval)."
from app.paths.diagnosis import Milestone
# Single-source reuse: the SAME prompt + reasoning call the DiagnosisPath uses.
from app.paths.diagnosis.path import _SYSTEM_PROMPT as DIAGNOSIS_SYSTEM_PROMPT
from app.paths.diagnosis.reasoning import reason_over_context


def _render_picture(p: ClinicalPicture) -> str:
    lines: list[str] = []
    if p.age is not None:
        lines.append(f"Age: {p.age}")
    if p.sex:
        lines.append(f"Sex: {p.sex}")
    if p.chief_complaint:
        lines.append(f"Chief complaint: {p.chief_complaint}")
    if p.symptoms:
        lines.append("Symptoms: " + ", ".join(p.symptoms))
    if p.duration:
        lines.append(f"Duration: {p.duration}")
    if p.severity:
        lines.append(f"Severity: {p.severity}")
    if p.location:
        lines.append(f"Location: {p.location}")
    if p.associated_symptoms:
        lines.append("Associated symptoms: " + ", ".join(p.associated_symptoms))
    if p.medications:
        lines.append("Medications: " + ", ".join(p.medications))
    if p.allergies:
        lines.append("Allergies: " + ", ".join(p.allergies))
    if p.chronic_conditions:
        lines.append("Chronic conditions: " + ", ".join(p.chronic_conditions))
    if p.family_history:
        lines.append("Family history: " + ", ".join(p.family_history))
    body = "\n".join(lines) or "  (very little information so far)"
    return ("Clinical picture:\n" + body + "\n\nProduce the differential, the single most "
            "informative next question, and your sufficiency judgement.")


class ReasoningFacade:
    """The CDSS reasoning engine, exposed conversation-free to the concierge."""

    def __init__(
        self, gateway: LLMGateway, *, model: ModelTier = ModelTier.REASONING,
        confidence_floor: float = 0.6, milestone: Milestone | None = None,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self.confidence_floor = confidence_floor
        # The EXISTING run-once pipeline (differential → tests → med-safety → evidence →
        # working diagnosis → report). Retrieval and safety are injected by the composition
        # root; this facade remains conversation-free and never owns emergency policy.
        self._milestone = milestone or Milestone(
            gateway, retrievers=(), node_timeout_s=settings.MILESTONE_NODE_TIMEOUT_S)

    def _assess_context(self, picture: ClinicalPicture) -> str:
        """Render the interview-stage context through an overridable retrieval seam."""
        return _render_picture(picture)

    async def assess(self, picture: ClinicalPicture) -> AssessResult:
        """The engine's medical view of the current picture: differential + the single suggested
        question + its own sufficiency/confidence. Reuses `IntakeResult`. Never runs the guard."""
        try:
            result = await reason_over_context(
                self._gateway, system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
                context=self._assess_context(picture), model=self._model)
        except LLMError:
            return AssessResult()
        confidence = max((h.confidence for h in result.hypotheses), default=0.0)
        return AssessResult(
            hypotheses=result.hypotheses,
            suggested_question=result.next_question or "",
            sufficient=result.sufficient,
            confidence=confidence,
            understanding=result.understanding,
        )

    async def finalize(self, picture: ClinicalPicture, *, differential=None) -> FinalizeResult:
        """Run the EXISTING milestone over the mature picture → working diagnosis + report +
        recommended tests. Pure wiring of capabilities that already exist; no new reasoning.
        A failed run returns an empty result so the concierge can present a safe fallback."""
        state = self._build_state(picture, differential)
        try:
            await self._milestone.run(state)
        except Exception:  # noqa: BLE001 - never crash the conversation; present safely
            return FinalizeResult()
        v = state.visit
        return FinalizeResult(
            working_diagnosis=v.working_diagnosis, report=v.report,
            recommended_tests=list(v.recommended_tests))

    async def prescribe(self, picture: ClinicalPicture, *, condition: str) -> list[Prescription]:
        """Propose STRUCTURED medications for the working diagnosis, each PENDING clinician
        approval. A deterministic allergy cross-check flags (never silently drops) any item that
        matches a recorded allergy — the doctor decides. Empty list if there is no condition or
        the call fails (the conversation must never break on this)."""
        if not condition.strip():
            return []
        try:
            result = await self._gateway.complete_typed(
                _RxResult, system=_PRESCRIBE_SYS, user=_render_rx(picture, condition),
                model=self._model)
        except LLMError:
            return []
        allergies = [a.lower() for a in picture.allergies if a]
        out: list[Prescription] = []
        for it in result.prescriptions:
            if not it.medication.strip():
                continue
            rationale = it.rationale
            if any(al in it.medication.lower() for al in allergies):
                rationale = ("⚠ Potential allergy conflict — review before approving. "
                             + rationale).strip()
            out.append(Prescription(
                medication=it.medication, dose=it.dose or None, frequency=it.frequency or None,
                rationale=rationale, status=ReportStatus.PENDING_APPROVAL))
        return out

    # ── build a blackboard state from the picture so the milestone can run on it ──
    def _build_state(self, picture: ClinicalPicture, differential) -> ConsultationState:
        pid = uuid.uuid4().hex
        sex = Sex(picture.sex) if picture.sex in ("male", "female") else Sex.UNKNOWN
        patient = PatientRecord(
            patient_id=pid, age=picture.age, sex=sex,
            chronic_conditions=list(picture.chronic_conditions),
            medications=[Medication(name=m) for m in picture.medications if m],
            allergies=[Allergy(substance=a) for a in picture.allergies if a])
        visit = VisitState(
            visit_id=uuid.uuid4().hex, patient_id=pid,
            chief_complaint=picture.chief_complaint, differential=list(differential or []))
        session = SessionState(
            session_id=uuid.uuid4().hex, patient_id=pid, visit_id=visit.visit_id)
        session.add_message(MessageRole.PATIENT, _summary_text(picture))
        return ConsultationState(patient=patient, visit=visit, session=session)


def _summary_text(p: ClinicalPicture) -> str:
    parts: list[str] = []
    if p.chief_complaint:
        parts.append(f"My main concern is {p.chief_complaint}.")
    if p.symptoms:
        parts.append("I have been experiencing " + ", ".join(p.symptoms) + ".")
    if p.duration:
        parts.append(f"It has lasted {p.duration}.")
    if p.severity:
        parts.append(f"Severity: {p.severity}.")
    if p.location:
        parts.append(f"Location: {p.location}.")
    if p.associated_symptoms:
        parts.append("Also: " + ", ".join(p.associated_symptoms) + ".")
    return " ".join(parts) or "I am not feeling well."

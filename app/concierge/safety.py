"""Safety adapter for Concierge-owned patient conversations.

The Concierge remains the only patient-facing persona, while the shared
SafetyGuard evaluates every clinical turn before extraction or diagnosis.
"""
from __future__ import annotations

import uuid

from app.concierge.schemas import ClinicalPicture
from app.core.enums import Sex, TriageLevel
from app.core.state import (
    Allergy,
    ConsultationState,
    Medication,
    PatientRecord,
    SessionState,
    VisitState,
)
from app.safety.guard import SafetyGuard


class ConciergeSafety:
    """Callable safety seam injected into both web and robot Concierges."""

    def __init__(self, guard: SafetyGuard) -> None:
        self._guard = guard

    async def __call__(self, message: str, picture: ClinicalPicture) -> str | None:
        result = await self._guard.check(
            message=message,
            state=self._state_from_picture(picture),
        )
        if result.triage_level is TriageLevel.ROUTINE:
            return None
        if picture.language == "ar":
            if result.triage_level is TriageLevel.EMERGENCY:
                return (
                    "قد تشير الأعراض إلى حالة طارئة. اتصل بخدمات الطوارئ المحلية الآن، "
                    "ولا تقد السيارة بنفسك إذا كنت غير مستقر."
                )
            return (
                "قد تحتاج هذه الأعراض إلى تقييم طبي عاجل اليوم. توجّه إلى أقرب جهة رعاية "
                "عاجلة أو اتصل بطبيب فوراً."
            )
        return result.advice or (
            "Your symptoms may need urgent in-person assessment. Please seek medical care now."
        )

    @staticmethod
    def _state_from_picture(picture: ClinicalPicture) -> ConsultationState:
        patient_id = uuid.uuid4().hex
        visit_id = uuid.uuid4().hex
        sex = Sex(picture.sex) if picture.sex in ("male", "female") else Sex.UNKNOWN
        patient = PatientRecord(
            patient_id=patient_id,
            age=picture.age,
            sex=sex,
            chronic_conditions=list(picture.chronic_conditions),
            medications=[Medication(name=item) for item in picture.medications if item],
            allergies=[Allergy(substance=item) for item in picture.allergies if item],
        )
        visit = VisitState(
            visit_id=visit_id,
            patient_id=patient_id,
            chief_complaint=picture.chief_complaint or None,
        )
        session = SessionState(
            session_id=uuid.uuid4().hex,
            patient_id=patient_id,
            visit_id=visit_id,
        )
        return ConsultationState(patient=patient, visit=visit, session=session)

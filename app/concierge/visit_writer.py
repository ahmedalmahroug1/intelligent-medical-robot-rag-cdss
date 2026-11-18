"""
app/concierge/visit_writer.py
-----------------------------
The single SAVE WIRE for a Concierge conversation. The Concierge owns the dialogue and stays
pure (no store); this module takes a public `ConversationSnapshot` and persists a FULL visit —
transcript, working diagnosis, report (pending approval), evidence, and structured prescriptions
(each pending approval) — so that BOTH the robot and the remote-patient website land in the same
database the doctor reviews.

It is WRITE-ONLY and runs AFTER the patient has already seen the reply, so the conversation never
depends on it (a persistence failure is logged, not surfaced to the patient). Approval is
asynchronous: nothing here signs anything — the diagnosis report and every prescription are saved
PENDING_APPROVAL and carry no authority until a clinician acts on them in the doctor UI.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.concierge.agent import ConversationSnapshot
from app.concierge.reasoning import ReasoningFacade
from app.core.enums import MessageRole, ReportStatus, Sex
from app.core.state import (
    Allergy, ConsultationState, Medication, PatientRecord, PastVisitSummary, SessionState,
    VisitState,
)
from app.persistence.store import ConsultationStore

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _merge_picture_into_record(record: PatientRecord, picture) -> None:
    """Fold what the interview learned into the permanent record (validated fields, no raw SQL)."""
    if picture.age is not None:
        record.age = picture.age
    if picture.sex in ("male", "female"):
        record.sex = Sex(picture.sex)
    chronic = list(record.chronic_conditions)
    for c in picture.chronic_conditions:
        if c and c not in chronic:
            chronic.append(c)
    record.chronic_conditions = chronic
    meds = list(record.medications)
    have = {m.name.lower() for m in meds}
    for m in picture.medications:
        if m and m.lower() not in have:
            meds.append(Medication(name=m))
    record.medications = meds
    algs = list(record.allergies)
    havea = {a.substance.lower() for a in algs}
    for a in picture.allergies:
        if a and a.lower() not in havea:
            algs.append(Allergy(substance=a))
    record.allergies = algs


async def persist_concierge_visit(
    snapshot: ConversationSnapshot, *, store: ConsultationStore, facade: ReasoningFacade,
) -> str | None:
    """Persist a Concierge conversation as a full visit. Returns the new visit_id, or None if
    there is no identified patient. Generates structured prescriptions (pending approval) for the
    working diagnosis. Never raises into the caller — a failure is logged and returns None."""
    if snapshot.patient_id is None:
        return None
    try:
        result = snapshot.result
        wd = result.working_diagnosis if result else None

        record = await store.patients.get(snapshot.patient_id) \
            or PatientRecord(patient_id=snapshot.patient_id)
        _merge_picture_into_record(record, snapshot.picture)
        # The conversation language is a locked patient preference: save it so every later
        # visit (web or robot) resumes in the SAME language without re-asking.
        if snapshot.language in ("ar", "en"):
            record.preferred_language = snapshot.language

        prescriptions = []
        if wd is not None:
            prescriptions = await facade.prescribe(snapshot.picture, condition=wd.condition)

        report = result.report if result else None
        if report is not None and report.status not in (
                ReportStatus.APPROVED, ReportStatus.REJECTED):
            report = report.model_copy(update={"status": ReportStatus.PENDING_APPROVAL})

        visit = VisitState(
            visit_id=uuid.uuid4().hex, patient_id=record.patient_id,
            chief_complaint=snapshot.picture.chief_complaint or None,
            differential=list(wd.differential) if wd else [],
            recommended_tests=list(result.recommended_tests) if result else [],
            evidence=list(wd.evidence) if wd else [],
            working_diagnosis=wd, report=report, prescriptions=prescriptions,
            ended_at=_utcnow())

        session = SessionState(
            session_id=uuid.uuid4().hex, patient_id=record.patient_id,
            visit_id=visit.visit_id, language=snapshot.language)
        for role, text in snapshot.transcript:
            session.add_message(
                MessageRole.PATIENT if role == "patient" else MessageRole.DOCTOR_AI, text)

        cond = wd.condition if wd else None
        if cond and cond not in record.past_diagnoses:
            record.past_diagnoses = [*record.past_diagnoses, cond]
        record.visit_history = [
            *record.visit_history,
            PastVisitSummary(
                visit_id=visit.visit_id, date=_utcnow(),
                chief_complaint=snapshot.picture.chief_complaint or "",
                working_diagnosis=cond,
                report_status=report.status if report else ReportStatus.DRAFT),
        ]
        record.updated_at = _utcnow()

        await store.save(ConsultationState(patient=record, visit=visit, session=session))
        logger.info("Persisted concierge visit %s for patient %s", visit.visit_id,
                    record.patient_id)
        return visit.visit_id
    except Exception:  # noqa: BLE001 - persistence must never break the conversation
        logger.exception("Failed to persist concierge visit")
        return None

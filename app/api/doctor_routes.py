"""
app/api/doctor_routes.py
------------------------
The CLINICIAN-facing API. It lets a doctor review every consultation (transcript, working
diagnosis, the evidence the system cited, recommended tests, full patient record, and the
structured prescriptions) and ACT on each item — approve, edit, or reject the diagnosis report
and each prescription.

Asynchronous approval by design: the patient conversation never blocks on the doctor. The system
diagnoses and proposes medication immediately (saved PENDING_APPROVAL, with no authority), and the
doctor signs off here independently — during or after the conversation. "Approve" keeps the item;
"edit" stores the doctor's corrected version as the signed one; "reject" marks it rejected.

Auth is intentionally minimal (a demo-grade shared token via the `X-Doctor-Token` header, checked
only when a `DOCTOR_TOKEN` env var is set). A real deployment needs proper authentication.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from app.config import settings
from app.core.enums import ReportStatus
from app.core.state import Report, VisitState
from app.knowledge.models import DynamicDiseaseRecord, KnowledgeProposalStatus
from app.knowledge.repository import KnowledgeProposalError, KnowledgeProposalRepository
from app.knowledge.retrieval import ClinicalKnowledgeBase
from app.persistence.store import ConsultationStore


def require_doctor(x_doctor_token: str | None = Header(default=None)) -> None:
    """Demo-grade gate: enforced only if DOCTOR_TOKEN is configured (so local demos need no setup)."""
    expected = os.environ.get("DOCTOR_TOKEN")
    if expected is not None and x_doctor_token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing doctor token.")


router = APIRouter(prefix="/doctor", tags=["doctor"], dependencies=[Depends(require_doctor)])


def _store(request: Request) -> ConsultationStore:
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialised.")
    return ConsultationStore(db)


def _knowledge_repo(request: Request) -> KnowledgeProposalRepository:
    agent = getattr(request.app.state, "knowledge_agent", None)
    if agent is not None:
        return agent.repository
    # The doctor can still inspect/review already-created proposals when the
    # curator model is temporarily unavailable.
    return KnowledgeProposalRepository()


def _knowledge_base(request: Request) -> ClinicalKnowledgeBase:
    agent = getattr(request.app.state, "knowledge_agent", None)
    if agent is not None and getattr(agent, "knowledge", None) is not None:
        return agent.knowledge
    return ClinicalKnowledgeBase()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── request bodies ────────────────────────────────────────────────────────────
class DiagnosisAction(BaseModel):
    action: Literal["approve", "edit", "reject"]
    reviewed_by: str = "doctor"
    condition: str | None = None        # edit: corrected diagnosis
    reasoning: str | None = None        # edit: corrected reasoning
    report_content: str | None = None   # edit: corrected report text
    notes: str | None = None            # reject: why


class PrescriptionAction(BaseModel):
    action: Literal["approve", "edit", "reject"]
    reviewed_by: str = "doctor"
    medication: str | None = None
    dose: str | None = None
    frequency: str | None = None
    rationale: str | None = None
    notes: str | None = None


class KnowledgeAction(BaseModel):
    action: Literal["approve", "edit", "reject", "delete"]
    reviewed_by: str = "doctor"
    notes: str | None = None
    record: DynamicDiseaseRecord | None = None


# ── read endpoints ────────────────────────────────────────────────────────────
@router.get("/visits")
async def list_visits(request: Request) -> list[dict]:
    """The review queue: every consultation, most recent first, with its approval status."""
    store = _store(request)
    out = []
    for v in await store.visits.list_all():
        out.append({
            "visit_id": v.visit_id,
            "patient_id": v.patient_id,
            "started_at": v.started_at.isoformat(),
            "chief_complaint": v.chief_complaint,
            "condition": v.working_diagnosis.condition if v.working_diagnosis else None,
            "report_status": v.report.status.value if v.report else "none",
            "pending_prescriptions": sum(
                1 for p in v.prescriptions if p.status is ReportStatus.PENDING_APPROVAL),
        })
    return out


@router.get("/patients")
async def list_patients(request: Request) -> list[dict]:
    store = _store(request)
    out = []
    for p in await store.patients.list_all():
        out.append({
            "patient_id": p.patient_id,
            "age": p.age,
            "sex": p.sex.value,
            "chronic_conditions": list(p.chronic_conditions),
            "num_visits": len(p.visit_history),
            "last_diagnosis": p.past_diagnoses[-1] if p.past_diagnoses else None,
        })
    return out


@router.get("/knowledge/status")
async def knowledge_status(request: Request) -> dict:
    """Operational snapshot for the doctor console's knowledge-governance dashboard."""
    repository = _knowledge_repo(request)
    proposals = repository.list()
    counts = {status.value: 0 for status in KnowledgeProposalStatus}
    for proposal in proposals:
        counts[proposal.status.value] += 1

    knowledge = _knowledge_base(request)
    inventory = knowledge.inventory()
    seed_records = sum(item["review_status"] == "seed" for item in inventory)
    provisional_records = sum(
        item["review_status"] == "unapproved" for item in inventory
    )
    approved_records = sum(
        item["review_status"] == "clinician_approved" for item in inventory
    )
    return {
        "dynamic_rag_enabled": settings.ENABLE_DYNAMIC_KB_RAG,
        "curator_enabled": settings.ENABLE_KNOWLEDGE_AGENT,
        "curator_available": getattr(request.app.state, "knowledge_agent", None) is not None,
        "operational_records": len(inventory),
        "seed_records": seed_records,
        "provisional_dynamic_records": provisional_records,
        "approved_dynamic_records": approved_records,
        "proposal_counts": counts,
        "authority_rule": (
            "Sourced machine additions are visible to diagnosis immediately and remain labelled "
            "unapproved until a clinician approves, rejects, or deletes them."
        ),
    }


@router.get("/knowledge/proposals")
async def list_knowledge_proposals(
    request: Request,
    status: KnowledgeProposalStatus | None = None,
) -> list[dict]:
    """Knowledge review queue, including live unapproved additions."""
    return [
        {
            "proposal_id": item.proposal_id,
            "target_condition": item.target_condition,
            "gap_kind": item.gap_kind,
            "gap_reason": item.gap_reason,
            "status": item.status.value,
            "approvable": item.approvable,
            "generation_error": item.generation_error,
            "model_provider": item.model_provider,
            "model_name": item.model_name,
            "trigger_visit_id": item.trigger_visit_id,
            "created_at": item.created_at.isoformat(),
            "official_guideline_coverage": (
                item.research_audit.official_guideline_coverage
                if item.research_audit else None
            ),
            "verifier_passed": item.verification.passed if item.verification else None,
            "repair_attempts": item.research_audit.repair_attempts if item.research_audit else 0,
        }
        for item in _knowledge_repo(request).list(status)
    ]


@router.get("/knowledge/proposals/{proposal_id}")
async def get_knowledge_proposal(proposal_id: str, request: Request) -> dict:
    proposal = _knowledge_repo(request).get(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"Unknown knowledge proposal '{proposal_id}'.")
    return proposal.model_dump(mode="json")


@router.post("/knowledge/proposals/{proposal_id}")
async def act_on_knowledge_proposal(
    proposal_id: str,
    body: KnowledgeAction,
    request: Request,
) -> dict:
    """Edit, approve, reject, or delete one live machine-authored addition."""
    repository = _knowledge_repo(request)
    try:
        if body.action == "delete":
            proposal = repository.delete(
                proposal_id, reviewed_by=body.reviewed_by, notes=body.notes,
            )
        elif body.action == "reject":
            proposal = repository.reject(
                proposal_id, reviewed_by=body.reviewed_by, notes=body.notes,
            )
        elif body.action == "edit":
            if body.record is None:
                raise KnowledgeProposalError("Editing requires a complete validated record.")
            proposal = repository.edit(
                proposal_id,
                record=body.record,
                reviewed_by=body.reviewed_by,
                notes=body.notes,
            )
        else:
            # A doctor may correct and approve in one action. Persisting the edit
            # first recomputes the proposal's approvable gate.
            if body.record is not None:
                repository.edit(
                    proposal_id,
                    record=body.record,
                    reviewed_by=body.reviewed_by,
                    notes=body.notes,
                )
            proposal = repository.approve(
                proposal_id, reviewed_by=body.reviewed_by, notes=body.notes,
            )
    except KnowledgeProposalError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return proposal.model_dump(mode="json")


async def _visit_detail(store: ConsultationStore, visit: VisitState) -> dict:
    sessions = await store.sessions.list_for_visit(visit.visit_id)
    transcript = []
    for s in reversed(sessions):                 # list_for_visit is newest-first → chronological
        for m in s.transcript:
            transcript.append({"role": m.role.value, "content": m.content,
                               "timestamp": m.timestamp.isoformat()})
    patient = await store.patients.get(visit.patient_id)
    wd = visit.working_diagnosis
    return {
        "visit_id": visit.visit_id,
        "patient_id": visit.patient_id,
        "started_at": visit.started_at.isoformat(),
        "chief_complaint": visit.chief_complaint,
        "working_diagnosis": wd.model_dump(mode="json") if wd else None,
        "evidence": [e.model_dump(mode="json") for e in visit.evidence],
        "recommended_tests": [t.model_dump(mode="json") for t in visit.recommended_tests],
        "report": visit.report.model_dump(mode="json") if visit.report else None,
        "prescriptions": [p.model_dump(mode="json") for p in visit.prescriptions],
        "transcript": transcript,
        "patient": patient.model_dump(mode="json") if patient else None,
    }


@router.get("/visits/{visit_id}")
async def get_visit(visit_id: str, request: Request) -> dict:
    store = _store(request)
    visit = await store.visits.get(visit_id)
    if visit is None:
        raise HTTPException(status_code=404, detail=f"Unknown visit '{visit_id}'.")
    return await _visit_detail(store, visit)


# ── approval / edit / reject ──────────────────────────────────────────────────
@router.post("/visits/{visit_id}/diagnosis")
async def act_on_diagnosis(visit_id: str, body: DiagnosisAction, request: Request) -> dict:
    """Approve / edit / reject the diagnosis report. 'edit' stores the doctor's corrected diagnosis
    as the APPROVED one. The report's status is the diagnosis's authority gate."""
    store = _store(request)
    visit = await store.visits.get(visit_id)
    if visit is None:
        raise HTTPException(status_code=404, detail=f"Unknown visit '{visit_id}'.")
    report = visit.report or Report()

    if body.action == "reject":
        visit.report = report.model_copy(update={
            "status": ReportStatus.REJECTED, "reviewed_by": body.reviewed_by,
            "reviewed_at": _now(), "doctor_notes": body.notes})
    else:  # approve or edit → APPROVED
        if body.action == "edit" and visit.working_diagnosis is not None:
            wd_update = {k: v for k, v in {"condition": body.condition,
                                           "reasoning": body.reasoning}.items() if v is not None}
            if wd_update:
                visit.working_diagnosis = visit.working_diagnosis.model_copy(update=wd_update)
        report_update = {"status": ReportStatus.APPROVED, "reviewed_by": body.reviewed_by,
                         "reviewed_at": _now()}
        if body.action == "edit" and body.report_content is not None:
            report_update["content"] = body.report_content
        visit.report = report.model_copy(update=report_update)

    await store.visits.save(visit)
    return await _visit_detail(store, visit)


@router.post("/visits/{visit_id}/prescriptions/{prescription_id}")
async def act_on_prescription(
    visit_id: str, prescription_id: str, body: PrescriptionAction, request: Request,
) -> dict:
    """Approve / edit / reject ONE prescription by id. 'edit' stores the doctor's corrected
    medication as the APPROVED one."""
    store = _store(request)
    visit = await store.visits.get(visit_id)
    if visit is None:
        raise HTTPException(status_code=404, detail=f"Unknown visit '{visit_id}'.")
    rxs = list(visit.prescriptions)
    idx = next((i for i, p in enumerate(rxs) if p.prescription_id == prescription_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Unknown prescription '{prescription_id}'.")
    rx = rxs[idx]

    if body.action == "reject":
        update = {"status": ReportStatus.REJECTED, "reviewed_by": body.reviewed_by,
                  "reviewed_at": _now(), "doctor_notes": body.notes}
    else:
        update = {"status": ReportStatus.APPROVED, "reviewed_by": body.reviewed_by,
                  "reviewed_at": _now()}
        if body.action == "edit":
            update.update({k: v for k, v in {
                "medication": body.medication, "dose": body.dose,
                "frequency": body.frequency, "rationale": body.rationale}.items()
                if v is not None})
    rxs[idx] = rx.model_copy(update=update)
    visit.prescriptions = rxs
    await store.visits.save(visit)
    return await _visit_detail(store, visit)

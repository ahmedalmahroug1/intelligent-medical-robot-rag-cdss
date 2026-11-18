"""Manual command for asking the independent curator to research conditions.

Example:
    python -m app.knowledge.cli --condition "Migraine" --condition "Knee osteoarthritis"

This creates live provisional additions labelled unapproved. It never marks a
record as clinician approved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from app.config import settings
from app.core.enums import RiskTier
from app.core.state import (
    ConsultationState,
    Hypothesis,
    PatientRecord,
    SessionState,
    VisitState,
    WorkingDiagnosis,
)
from app.knowledge.agent import KnowledgeCuratorAgent
from app.knowledge.repository import KnowledgeProposalRepository, normalise_condition, safe_slug
from app.knowledge.retrieval import ClinicalKnowledgeBase
from app.llm.base import ModelTier
from app.llm.factory import make_gateway
from app.paths.diagnosis.pubmed import PubMedRetriever


def _state(condition: str, confidence: float) -> ConsultationState:
    suffix = uuid.uuid4().hex[:10]
    patient_id = f"kb-manual-patient-{suffix}"
    visit_id = f"kb-manual-{safe_slug(condition)}-{suffix}"
    hypothesis = Hypothesis(condition=condition, confidence=confidence)
    return ConsultationState(
        patient=PatientRecord(patient_id=patient_id),
        visit=VisitState(
            visit_id=visit_id,
            patient_id=patient_id,
            differential=[hypothesis],
            working_diagnosis=WorkingDiagnosis(
                condition=condition,
                confidence=confidence,
                risk_tier=RiskTier.LOW,
                reasoning="Manual knowledge-coverage request; not a patient diagnosis.",
                differential=[hypothesis],
            ),
        ),
        session=SessionState(
            session_id=f"kb-manual-session-{suffix}",
            patient_id=patient_id,
            visit_id=visit_id,
        ),
    )


def _summary(proposal) -> dict:
    return {
        "condition": proposal.target_condition,
        "proposal_id": proposal.proposal_id,
        "status": proposal.status.value,
        "approvable": proposal.approvable,
        "sources": len(proposal.record.sources) if proposal.record else 0,
        "official_guideline_coverage": (
            proposal.research_audit.official_guideline_coverage
            if proposal.research_audit else False
        ),
        "drug_labels_found": (
            proposal.research_audit.medication_labels_found
            if proposal.research_audit else []
        ),
        "repair_attempts": (
            proposal.research_audit.repair_attempts if proposal.research_audit else 0
        ),
        "verification_passed": (
            proposal.verification.passed if proposal.verification else False
        ),
        "verification_issues": len(proposal.verification.issues) if proposal.verification else None,
        "generation_error": proposal.generation_error,
    }


async def run(
    conditions: list[str],
    *,
    confidence: float,
    supersede_pending: bool,
    repair_proposals: list[str] | None = None,
) -> list[dict]:
    gateway = make_gateway(
        provider=settings.KNOWLEDGE_AGENT_PROVIDER,
        models={
            ModelTier.FAST: settings.KNOWLEDGE_AGENT_FAST_MODEL,
            ModelTier.REASONING: settings.KNOWLEDGE_AGENT_MODEL,
        },
        temperature=0,
        timeout=settings.KNOWLEDGE_AGENT_TIMEOUT_S,
        max_retries=settings.KNOWLEDGE_AGENT_MAX_RETRIES,
    )
    repository = KnowledgeProposalRepository()
    agent = KnowledgeCuratorAgent(
        gateway=gateway,
        repository=repository,
        knowledge=ClinicalKnowledgeBase(),
        retriever=PubMedRetriever(max_results=settings.KNOWLEDGE_AGENT_MAX_SOURCES),
        provider_name=settings.KNOWLEDGE_AGENT_PROVIDER,
        model_name=settings.KNOWLEDGE_AGENT_MODEL,
    )
    summaries: list[dict] = []
    for proposal_id in repair_proposals or []:
        summaries.append(_summary(await agent.repair_saved_proposal(proposal_id)))
    for condition in conditions:
        if supersede_pending:
            target = normalise_condition(condition)
            for proposal in repository.list():
                if (proposal.status.value == "pending"
                        and normalise_condition(proposal.target_condition) == target):
                    repository.reject(
                        proposal.proposal_id,
                        reviewed_by="curator-operator",
                        notes="Superseded by a staged research/verification run.",
                    )
        proposals = await agent.analyse(_state(condition, confidence))
        if not proposals:
            summaries.append({
                "condition": condition,
                "result": "no proposal (covered, duplicate pending, or gap audit found no gap)",
            })
            continue
        proposal = proposals[0]
        summaries.append(_summary(proposal))
    return summaries


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Create clinician-pending KB proposals.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--condition", action="append")
    target.add_argument(
        "--repair-proposal",
        action="append",
        help="Repair an existing pending proposal from its saved evidence and verifier issues.",
    )
    parser.add_argument("--confidence", type=float, default=0.82)
    parser.add_argument(
        "--supersede-pending",
        action="store_true",
        help="Reject older pending drafts for the same condition before researching it.",
    )
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    result = asyncio.run(run(
        args.condition or [],
        confidence=args.confidence,
        supersede_pending=args.supersede_pending,
        repair_proposals=args.repair_proposal,
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Offline tests for live provisional KB curation and clinician review."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from app.core.enums import RiskTier
from app.core.enums import DiagnosticPhase
from app.core.state import (
    ConsultationState,
    EvidenceItem,
    Hypothesis,
    PatientRecord,
    SessionState,
    VisitState,
    WorkingDiagnosis,
)
from app.knowledge.agent import KnowledgeCuratorAgent
from app.knowledge.drug_labels import StaticDrugLabelRetriever
from app.knowledge.models import KnowledgeProposalStatus
from app.knowledge.models import DynamicDiseaseRecord, KnowledgeProposal, KnowledgeSource
from app.knowledge.repository import KnowledgeProposalRepository
from app.knowledge.retrieval import ClinicalKnowledgeBase
from app.llm.base import ModelTier
from app.paths.diagnosis.retrieval import StaticRetriever
from tests.fakes import FakeGateway
from app.api.doctor_routes import router as doctor_router
from tests.test_service import _make_service


def _draft() -> dict:
    sid = ["PUBMED_123"]
    return {
        "schema_version": "dynamic-1.0",
        "name": "Example syndrome",
        "name_ar": "متلازمة تجريبية",
        "synonyms": ["Example disease"],
        "metadata_source_ids": sid,
        "overview": "A sourced example condition used only in an offline unit test.",
        "overview_source_ids": sid,
        "symptoms": [
            {"text": "feature one", "diagnostic_value": "suggestive", "source_ids": sid},
            {"text": "feature two", "diagnostic_value": "compatible", "source_ids": sid},
            {"text": "feature three", "diagnostic_value": "nonspecific", "source_ids": sid},
        ],
        "negative_findings": [
            {"text": "negative feature one", "diagnostic_value": "reduces likelihood", "source_ids": sid},
            {"text": "negative feature two", "diagnostic_value": "reduces likelihood", "source_ids": sid},
        ],
        "history_questions": [
            {"question": "When did it start?", "purpose": "chronology", "source_ids": sid},
            {"question": "What makes it worse?", "purpose": "pattern", "source_ids": sid},
            {"question": "What medicines do you take?", "purpose": "safety", "source_ids": sid},
        ],
        "diagnostic_questions": [
            {"question": "Is feature one present?", "purpose": "discriminate", "source_ids": sid},
            {"question": "Is feature two absent?", "purpose": "discriminate", "source_ids": sid},
        ],
        "diagnostic_criteria": [{"text": "Clinical assessment is required.", "source_ids": sid}],
        "tests": [
            {"name": "Example test", "indication": "when uncertain",
             "expected_interpretation": "context dependent", "source_ids": sid},
        ],
        "differential_diagnosis": [
            {"condition": "Alternative A", "distinguishing_features": "feature A", "source_ids": sid},
            {"condition": "Alternative B", "distinguishing_features": "feature B", "source_ids": sid},
            {"condition": "Alternative C", "distinguishing_features": "feature C", "source_ids": sid},
        ],
        "red_flags": [
            {"text": "Rapid deterioration requires urgent assessment.", "source_ids": sid},
            {"text": "Severe systemic illness requires urgent assessment.", "source_ids": sid},
        ],
        "non_drug_management": [{"text": "Supportive care may be considered.", "source_ids": sid}],
        "treatment_options": [
            {"name": "Example class", "role": "candidate only", "when_to_consider": "after confirmation",
             "avoid_if": ["contraindication"], "use_caution_if": ["comorbidity"],
             "required_history_checks": ["allergy and medication history"],
             "monitoring": ["clinical response"], "source_ids": sid,
             "do_not_autoprescribe": True},
        ],
        "patient_advice": [
            {"text": "Seek clinical review if symptoms worsen.", "source_ids": sid},
            {"text": "Keep a record of symptoms for follow-up.", "source_ids": sid},
        ],
        "limitations": ["This is a test draft."],
        "sources": [],
        "generated_by_model": "",
    }


def _diagnostic_draft() -> dict:
    full = _draft()
    keys = {
        "name", "name_ar", "synonyms", "metadata_source_ids", "overview",
        "overview_source_ids", "symptoms", "negative_findings", "history_questions",
        "diagnostic_questions", "diagnostic_criteria", "tests",
        "differential_diagnosis", "red_flags",
    }
    return {**{key: full[key] for key in keys},
            "diagnostic_limitations": ["This is a test draft."]}


def _treatment_draft() -> dict:
    full = _draft()
    for option in full["treatment_options"]:
        option["source_ids"] = ["PUBMED_123", "OFFICIAL_EXAMPLE"]
    return {
        "non_drug_management": full["non_drug_management"],
        "treatment_options": full["treatment_options"],
        "patient_advice": full["patient_advice"],
        "treatment_limitations": [],
    }


class _StaticGuidelines:
    source_name = "Test official guidelines"

    async def search(self, condition: str, topics: list[str] | None = None):
        return [KnowledgeSource(
            source_id="OFFICIAL_EXAMPLE",
            source_type="Official guideline",
            title="Official example guideline",
            identifier="EXAMPLE-1",
            url="https://example.test/guideline",
            study_type="Clinical guideline",
            summary="Reviewed test-only official guidance for Example syndrome.",
            evidence_scope=["diagnosis", "treatment", "medication_safety"],
        )]


def _state() -> ConsultationState:
    hypothesis = Hypothesis(condition="Example syndrome", confidence=0.8)
    return ConsultationState(
        patient=PatientRecord(patient_id="p1"),
        visit=VisitState(
            visit_id="v1",
            patient_id="p1",
            differential=[hypothesis],
            working_diagnosis=WorkingDiagnosis(
                condition="Example syndrome",
                confidence=0.8,
                risk_tier=RiskTier.LOW,
                reasoning="test",
                differential=[hypothesis],
            ),
        ),
        session=SessionState(session_id="s1", patient_id="p1", visit_id="v1"),
    )


async def _run(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    pending = tmp_path / "pending"
    approved = tmp_path / "approved"
    repository = KnowledgeProposalRepository(pending_dir=pending, approved_dir=approved)
    knowledge = ClinicalKnowledgeBase(
        seed_dir=seed,
        pending_dir=pending,
        approved_dir=approved,
    )
    gateway = FakeGateway(structured=[
        {"gaps": [{"condition": "Example syndrome", "gap_kind": "missing",
                    "reason": "No active record", "matched_record_name": None}]},
        {
            "diagnostic_queries": ["Example syndrome diagnosis guideline"],
            "treatment_queries": ["Example syndrome treatment guideline"],
            "official_guideline_topics": ["Example syndrome"],
            "medication_search_terms": [],
            "research_notes": [],
        },
        _diagnostic_draft(),
        _treatment_draft(),
        {"passed": True, "issues": [], "checked_claims": 24,
         "summary": "Every test claim is supported."},
    ])
    retriever = StaticRetriever("PubMed", [EvidenceItem(
        source="PubMed",
        title="Example systematic review",
        identifier="PMID:123",
        summary="A test-only abstract supporting the example fields.",
        published_year=2025,
        study_type="Systematic Review",
    ), EvidenceItem(
        source="PubMed", title="Second example guideline", identifier="PMID:124",
        summary="A second test-only evidence summary.", published_year=2025,
        study_type="Practice Guideline",
    )])
    agent = KnowledgeCuratorAgent(
        gateway=gateway,
        repository=repository,
        knowledge=knowledge,
        retriever=retriever,
        guideline_catalog=_StaticGuidelines(),
        drug_labels=StaticDrugLabelRetriever(),
        provider_name="test-provider",
        model_name="strong-test-model",
        min_confidence=0.35,
    )

    proposals = await agent.analyse(_state())
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.status is KnowledgeProposalStatus.PENDING and proposal.approvable
    assert proposal.record and any(
        source.source_id == "PUBMED_123" for source in proposal.record.sources
    )
    assert proposal.research_audit and proposal.research_audit.official_guideline_coverage
    assert proposal.verification and proposal.verification.passed
    assert len(gateway.calls) == 5
    assert [call["model"] for call in gateway.calls[:2]] == [
        ModelTier.FAST, ModelTier.FAST,
    ]
    assert all(call["model"] is ModelTier.REASONING for call in gateway.calls[2:])

    # A sourced generated record is operational immediately but visibly
    # unapproved until the doctor reviews it.
    retrieved = knowledge.retrieve("Example syndrome feature one")
    assert retrieved and retrieved[0]["name"] == "Example syndrome"
    assert retrieved[0]["review_status"] == "unapproved"
    assert "UNAPPROVED" in knowledge.guidance("Example syndrome feature one")

    approved_proposal = repository.approve(
        proposal.proposal_id, reviewed_by="Dr Test", notes="unit-test approval",
    )
    assert approved_proposal.status is KnowledgeProposalStatus.APPROVED
    retrieved = knowledge.retrieve("Example syndrome feature one")
    assert retrieved and retrieved[0]["name"] == "Example syndrome"
    assert retrieved[0]["review_status"] == "clinician_approved"


def test_generated_record_is_live_and_labelled_until_doctor_review() -> None:
    root = Path("tests/fixtures/dynamic_kb")
    generated = [*root.joinpath("pending").glob("*.json"),
                 *root.joinpath("approved").glob("*.json")]
    for path in generated:
        path.unlink()
    try:
        asyncio.run(_run(root))
    finally:
        for path in [*root.joinpath("pending").glob("*.json"),
                     *root.joinpath("approved").glob("*.json")]:
            path.unlink()


def test_blocked_machine_draft_does_not_enter_operational_retrieval() -> None:
    root = Path("tests/fixtures/dynamic_kb_blocked")
    pending = root / "pending"
    approved = root / "approved"
    pending.mkdir(parents=True, exist_ok=True)
    approved.mkdir(parents=True, exist_ok=True)
    try:
        record = DynamicDiseaseRecord.model_validate(_draft())
        repository = KnowledgeProposalRepository(
            pending_dir=pending,
            approved_dir=approved,
        )
        repository.save(KnowledgeProposal(
            proposal_id="blocked-proposal",
            target_condition=record.name,
            gap_kind="missing",
            gap_reason="Automated validation did not pass",
            trigger_visit_id="blocked-visit",
            approvable=False,
            record=record,
        ))
        knowledge = ClinicalKnowledgeBase(
            seed_dir=root / "seed",
            pending_dir=pending,
            approved_dir=approved,
        )
        assert knowledge.retrieve("Example syndrome feature one") == []
    finally:
        for path in [*pending.glob("*.json"), *approved.glob("*.json")]:
            path.unlink()


async def _run_repair(root: Path) -> None:
    repository = KnowledgeProposalRepository(
        pending_dir=root / "pending", approved_dir=root / "approved",
    )
    knowledge = ClinicalKnowledgeBase(
        seed_dir=root / "seed",
        pending_dir=root / "pending",
        approved_dir=root / "approved",
    )
    plan = {
        "diagnostic_queries": ["Example syndrome diagnosis guideline"],
        "treatment_queries": ["Example syndrome treatment guideline"],
        "official_guideline_topics": ["Example syndrome"],
        "medication_search_terms": [],
        "research_notes": [],
    }
    gateway = FakeGateway(structured=[
        {"gaps": [{"condition": "Example syndrome", "gap_kind": "missing",
                    "reason": "No active record", "matched_record_name": None}]},
        plan,
        _diagnostic_draft(),
        _treatment_draft(),
        {
            "passed": False,
            "issues": [{
                "severity": "minor",
                "location": "treatment_options[1]",
                "message": "The safety citation needs correction.",
                "source_ids": ["OFFICIAL_EXAMPLE"],
                "suggested_fix": "Use the supplied safety source.",
            }],
            "checked_claims": 24,
            "summary": "One material issue.",
        },
        {"operations": [{
            "op": "replace",
            "path": "/treatment_options/0/use_caution_if",
            "value": ["comorbidity"],
            "rationale": "Keep only the supported test caution.",
        }]},
        {"passed": True, "issues": [], "checked_claims": 24,
         "summary": "The repaired test record is supported."},
    ])
    retriever = StaticRetriever("PubMed", [EvidenceItem(
        source="PubMed", title="Example review", identifier="PMID:123",
        summary="Test evidence for diagnosis and treatment.",
        published_year=2025, study_type="Systematic Review",
    ), EvidenceItem(
        source="PubMed", title="Second review", identifier="PMID:124",
        summary="Second test evidence summary.",
        published_year=2025, study_type="Practice Guideline",
    )])
    agent = KnowledgeCuratorAgent(
        gateway=gateway,
        repository=repository,
        knowledge=knowledge,
        retriever=retriever,
        guideline_catalog=_StaticGuidelines(),
        drug_labels=StaticDrugLabelRetriever(),
        provider_name="test-provider",
        model_name="strong-test-model",
        repair_attempts=1,
    )
    proposals = await agent.analyse(_state())
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.approvable
    assert proposal.research_audit and proposal.research_audit.repair_attempts == 1
    assert proposal.verification and proposal.verification.passed
    assert len(gateway.calls) == 7


def test_verifier_issue_triggers_evidence_bounded_repair() -> None:
    root = Path("tests/fixtures/dynamic_kb_repair")
    try:
        asyncio.run(_run_repair(root))
    finally:
        for directory in (root / "pending", root / "approved"):
            for path in directory.glob("*.json"):
                path.unlink()


def test_targeted_patch_normalises_plain_string_limitations() -> None:
    payload = {"limitations": ["unsupported sentence"]}
    KnowledgeCuratorAgent._apply_patch_operation(
        payload, "replace", "/limitations/0/text", "supported limitation",
    )
    assert payload["limitations"] == ["supported limitation"]


async def _run_api(root: Path) -> None:
    repository = KnowledgeProposalRepository(
        pending_dir=root / "pending", approved_dir=root / "approved",
    )
    draft = _draft()
    draft["sources"] = [{
        "source_id": "PUBMED_123",
        "source_type": "PubMed",
        "title": "Example review",
        "identifier": "PMID:123",
        "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
        "summary": "test",
    }, {
        "source_id": "PUBMED_124",
        "source_type": "PubMed",
        "title": "Second example review",
        "identifier": "PMID:124",
        "url": "https://pubmed.ncbi.nlm.nih.gov/124/",
        "summary": "test two",
    }]
    record = DynamicDiseaseRecord.model_validate(draft)
    repository.save(KnowledgeProposal(
        proposal_id="api-proposal",
        target_condition=record.name,
        gap_kind="missing",
        gap_reason="No record",
        trigger_visit_id="visit-api",
        approvable=True,
        record=record,
    ))
    knowledge = ClinicalKnowledgeBase(
        seed_dir=root / "seed",
        pending_dir=root / "pending",
        approved_dir=root / "approved",
    )
    app = FastAPI()
    app.state.knowledge_agent = SimpleNamespace(
        repository=repository,
        knowledge=knowledge,
    )
    app.include_router(doctor_router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        status = await client.get("/doctor/knowledge/status")
        assert status.status_code == 200
        assert status.json()["authority_rule"].startswith("Sourced machine additions")
        assert "operational_records" in status.json()
        assert status.json()["provisional_dynamic_records"] == 1
        queue = await client.get("/doctor/knowledge/proposals?status=pending")
        assert queue.status_code == 200 and len(queue.json()) == 1
        approved = await client.post(
            "/doctor/knowledge/proposals/api-proposal",
            json={"action": "approve", "reviewed_by": "Dr API"},
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["status"] == "approved"
        deleted = await client.post(
            "/doctor/knowledge/proposals/api-proposal",
            json={"action": "delete", "reviewed_by": "Dr API"},
        )
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["status"] == "deleted"
    assert not (root / "approved" / "example_syndrome.json").exists()


def test_doctor_api_controls_review_and_removal() -> None:
    root = Path("tests/fixtures/dynamic_kb")
    for path in [*root.joinpath("pending").glob("*.json"),
                 *root.joinpath("approved").glob("*.json")]:
        path.unlink()
    try:
        asyncio.run(_run_api(root))
    finally:
        for path in [*root.joinpath("pending").glob("*.json"),
                     *root.joinpath("approved").glob("*.json")]:
            path.unlink()


async def _run_completion_hook() -> None:
    service, store, _, db = await _make_service()
    observed: list[ConsultationState] = []
    service.set_completion_observer(observed.append)
    try:
        state = await service.start()
        state.session.phase = DiagnosticPhase.COMPLETE
        await store.save(state)
        await service.send(session_id=state.session.session_id, message="first follow-up")
        await service.send(session_id=state.session.session_id, message="second follow-up")
        assert len(observed) == 1
        assert observed[0].visit.visit_id == state.visit.visit_id
    finally:
        await db.close()


def test_completed_visit_notifies_curator_once() -> None:
    asyncio.run(_run_completion_hook())

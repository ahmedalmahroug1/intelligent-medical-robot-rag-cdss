"""Validated records and review-state models for the dynamic knowledge agent."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DELETED = "deleted"


class SourcedText(BaseModel):
    text: str = Field(min_length=2)
    source_ids: list[str] = Field(min_length=1)


class DiagnosticClue(BaseModel):
    text: str = Field(min_length=2)
    diagnostic_value: str = ""
    source_ids: list[str] = Field(min_length=1)


class DiagnosticQuestion(BaseModel):
    question: str = Field(min_length=3)
    purpose: str = ""
    source_ids: list[str] = Field(min_length=1)


class DiagnosticTest(BaseModel):
    name: str = Field(min_length=2)
    indication: str = ""
    expected_interpretation: str = ""
    source_ids: list[str] = Field(min_length=1)


class DifferentialEntry(BaseModel):
    condition: str = Field(min_length=2)
    distinguishing_features: str = ""
    source_ids: list[str] = Field(min_length=1)


class TreatmentOption(BaseModel):
    name: str = Field(min_length=2)
    role: str = ""
    when_to_consider: str = ""
    avoid_if: list[str] = Field(default_factory=list)
    use_caution_if: list[str] = Field(default_factory=list)
    required_history_checks: list[str] = Field(default_factory=list)
    monitoring: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(min_length=1)
    do_not_autoprescribe: bool = True

    @field_validator("do_not_autoprescribe")
    @classmethod
    def _must_require_review(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("dynamic treatment options must never be auto-prescribed")
        return value


class KnowledgeSource(BaseModel):
    source_id: str = Field(min_length=2)
    source_type: str = "PubMed"
    title: str = Field(min_length=2)
    identifier: str = Field(min_length=2)
    url: str = Field(min_length=8)
    published_year: int | None = None
    study_type: str | None = None
    summary: str = ""
    evidence_scope: list[Literal["diagnosis", "treatment", "medication_safety", "general"]] = (
        Field(default_factory=list)
    )

    @field_validator("url")
    @classmethod
    def _https_only(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("knowledge sources must use HTTPS")
        return value


class DynamicDiseaseRecord(BaseModel):
    """Compact operational schema produced by the curator.

    It intentionally contains the fields consumed by diagnostic RAG and treatment
    review, rather than pretending that a generated proposal completed every
    research field in the larger hand-authored KB 1.3 schema.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dynamic-1.0"] = "dynamic-1.0"
    name: str = Field(min_length=2)
    name_ar: str = ""
    synonyms: list[str] = Field(default_factory=list)
    metadata_source_ids: list[str] = Field(min_length=1)
    overview: str = Field(min_length=10)
    overview_source_ids: list[str] = Field(min_length=1)
    symptoms: list[DiagnosticClue] = Field(min_length=3)
    negative_findings: list[DiagnosticClue] = Field(default_factory=list)
    history_questions: list[DiagnosticQuestion] = Field(min_length=3)
    diagnostic_questions: list[DiagnosticQuestion] = Field(min_length=1)
    diagnostic_criteria: list[SourcedText] = Field(min_length=1)
    tests: list[DiagnosticTest] = Field(min_length=1)
    differential_diagnosis: list[DifferentialEntry] = Field(min_length=2)
    red_flags: list[SourcedText] = Field(min_length=1)
    non_drug_management: list[SourcedText] = Field(default_factory=list)
    treatment_options: list[TreatmentOption] = Field(default_factory=list)
    patient_advice: list[SourcedText] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[KnowledgeSource] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
    generated_by_model: str = ""


class GapTarget(BaseModel):
    condition: str = Field(min_length=2)
    gap_kind: Literal["missing", "weak"]
    reason: str = Field(min_length=3)
    matched_record_name: str | None = None


class GapAssessment(BaseModel):
    gaps: list[GapTarget] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Small search plan produced before any clinical record is authored."""

    diagnostic_queries: list[str] = Field(min_length=1, max_length=4)
    treatment_queries: list[str] = Field(min_length=1, max_length=4)
    official_guideline_topics: list[str] = Field(default_factory=list, max_length=4)
    medication_search_terms: list[str] = Field(default_factory=list, max_length=8)
    research_notes: list[str] = Field(default_factory=list)


class DiagnosticDraft(BaseModel):
    """Diagnostic half of a record, authored only from diagnostic evidence."""

    name: str = Field(min_length=2)
    name_ar: str = Field(min_length=2)
    synonyms: list[str] = Field(default_factory=list)
    metadata_source_ids: list[str] = Field(min_length=1)
    overview: str = Field(min_length=10)
    overview_source_ids: list[str] = Field(min_length=1)
    symptoms: list[DiagnosticClue] = Field(min_length=3)
    negative_findings: list[DiagnosticClue] = Field(min_length=2)
    history_questions: list[DiagnosticQuestion] = Field(min_length=3)
    diagnostic_questions: list[DiagnosticQuestion] = Field(min_length=2)
    diagnostic_criteria: list[SourcedText] = Field(min_length=1)
    tests: list[DiagnosticTest] = Field(min_length=1)
    differential_diagnosis: list[DifferentialEntry] = Field(min_length=3)
    red_flags: list[SourcedText] = Field(min_length=2)
    diagnostic_limitations: list[str] = Field(default_factory=list)


class TreatmentDraft(BaseModel):
    """Treatment half, written after drug-label evidence has been retrieved."""

    non_drug_management: list[SourcedText] = Field(default_factory=list)
    treatment_options: list[TreatmentOption] = Field(default_factory=list)
    patient_advice: list[SourcedText] = Field(min_length=2)
    treatment_limitations: list[str] = Field(default_factory=list)


class VerificationIssue(BaseModel):
    severity: Literal["critical", "major", "minor"]
    location: str = Field(min_length=2)
    message: str = Field(min_length=3)
    source_ids: list[str] = Field(default_factory=list)
    suggested_fix: str = ""


class DraftVerification(BaseModel):
    """Independent claim/evidence review; never an approval by itself."""

    passed: bool = False
    issues: list[VerificationIssue] = Field(default_factory=list)
    checked_claims: int = Field(default=0, ge=0)
    summary: str = ""

    def blocking_issues(self) -> list[VerificationIssue]:
        # For a reference record, even a "minor" citation mismatch is not ready:
        # minor describes clinical severity, not permission to publish an
        # unsupported statement. The repair loop may reword or remove it.
        return list(self.issues)


class RepairPatchOperation(BaseModel):
    op: Literal["replace", "remove"]
    path: str = Field(min_length=2)
    value: Any | None = None
    rationale: str = ""


class TargetedRepair(BaseModel):
    """Small RFC-6902-like patch; validated again as a full record after use."""

    operations: list[RepairPatchOperation] = Field(min_length=1, max_length=20)


class EvidenceToolRun(BaseModel):
    tool: str
    query: str
    source_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class ResearchAudit(BaseModel):
    """Persisted audit trail showing what the curator actually searched."""

    plan: ResearchPlan | None = None
    tool_runs: list[EvidenceToolRun] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    official_guideline_coverage: bool = False
    medication_labels_requested: list[str] = Field(default_factory=list)
    medication_labels_found: list[str] = Field(default_factory=list)
    repair_attempts: int = Field(default=0, ge=0)


class KnowledgeProposal(BaseModel):
    proposal_id: str
    target_condition: str
    gap_kind: Literal["missing", "weak"]
    gap_reason: str
    trigger_visit_id: str
    trigger_hypotheses: list[dict] = Field(default_factory=list)
    status: KnowledgeProposalStatus = KnowledgeProposalStatus.PENDING
    approvable: bool = False
    record: DynamicDiseaseRecord | None = None
    generation_error: str | None = None
    model_provider: str = ""
    model_name: str = ""
    research_audit: ResearchAudit | None = None
    verification: DraftVerification | None = None
    created_at: datetime = Field(default_factory=utcnow)
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    reviewer_notes: str | None = None


def record_citation_errors(record: DynamicDiseaseRecord) -> list[str]:
    """Return traceability failures that make a record non-approvable."""
    ids = [source.source_id for source in record.sources]
    allowed = set(ids)
    errors: list[str] = []
    if len(ids) != len(allowed):
        errors.append("source IDs are not unique")
    for field_name, cited in (
        ("metadata_source_ids", set(record.metadata_source_ids)),
        ("overview_source_ids", set(record.overview_source_ids)),
    ):
        if not cited or not cited.issubset(allowed):
            errors.append(f"{field_name} has unknown or missing source IDs")
    groups = {
        "symptoms": record.symptoms,
        "negative_findings": record.negative_findings,
        "history_questions": record.history_questions,
        "diagnostic_questions": record.diagnostic_questions,
        "diagnostic_criteria": record.diagnostic_criteria,
        "tests": record.tests,
        "differential_diagnosis": record.differential_diagnosis,
        "red_flags": record.red_flags,
        "non_drug_management": record.non_drug_management,
        "treatment_options": record.treatment_options,
        "patient_advice": record.patient_advice,
    }
    for group, items in groups.items():
        for index, item in enumerate(items, 1):
            cited = set(item.source_ids)
            if not cited or not cited.issubset(allowed):
                errors.append(f"{group}[{index}] has unknown or missing source IDs")
    return errors


def record_approval_errors(record: DynamicDiseaseRecord) -> list[str]:
    """Strict operational gate applied before a clinician can approve a draft.

    Source-ID traceability alone is not enough. A clinically thin record remains
    useful as a visible draft, but must not be labelled ready for activation.
    """
    errors = record_citation_errors(record)
    if not any("\u0600" <= char <= "\u06ff" for char in record.name_ar):
        errors.append("Arabic disease name is missing")
    if len(record.negative_findings) < 2:
        errors.append("at least two sourced negative findings are required")
    if len(record.diagnostic_questions) < 2:
        errors.append("at least two sourced discriminating questions are required")
    if len(record.differential_diagnosis) < 3:
        errors.append("at least three sourced differential diagnoses are required")
    if any(not item.distinguishing_features.strip() for item in record.differential_diagnosis):
        errors.append("every differential needs distinguishing features")
    if len(record.red_flags) < 2:
        errors.append("at least two separately sourced red flags are required")
    if any(not item.diagnostic_value.strip() for item in record.symptoms):
        errors.append("every symptom needs an explicit diagnostic value")
    if len(record.patient_advice) < 2:
        errors.append("patient advice is incomplete")
    if len(record.sources) < 2 or any(not source.summary.strip() for source in record.sources):
        errors.append("at least two non-empty evidence summaries are required")

    source_map = {source.source_id: source for source in record.sources}
    diagnostic_groups = {
        "symptoms": record.symptoms,
        "negative_findings": record.negative_findings,
        "history_questions": record.history_questions,
        "diagnostic_questions": record.diagnostic_questions,
        "diagnostic_criteria": record.diagnostic_criteria,
        "tests": record.tests,
        "differential_diagnosis": record.differential_diagnosis,
        "red_flags": record.red_flags,
    }
    for group, items in diagnostic_groups.items():
        for index, item in enumerate(items, 1):
            cited = [source_map[source_id] for source_id in item.source_ids
                     if source_id in source_map]
            if cited and all(
                source.evidence_scope
                and "diagnosis" not in source.evidence_scope
                and "general" not in source.evidence_scope
                for source in cited
            ):
                errors.append(f"{group}[{index}] cites no diagnostic-scope evidence")

    treatment_groups = {
        "non_drug_management": record.non_drug_management,
        "treatment_options": record.treatment_options,
        "patient_advice": record.patient_advice,
    }
    for group, items in treatment_groups.items():
        for index, item in enumerate(items, 1):
            cited = [source_map[source_id] for source_id in item.source_ids
                     if source_id in source_map]
            if cited and all(
                source.evidence_scope
                and not {"treatment", "medication_safety", "general"}
                .intersection(source.evidence_scope)
                for source in cited
            ):
                errors.append(f"{group}[{index}] cites no treatment-scope evidence")

    for index, option in enumerate(record.treatment_options, 1):
        text = f"{option.role} {option.when_to_consider}".casefold()
        explicitly_not_used = "not recommended" in text or "should not" in text
        if explicitly_not_used:
            continue
        if not option.required_history_checks:
            errors.append(f"treatment_options[{index}] lacks required history checks")
        if not option.monitoring:
            errors.append(f"treatment_options[{index}] lacks monitoring")
        if not option.avoid_if and not option.use_caution_if:
            errors.append(f"treatment_options[{index}] lacks avoidance/caution guidance")
        cited = [source_map[source_id] for source_id in option.source_ids
                 if source_id in source_map]
        if cited and not any(
            not source.evidence_scope or "medication_safety" in source.evidence_scope
            for source in cited
        ):
            errors.append(f"treatment_options[{index}] cites no medication-safety evidence")

    limitation_text = " ".join(record.limitations).casefold()
    if "inferred" in limitation_text or "not detailed" in limitation_text:
        errors.append("the draft self-reports inferred or unsupported operational content")
    return errors


def proposal_approval_errors(proposal: KnowledgeProposal) -> list[str]:
    """Full machine-draft gate: content, research coverage and independent review."""
    if proposal.record is None:
        return ["record is missing"]
    errors = record_approval_errors(proposal.record)
    if proposal.research_audit is not None and not proposal.research_audit.official_guideline_coverage:
        errors.append("no reviewed official guideline covered this condition")
    if proposal.verification is not None:
        blocking = proposal.verification.blocking_issues()
        if not proposal.verification.passed:
            errors.append("independent claim verification did not pass")
        errors.extend(
            f"verification {issue.severity} at {issue.location}: {issue.message}"
            for issue in blocking
        )
    return list(dict.fromkeys(errors))

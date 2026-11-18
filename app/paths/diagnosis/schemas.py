"""
app/paths/diagnosis/schemas.py
------------------------------
The structured output contract for the diagnostic interview loop.

`IntakeResult` is what the loop's single LLM call returns each turn. One call does
the whole cheap step — update the understanding, refresh the working differential,
judge sufficiency, and choose the next question — and this schema is how that
result is parsed and validated (via `gateway.complete_typed(IntakeResult, ...)`).

It reuses the domain `Hypothesis` directly, so the model's differential lands in
`VisitState.differential` with no translation. The schema is a transient parse
target (constructed once per turn from the model's JSON, never persisted), so it
is a plain BaseModel — not one of the validate-on-assignment blackboard models.

All fields have defaults: if the model omits one, parsing still succeeds with a
safe value (notably `sufficient=False`, so a missing flag never forces an
unwanted convergence). Unknown extra keys are ignored (Pydantic v2 default),
keeping the parse robust to a chatty model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.core.enums import RiskTier
from app.core.state import Confidence, EvidenceItem, Hypothesis, RecommendedTest


class IntakeResult(BaseModel):
    """One turn's worth of diagnostic reasoning, parsed from the loop's LLM call."""

    # The model's running understanding of the case (audit + continuity).
    understanding: str = ""

    # A short phrase naming the main problem. Set onto the visit once it is clear.
    chief_complaint: str | None = None

    # The updated working differential. Reuses the domain Hypothesis so it maps
    # straight onto VisitState.differential. Drives next-question selection.
    hypotheses: list[Hypothesis] = Field(default_factory=list)

    # The model's view on whether enough has been gathered to stop interviewing.
    # The path treats this as advisory: a deterministic gate (confidence floor +
    # iteration cap) decides the actual transition on top of this flag.
    sufficient: bool = False
    sufficiency_rationale: str = ""

    # The single most informative next question, phrased for the patient. Null
    # when the model considers the interview sufficient (the path supplies a
    # fallback question if it overrides that and keeps interviewing).
    next_question: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Milestone node output contracts (the run-once evidence pipeline, File 7b)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each milestone node that calls the model parses its output into one of these.
# They are transient parse targets (plain BaseModels), same as IntakeResult; the
# node assembles the durable domain objects (WorkingDiagnosis, Report, ...) and
# writes them to the blackboard.


class DifferentialResult(BaseModel):
    """The consolidated differential at the start of the milestone (a careful
    pass over the whole interview, seeded from the loop's running differential)."""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    summary: str = ""


class RefutedCondition(BaseModel):
    """One candidate the interview DIRECTLY contradicts, with the patient statement
    that rules it out."""
    condition: str = ""
    reason: str = ""


class RefutationResult(BaseModel):
    """The conservative refutation pass: ONLY the candidates the patient's own
    account explicitly contradicts. Anything merely unsupported or less-likely is
    left in place — the node removes exactly these and keeps the rest, so it can
    prune a distractor but can never demote the lead on a published prior."""
    remove: list[RefutedCondition] = Field(default_factory=list)


class TestRecommendationResult(BaseModel):
    """Investigations that would best discriminate the leading hypotheses."""
    tests: list[RecommendedTest] = Field(default_factory=list)


class EvidenceAppraisal(BaseModel):
    """The verification verdict on ONE retrieved item, keyed by its identifier so
    the node can apply the score back onto the matching EvidenceItem."""
    identifier: str
    quality_score: Confidence = 0.0
    keep: bool = True
    notes: str = ""


class EvidenceRankingResult(BaseModel):
    """Output of the evidence-verification step: a per-item appraisal plus an
    overall read of how well the retrieved evidence supports the differential."""
    appraisals: list[EvidenceAppraisal] = Field(default_factory=list)
    overall: str = ""


class PlannedQuery(BaseModel):
    """One targeted literature query the query-planner produced for a hypothesis."""
    hypothesis: str = ""
    query: str = ""          # a boolean search expression (e.g. for PubMed)
    rationale: str = ""


class QueryPlanResult(BaseModel):
    """The query-planner's output: one focused query per leading hypothesis,
    replacing the old naive OR-of-condition-names search."""
    queries: list[PlannedQuery] = Field(default_factory=list)


class PaperAppraisal(BaseModel):
    """The per-paper verdict from the appraisal agent (one call per paper). Carries
    the EXTERNAL finding the paper contributes, so synthesis and the diagnosis can
    be grounded on specifics rather than a truncated abstract."""
    identifier: str = ""
    quality_score: Confidence = 0.0
    keep: bool = True
    supports: list[str] = Field(default_factory=list)  # hypotheses it supports
    refutes: list[str] = Field(default_factory=list)   # hypotheses it argues against
    key_finding: str = ""    # a concrete result/statistic/distinguishing feature
    notes: str = ""


class HypothesisEvidence(BaseModel):
    """Evidence gathered for ONE hypothesis, as organised by the synthesis agent."""
    condition: str = ""
    supporting_refs: list[str] = Field(default_factory=list)
    refuting_refs: list[str] = Field(default_factory=list)
    key_finding: str = ""
    strength: str = ""       # strong / moderate / weak / none


class EvidenceSynthesisResult(BaseModel):
    """The synthesis agent's decision-ready brief: the appraised papers combined
    and organised by hypothesis. This — not raw abstracts — is what feeds the
    working diagnosis, so external evidence can actually move the conclusion."""
    by_hypothesis: list[HypothesisEvidence] = Field(default_factory=list)
    contradictions: str = ""
    overall: str = ""


class MedicationSafetyResult(BaseModel):
    """Interaction / contraindication / allergy concerns relevant to the
    differential and any likely treatment, given the patient's active meds.

    Smaller local models often structure their responses more than GPT-4o and
    return concerns as objects with condition+explanation. Accept both for
    robustness: the node flattens them to strings for the downstream summary."""
    has_concern: bool = False
    concerns: list[str | dict] = Field(default_factory=list)  # type: ignore[assignment]
    summary: str = ""


class WorkingDiagnosisResult(BaseModel):
    """The synthesised conclusion. The node attaches the differential and evidence
    from the blackboard; the model supplies the calibrated confidence, the
    harm-if-wrong tier, the reasoning, and the confirmation pathway."""
    condition: str = ""
    confidence: Confidence = 0.0
    risk_tier: RiskTier = RiskTier.MODERATE
    reasoning: str = ""
    confirmation_pathway: str = ""


class ReportResult(BaseModel):
    """The clinician-facing report body. Status/timestamps are set by the node,
    not the model — the model only writes the prose."""
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, v: object) -> str:
        if isinstance(v, dict):
            return "\n\n".join(
                f"**{k}**\n{val}" for k, val in v.items()
            )
        return str(v) if v is not None else ""

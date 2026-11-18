"""
app/core/state.py
-----------------
The SHARED STATE — the blackboard.

This is the heart of the architecture. Every agent, path, and layer coordinates
by reading from and writing structured updates back to these models. No agent
calls another agent; they communicate only through this shared state. That single
rule is what keeps a multi-agent system from collapsing into the tangled,
hard-to-debug complexity that the literature (and our own design constraints)
identify as the central failure mode of MAS.

The state is split into THREE tiers by LIFETIME, not by feature:

    PatientRecord  (permanent)  — survives across all visits. The patient's
                                  long-term medical memory.
    VisitState     (per visit)  — one clinical encounter. Persisted when the
                                  visit ends so a patient can return to it.
    SessionState   (per chat)   — one conversation session. The most ephemeral.

Splitting by lifetime (rather than by path) is deliberate: lifetime is
path-independent, so the SAME three tiers serve every clinical path — diagnosis
today, treatment / Q&A / follow-up / test-review later — without duplication.
A new path reads and writes these same tiers; it does not get its own state.

`ConsultationState` bundles one instance of each tier into the single object an
agent receives for a turn. It is reconstructed from persistence at the start of
each turn and written back after.

Design choices (and why):
    - Pydantic v2 for validation + JSON round-tripping. Persistence depends on
      `model_dump_json()` / `model_validate_json()` working losslessly.
    - `validate_assignment=True`: the blackboard is MUTATED IN PLACE by agents,
      so we want an illegal write (e.g. confidence = 1.5) to fail at the moment
      of assignment, not silently corrupt state.
    - Models are NOT frozen. The blackboard is mutable by design. The
      "agents see a stable snapshot during their own run" guarantee is provided
      later by the executor (which snapshots context), not by freezing here.
    - No business logic and no I/O live in this file. It is pure data + a few
      small, obviously-useful accessors. Keeping it logic-free means it has zero
      coupling to OpenAI, the database, or any framework.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import (
    DiagnosticPhase,
    MessageRole,
    ReportStatus,
    RiskTier,
    Route,
    Sex,
    TriageLevel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp. Used as the default for all time fields."""
    return datetime.now(timezone.utc)


# A confidence/probability in [0, 1]. Declared once, reused everywhere, so the
# bound is enforced consistently and a value outside the range is a validation
# error rather than a downstream surprise.
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class _Base(BaseModel):
    """
    Shared configuration for every model in the system.

    validate_assignment=True is the important one: because agents write to the
    blackboard field-by-field, we re-validate on every assignment so bad writes
    fail loudly and immediately.
    """
    model_config = ConfigDict(validate_assignment=True)


# ─────────────────────────────────────────────────────────────────────────────
# Small value objects (the building blocks the tiers are composed from)
# ─────────────────────────────────────────────────────────────────────────────

class Message(_Base):
    """One entry in the conversation transcript."""
    role: MessageRole
    content: str
    route: Route | None = None          # which path produced/handled this turn
    timestamp: datetime = Field(default_factory=_utcnow)


class Medication(_Base):
    """A medication on the patient's record. `active` drives the safety check."""
    name: str
    dose: str | None = None
    frequency: str | None = None
    active: bool = True


class Allergy(_Base):
    """A known allergy. Checked against any recommendation before it is issued."""
    substance: str
    reaction: str | None = None
    severity: str | None = None


class Hypothesis(_Base):
    """
    One candidate condition in the working differential.

    Maintained cheaply during the interview loop and used to choose the next
    most-informative question (the question that best separates live hypotheses).
    """
    condition: str
    confidence: Confidence = 0.0
    supporting: list[str] = Field(default_factory=list)  # findings that support it
    against: list[str] = Field(default_factory=list)     # findings that argue against


class RecommendedTest(_Base):
    """
    A lab/imaging/exam the system suggests to reduce diagnostic uncertainty.

    `discriminates` records which hypotheses the test would separate — this is
    what makes test recommendation principled rather than a generic checklist.
    """
    name: str
    rationale: str
    discriminates: list[str] = Field(default_factory=list)


class EvidenceItem(_Base):
    """
    A retrieved piece of literature (e.g. a PubMed article).

    Quality fields are populated by the Evidence Ranking & Verification step;
    they start empty at retrieval time and are filled in before synthesis.
    """
    source: str                                 # e.g. "PubMed"
    title: str
    identifier: str                             # PMID / DOI / URL
    summary: str = ""
    published_year: int | None = None
    study_type: str | None = None               # e.g. "RCT", "meta-analysis"
    quality_score: Confidence | None = None     # set by verification step
    quality_notes: str | None = None


class WorkingDiagnosis(_Base):
    """
    The converged, most-probable conclusion — held as a *working* diagnosis
    pending confirmation, never asserted as a final verdict.

    Carries its calibrated confidence, its harm-if-wrong tier (which gates
    whether closure is even permitted), the differential it emerged from, the
    evidence backing it, and the concrete pathway that would confirm it.
    """
    condition: str
    confidence: Confidence
    risk_tier: RiskTier
    reasoning: str
    differential: list[Hypothesis] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confirmation_pathway: str = ""              # the test/exam that would confirm


class SafetyResult(_Base):
    """
    Output of the red-flag/emergency check that runs on EVERY turn, before the
    router. If `triage_level` is EMERGENCY/URGENT, `advice` holds the escalation
    message and the consultation is interrupted.
    """
    triage_level: TriageLevel
    red_flags: list[str] = Field(default_factory=list)
    advice: str | None = None
    checked_at: datetime = Field(default_factory=_utcnow)


class Report(_Base):
    """
    The clinical report and its approval state. Has no authority until APPROVED.
    `doctor_notes` carries rejection feedback that drives revision.
    """
    status: ReportStatus = ReportStatus.DRAFT
    content: str | None = None
    generated_at: datetime | None = None
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    doctor_notes: str | None = None


class Prescription(_Base):
    """
    A STRUCTURED medication order proposed by the system, carrying its own approval gate.
    It has NO authority until a clinician APPROVES it (mirrors `Report`): the patient is told
    only a preliminary suggestion to confirm with a clinician, while the doctor independently
    approves, edits, or rejects each prescription. `prescription_id` lets the doctor UI act on
    one specific item.
    """
    prescription_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    medication: str
    dose: str | None = None
    frequency: str | None = None
    rationale: str = ""
    status: ReportStatus = ReportStatus.PENDING_APPROVAL
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    doctor_notes: str | None = None


class PastVisitSummary(_Base):
    """
    A lightweight pointer to a completed visit, stored on the permanent record
    so later visits (and the follow-up path) can see what happened before
    without loading the entire historical VisitState.
    """
    visit_id: str
    date: datetime
    chief_complaint: str = ""
    working_diagnosis: str | None = None
    report_status: ReportStatus = ReportStatus.DRAFT


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — PatientRecord (permanent)
# ─────────────────────────────────────────────────────────────────────────────

class PatientRecord(_Base):
    """
    The patient's permanent medical memory. Loaded at the start of every visit
    and persisted as it changes. Path-agnostic: it holds nothing specific to the
    diagnosis path, so every path reads the same record.
    """
    patient_id: str

    # Demographics
    age: int | None = None
    sex: Sex = Sex.UNKNOWN

    # Preferred conversation language ("ar" | "en"; "" = not chosen yet). Asked once, then
    # LOCKED for every later visit: the concierge greets and replies in this language and only
    # an EXPLICIT patient request changes it (mid-conversation script changes must not).
    preferred_language: str = ""

    # Longitudinal medical memory (the "remember the patient" requirement)
    chronic_conditions: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    past_diagnoses: list[str] = Field(default_factory=list)
    visit_history: list[PastVisitSummary] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @property
    def active_medications(self) -> list[Medication]:
        """Currently-active meds — the list the medication-safety check uses."""
        return [m for m in self.medications if m.active]


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — VisitState (per encounter)
# ─────────────────────────────────────────────────────────────────────────────

class VisitState(_Base):
    """
    One clinical encounter. Persisted when the visit ends so the patient can
    return and continue (e.g. coming back with a lab result the system asked for).

    Diagnosis-specific working fields live here, but the tier itself is
    path-agnostic: a treatment / follow-up / review visit reuses the same object
    and simply fills different fields.
    """
    visit_id: str
    patient_id: str
    route: Route = Route.UNKNOWN              # which path this encounter is on

    # Cross-path workflow (suspend/resume). When a path SUSPENDS itself to let
    # another path run for a while — e.g. the Diagnosis path pausing for a
    # Test-Review round trip — it records here the route to resume once the
    # interim path is done. The interim path READS this to decide where to hand
    # control back; it never learns who suspended it (same blackboard discipline
    # as the agents). None when no resume is pending. It lives on the VISIT, not
    # the session, so a round trip survives even across sessions of one encounter.
    resume_route: Route | None = None

    chief_complaint: str | None = None

    # Diagnostic working state (filled as the encounter progresses)
    structured_symptoms: dict[str, Any] = Field(default_factory=dict)
    differential: list[Hypothesis] = Field(default_factory=list)
    recommended_tests: list[RecommendedTest] = Field(default_factory=list)
    # Objective lab results entered by the patient (canonical lab name -> value),
    # written by the LAB_REVIEW path to close the diagnosis loop. Additive and
    # migration-free: visits persist as JSON, so visits saved before this field
    # existed load with an empty dict.
    lab_results: dict[str, float] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    working_diagnosis: WorkingDiagnosis | None = None

    # Seam for the future image-analysis agent. Typed loosely on purpose: we do
    # not yet know the finding schema, and inventing one now would be premature.
    image_findings: list[dict[str, Any]] = Field(default_factory=list)

    report: Report | None = None

    # Structured medication orders proposed by the system, each pending clinician approval.
    # Additive and migration-free (visits persist as JSON), like `lab_results` above.
    prescriptions: list[Prescription] = Field(default_factory=list)

    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None

    @property
    def top_hypothesis(self) -> Hypothesis | None:
        """Highest-confidence hypothesis, or None if the differential is empty.
        Used by the sufficiency gate to decide whether to converge."""
        if not self.differential:
            return None
        return max(self.differential, key=lambda h: h.confidence)


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — SessionState (per chat session)
# ─────────────────────────────────────────────────────────────────────────────

class SessionState(_Base):
    """
    One conversation session. The most ephemeral tier. Holds the live transcript,
    the current phase, the active path, and the loop-guard counter.
    """
    session_id: str
    patient_id: str
    visit_id: str

    # Translation seam — kept so the field exists from day one, but the
    # translation layer itself is future work. Defaults to English.
    language: str = "en"

    phase: DiagnosticPhase = DiagnosticPhase.INTAKE
    active_route: Route = Route.UNKNOWN

    transcript: list[Message] = Field(default_factory=list)

    # Guardrail for the sufficiency gate: capped so the interview can never loop
    # forever. When this hits its limit the path forces progression or escalates.
    interview_iterations: int = 0

    # Result of the most recent red-flag check, for audit and quick inspection.
    last_safety: SafetyResult | None = None

    started_at: datetime = Field(default_factory=_utcnow)

    def add_message(
        self,
        role: MessageRole,
        content: str,
        route: Route | None = None,
    ) -> Message:
        """
        Append a message to the transcript and return it.

        Centralising this keeps timestamping consistent and means callers never
        construct transcript entries by hand. (Re-assigning the list triggers
        validate_assignment, so the transcript stays valid.)
        """
        message = Message(role=role, content=content, route=route)
        self.transcript = [*self.transcript, message]
        return message


# ─────────────────────────────────────────────────────────────────────────────
# The blackboard handle passed to agents for a single turn
# ─────────────────────────────────────────────────────────────────────────────

class ConsultationState(_Base):
    """
    The blackboard for one turn: one of each lifetime tier bundled together.

    This is the single object an agent receives. An agent reads what it needs
    from the three tiers and writes its structured contribution back to the
    appropriate tier. Reconstructed from persistence at the start of each turn,
    written back after.
    """
    patient: PatientRecord
    visit: VisitState
    session: SessionState

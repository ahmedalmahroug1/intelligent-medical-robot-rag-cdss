"""
app/core/enums.py
-----------------
Controlled vocabularies used across the whole system.

Why a dedicated module:
    These enums are the shared "words" every layer agrees on — the router,
    the safety guard, the diagnostic path, persistence, and the API all speak
    in these terms. Keeping them in one place (instead of scattering string
    literals like "emergency" or "approved" through the codebase) means a typo
    becomes an ImportError instead of a silent bug, and the set of legal values
    is defined exactly once.

All enums inherit from `str` so that:
    - they serialize to plain strings in JSON (clean DB rows, clean API), and
    - `state.session.phase == "intake"` works without `.value` juggling.
"""

from enum import Enum


class Sex(str, Enum):
    """Biological sex, relevant to clinical reasoning. UNKNOWN until provided."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class Route(str, Enum):
    """
    Which clinical PATH handles a turn.

    The Intake Router classifies each patient turn into exactly one of these.
    DIAGNOSIS is the first path we build; the others are registered later and
    plug into the same router + blackboard with no change to existing paths.
    UNKNOWN means the router could not classify the turn (→ ask to clarify).
    """
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    QA = "qa"
    FOLLOW_UP = "follow_up"
    TEST_REVIEW = "test_review"
    UNKNOWN = "unknown"


class DiagnosticPhase(str, Enum):
    """
    Position within the DIAGNOSIS path's state machine.

    INTAKE/INTERVIEW are the cheap loop (one streamed LLM call per turn).
    MILESTONE is the expensive, run-once evidence pipeline (executor fan-out).
    ESCALATED is a terminal safety state entered when the guard fires; it can
    be reached from ANY phase and overrides normal progression.
    """
    INTAKE = "intake"           # establishing the chief complaint / understanding
    INTERVIEW = "interview"     # looping: hypotheses → next question → answer
    MILESTONE = "milestone"     # sufficiency met → evidence retrieval + synthesis
    COMPLETE = "complete"       # working diagnosis produced, report drafted
    ESCALATED = "escalated"     # red-flag fired → emergency escalation (terminal)


class TriageLevel(str, Enum):
    """
    Output of the safety guard's red-flag check, run on EVERY turn.

    Ordered by urgency. EMERGENCY interrupts the whole flow and tells the
    patient to seek emergency care immediately; nothing downstream overrides it.
    """
    EMERGENCY = "emergency"     # life-threatening → seek emergency care now
    URGENT = "urgent"           # seek care within hours
    ROUTINE = "routine"         # safe to continue the consultation


class RiskTier(str, Enum):
    """
    Harm-if-wrong tier for a candidate conclusion.

    Convergence is gated on confidence AND this tier: a HIGH/CRITICAL tier
    forces routing to in-person care even when model confidence is high.
    Safety scales with stakes, not probability alone.
    """
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ReportStatus(str, Enum):
    """
    State machine for the doctor-approval gate.

    A report has no clinical authority until APPROVED. REJECTED carries the
    doctor's notes and sends the report back for revision — the rejection path
    is explicit, not an afterthought.
    """
    DRAFT = "draft"                     # generated, not yet submitted
    PENDING_APPROVAL = "pending_approval"  # awaiting doctor review
    APPROVED = "approved"              # doctor signed off → has authority
    REJECTED = "rejected"             # doctor rejected with notes → revise


class MessageRole(str, Enum):
    """
    Author of a transcript message.

    DOCTOR_AI is the single "doctor" persona the patient sees, regardless of
    how many capabilities ran underneath to produce the turn.
    """
    PATIENT = "patient"
    DOCTOR_AI = "doctor_ai"
    SYSTEM = "system"

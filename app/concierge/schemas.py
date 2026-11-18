"""
app/concierge/schemas.py
------------------------
Concierge-side contracts.

  IntakeExtraction — what the concierge extracts from ONE patient message (Arabic or English),
                     then merges into the running ClinicalPicture. All-default, so a partial or
                     failed parse is safe. Clinical content is English-canonical; original_text
                     keeps the verbatim message.
  ClinicalPicture  — the accumulated, structured picture the concierge OWNS across the whole
                     interview. It is what gets sent to the CDSS reasoning engine.
  AssessResult     — the CDSS reasoning engine's view of the current picture (it wraps the
                     domain IntakeResult): differential + the single suggested question +
                     the engine's own sufficiency/confidence.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.state import Hypothesis, RecommendedTest, Report, WorkingDiagnosis


class IdentityParse(BaseModel):
    """LLM-driven name detection. `has_name=False` when the message is a greeting, a request
    ('speak in english'), a question, or small-talk — so a non-name is never taken as the name."""
    name: str = ""
    has_name: bool = False
    language: str = "en"


class IntakeExtraction(BaseModel):
    original_text: str = ""                                   # verbatim patient message
    language: str = "en"                                      # "ar" | "en"
    # False when the message carries NO new clinical information (greeting, thanks, language
    # preference, "are you a doctor?", frustration, small-talk) — answered by the concierge,
    # never forwarded to the reasoning engine.
    is_clinical: bool = True
    chief_complaint: str = ""                                 # short English phrase
    symptoms: list[str] = Field(default_factory=list)         # English terms
    duration: str = ""                                        # e.g. "2 days"
    severity: str = ""                                        # e.g. "7/10", "mild"
    location: str = ""                                        # e.g. "left temple"
    associated_symptoms: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    known_conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    age: int | None = None
    sex: str = ""                                             # "male" | "female" | ""


class ClinicalPicture(BaseModel):
    """The accumulated structured picture the concierge owns and grows across the interview."""
    chief_complaint: str = ""
    symptoms: list[str] = Field(default_factory=list)
    duration: str = ""
    severity: str = ""
    location: str = ""
    associated_symptoms: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    chronic_conditions: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    age: int | None = None
    sex: str = ""
    language: str = "en"

    def merge(self, e: IntakeExtraction) -> None:
        """Fold one turn's extraction into the running picture (last non-empty wins for scalars;
        union for lists)."""
        if e.chief_complaint and not self.chief_complaint:
            self.chief_complaint = e.chief_complaint
        for s in ("duration", "severity", "location"):
            val = getattr(e, s)
            if val:
                setattr(self, s, val)
        if e.age is not None:
            self.age = e.age
        if e.sex in ("male", "female"):
            self.sex = e.sex
        if e.language:
            self.language = e.language
        _extend(self.symptoms, e.symptoms)
        _extend(self.associated_symptoms, e.associated_symptoms)
        _extend(self.medications, e.medications)
        _extend(self.allergies, e.allergies)
        _extend(self.chronic_conditions, e.known_conditions)
        _extend(self.family_history, e.family_history)


class AssessResult(BaseModel):
    """The reasoning engine's view of the current picture (wraps the domain IntakeResult)."""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    suggested_question: str = ""
    sufficient: bool = False
    confidence: float = 0.0
    understanding: str = ""


class FinalizeResult(BaseModel):
    """The CDSS's final output for a mature picture (from the existing milestone): the working
    diagnosis, the clinician report, and recommended tests. The concierge PRESENTS these — it
    does not produce them. Empty (working_diagnosis=None) signals a failed run → safe fallback."""
    working_diagnosis: WorkingDiagnosis | None = None
    report: Report | None = None
    recommended_tests: list[RecommendedTest] = Field(default_factory=list)


def _extend(target: list[str], items: list[str]) -> None:
    seen = {t.strip().lower() for t in target}
    for it in items:
        if it and it.strip().lower() not in seen:
            target.append(it)
            seen.add(it.strip().lower())

"""
tests/test_state.py
-------------------
Tests for the shared-state (blackboard) models — File 1.

What these tests prove, and why each matters for the architecture:
    1. The three tiers and the ConsultationState handle construct correctly.
    2. Bounded fields (confidence) reject out-of-range values  → data integrity.
    3. validate_assignment catches bad WRITES, not just bad construction
       → the blackboard cannot be silently corrupted by an agent.
    4. JSON round-trips losslessly (dump → load == original)
       → the persistence layer (File: persistence) can store/restore state.
    5. Enums serialize as plain strings → clean DB rows and API payloads.
    6. The small accessors (add_message, active_medications, top_hypothesis)
       behave as the diagnostic path will rely on.

Run:
    pip install pydantic
    python -m tests.test_state          # plain-script mode (no pytest needed)
    # or, if you have pytest:
    pytest tests/test_state.py -v
"""

from datetime import datetime, timezone

from pydantic import ValidationError

from app.core.enums import (
    DiagnosticPhase,
    MessageRole,
    ReportStatus,
    RiskTier,
    Route,
    Sex,
    TriageLevel,
)
from app.core.state import (
    Allergy,
    ConsultationState,
    EvidenceItem,
    Hypothesis,
    Medication,
    PatientRecord,
    Report,
    SafetyResult,
    SessionState,
    VisitState,
    WorkingDiagnosis,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures-as-functions (kept framework-light so the file also runs standalone)
# ─────────────────────────────────────────────────────────────────────────────

def make_patient() -> PatientRecord:
    return PatientRecord(
        patient_id="PT-001",
        age=67,
        sex=Sex.MALE,
        chronic_conditions=["hypertension", "type 2 diabetes"],
        medications=[
            Medication(name="lisinopril", dose="10mg", frequency="daily"),
            Medication(name="metformin", dose="500mg", frequency="twice daily"),
            Medication(name="old-statin", active=False),  # discontinued
        ],
        allergies=[Allergy(substance="penicillin", reaction="rash", severity="moderate")],
        past_diagnoses=["hyperlipidaemia"],
    )


def make_visit() -> VisitState:
    return VisitState(
        visit_id="V-001",
        patient_id="PT-001",
        route=Route.DIAGNOSIS,
        chief_complaint="chest pain for two days",
    )


def make_session() -> SessionState:
    return SessionState(
        session_id="S-001",
        patient_id="PT-001",
        visit_id="V-001",
    )


def make_consultation() -> ConsultationState:
    return ConsultationState(
        patient=make_patient(),
        visit=make_visit(),
        session=make_session(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Construction
# ─────────────────────────────────────────────────────────────────────────────

def test_consultation_constructs_with_three_tiers():
    c = make_consultation()
    assert c.patient.patient_id == "PT-001"
    assert c.visit.visit_id == "V-001"
    assert c.session.session_id == "S-001"
    # Defaults applied correctly
    assert c.session.phase is DiagnosticPhase.INTAKE
    assert c.session.active_route is Route.UNKNOWN
    assert c.session.language == "en"
    assert c.session.interview_iterations == 0
    assert c.visit.working_diagnosis is None
    assert c.visit.report is None


def test_defaults_do_not_share_mutable_state():
    # Two fresh visits must not share the same list instance (classic dataclass
    # footgun that default_factory avoids — verify it really is avoided).
    v1 = make_visit()
    v2 = make_visit()
    v1.differential.append(Hypothesis(condition="angina"))
    assert v1.differential != v2.differential
    assert v2.differential == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bounded-field validation
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_above_one_is_rejected():
    try:
        Hypothesis(condition="x", confidence=1.5)
    except ValidationError:
        pass
    else:
        raise AssertionError("confidence=1.5 should have raised ValidationError")


def test_confidence_below_zero_is_rejected():
    try:
        Hypothesis(condition="x", confidence=-0.1)
    except ValidationError:
        pass
    else:
        raise AssertionError("confidence=-0.1 should have raised ValidationError")


def test_confidence_within_range_is_accepted():
    h = Hypothesis(condition="angina", confidence=0.7)
    assert h.confidence == 0.7


# ─────────────────────────────────────────────────────────────────────────────
# 3. validate_assignment — bad WRITES to the blackboard fail immediately
# ─────────────────────────────────────────────────────────────────────────────

def test_assignment_is_validated():
    # This is the property that protects the mutable blackboard: an agent writing
    # an illegal value must fail at the moment of the write.
    h = Hypothesis(condition="angina", confidence=0.5)
    try:
        h.confidence = 2.0
    except ValidationError:
        pass
    else:
        raise AssertionError("assigning confidence=2.0 should have raised")
    # Original value is unchanged after the rejected write.
    assert h.confidence == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 4. JSON round-trip (persistence depends on this)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_state_json_round_trip():
    c = make_consultation()

    # Populate a rich, realistic state so the round-trip exercises nested models.
    c.session.add_message(MessageRole.PATIENT, "I have chest pain", Route.DIAGNOSIS)
    c.session.add_message(MessageRole.DOCTOR_AI, "How long has it lasted?", Route.DIAGNOSIS)
    c.session.phase = DiagnosticPhase.INTERVIEW
    c.session.active_route = Route.DIAGNOSIS
    c.session.interview_iterations = 2
    c.session.last_safety = SafetyResult(
        triage_level=TriageLevel.URGENT,
        red_flags=["chest pain in diabetic patient"],
        advice="Seek care promptly.",
    )

    c.visit.differential = [
        Hypothesis(condition="stable angina", confidence=0.55, supporting=["exertional"]),
        Hypothesis(condition="GERD", confidence=0.25, against=["radiates to arm"]),
    ]
    c.visit.evidence = [
        EvidenceItem(
            source="PubMed",
            title="Management of stable angina",
            identifier="PMID:123456",
            study_type="meta-analysis",
            published_year=2021,
            quality_score=0.9,
        )
    ]
    c.visit.working_diagnosis = WorkingDiagnosis(
        condition="stable angina",
        confidence=0.7,
        risk_tier=RiskTier.HIGH,
        reasoning="Exertional chest pain in a 67-year-old diabetic.",
        confirmation_pathway="stress test / troponin",
    )
    c.visit.report = Report(status=ReportStatus.PENDING_APPROVAL, content="draft text")

    # Round-trip: serialize to JSON, then rebuild from that JSON.
    payload = c.model_dump_json()
    restored = ConsultationState.model_validate_json(payload)

    # Deep equality: the rebuilt object must equal the original exactly.
    assert restored == c
    # Spot-check a few nested values survived with correct types.
    assert restored.visit.working_diagnosis.risk_tier is RiskTier.HIGH
    assert restored.session.last_safety.triage_level is TriageLevel.URGENT
    assert len(restored.session.transcript) == 2
    assert restored.visit.evidence[0].quality_score == 0.9


# ─────────────────────────────────────────────────────────────────────────────
# 5. Enums serialize as plain strings (clean DB rows / API)
# ─────────────────────────────────────────────────────────────────────────────

def test_enums_serialize_as_strings():
    c = make_consultation()
    c.session.phase = DiagnosticPhase.MILESTONE
    data = c.model_dump(mode="json")
    assert data["session"]["phase"] == "milestone"
    assert data["visit"]["route"] == "diagnosis"
    assert data["patient"]["sex"] == "male"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Accessors the diagnostic path will rely on
# ─────────────────────────────────────────────────────────────────────────────

def test_active_medications_excludes_inactive():
    p = make_patient()
    names = [m.name for m in p.active_medications]
    assert "lisinopril" in names
    assert "metformin" in names
    assert "old-statin" not in names      # active=False → excluded


def test_top_hypothesis_returns_highest_confidence():
    v = make_visit()
    assert v.top_hypothesis is None       # empty differential
    v.differential = [
        Hypothesis(condition="GERD", confidence=0.3),
        Hypothesis(condition="angina", confidence=0.6),
        Hypothesis(condition="anxiety", confidence=0.1),
    ]
    assert v.top_hypothesis.condition == "angina"


def test_add_message_appends_and_timestamps():
    s = make_session()
    before = datetime.now(timezone.utc)
    msg = s.add_message(MessageRole.PATIENT, "hello", Route.DIAGNOSIS)
    assert len(s.transcript) == 1
    assert s.transcript[0] is msg
    assert msg.role is MessageRole.PATIENT
    assert msg.route is Route.DIAGNOSIS
    assert msg.timestamp >= before        # timestamp was set on creation


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (so `python -m tests.test_state` works without pytest)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
        except Exception:
            failed += 1
            print(f"[FAIL]  {t.__name__}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[PASS]  {t.__name__}")

    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)

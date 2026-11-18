"""
tests/test_safety_rules.py
--------------------------
Tests for the deterministic red-flag layer — part of File 4.

This layer is stdlib-only (no pydantic, no gateway), so these tests run fully
offline and verify the safety NET independently of the LLM. They prove the
high-recall behaviour, the severity-maximum combination, and the helper ordering.

Run:
    python -m tests.test_safety_rules
"""

from __future__ import annotations

from app.core.enums import TriageLevel
from app.safety.rules import evaluate_rules, max_triage, severity_rank


# ── Emergency detection ──────────────────────────────────────────────────────

def test_crushing_chest_pain_is_emergency():
    f = evaluate_rules("I have crushing chest pain radiating to my left arm")
    assert f.level is TriageLevel.EMERGENCY
    assert "cardiac" in f.categories


def test_cant_breathe_is_emergency():
    assert evaluate_rules("I can't breathe").level is TriageLevel.EMERGENCY


def test_stroke_signs_are_emergency():
    f = evaluate_rules("my face is drooping and I have slurred speech")
    assert f.level is TriageLevel.EMERGENCY
    assert "neuro" in f.categories


def test_suicidal_ideation_is_emergency_and_flagged_self_harm():
    f = evaluate_rules("I want to die")
    assert f.level is TriageLevel.EMERGENCY
    assert "self_harm" in f.categories


def test_unconscious_is_emergency():
    assert evaluate_rules("my husband passed out and is unresponsive").level is TriageLevel.EMERGENCY


def test_anaphylaxis_is_emergency():
    assert evaluate_rules("my throat is closing").level is TriageLevel.EMERGENCY


# ── Urgent detection ─────────────────────────────────────────────────────────

def test_plain_chest_pain_is_urgent():
    f = evaluate_rules("I've had some chest pain since this morning")
    assert f.level is TriageLevel.URGENT
    assert "cardiac" in f.categories


def test_shortness_of_breath_is_urgent():
    assert evaluate_rules("I have shortness of breath").level is TriageLevel.URGENT


# ── Routine ──────────────────────────────────────────────────────────────────

def test_benign_message_is_routine():
    f = evaluate_rules("I have a mild runny nose and a slight sore throat")
    assert f.level is TriageLevel.ROUTINE
    assert f.reasons == ()
    assert f.categories == ()


def test_directly_negated_chest_pain_is_not_flagged():
    assert evaluate_rules("I have no chest pain").level is TriageLevel.ROUTINE


def test_directly_negated_arabic_chest_pain_is_not_flagged():
    assert evaluate_rules("لا يوجد ألم في الصدر").level is TriageLevel.ROUTINE


def test_not_breathing_remains_an_emergency_not_a_negation():
    assert evaluate_rules("The patient is not breathing").level is TriageLevel.EMERGENCY


def test_overdose_is_emergency():
    assert evaluate_rules("I took too many pills").level is TriageLevel.EMERGENCY


def test_pregnancy_bleeding_is_at_least_urgent():
    assert evaluate_rules("I am pregnant and bleeding").level is TriageLevel.URGENT


def test_infant_fever_has_vulnerability_floor():
    finding = evaluate_rules("The baby has a fever", age=0)
    assert finding.level is TriageLevel.URGENT
    assert "vulnerable_age" in finding.categories


# ── Combination behaviour ────────────────────────────────────────────────────

def test_max_severity_wins_when_multiple_match():
    # "chest pain" (urgent) + "radiating to arm" (emergency) → emergency overall.
    f = evaluate_rules("chest pain radiating to my arm")
    assert f.level is TriageLevel.EMERGENCY


def test_multiple_distinct_reasons_collected():
    f = evaluate_rules("I can't breathe and there is severe bleeding")
    assert f.level is TriageLevel.EMERGENCY
    assert len(f.reasons) >= 2


def test_case_insensitive():
    assert evaluate_rules("CRUSHING CHEST PAIN").level is TriageLevel.EMERGENCY


def test_reasons_are_deduplicated():
    # Two phrasings that both map to the respiratory category/reason.
    f = evaluate_rules("I cannot breathe, I really can't breathe")
    # Same reason should not be duplicated.
    assert len(f.reasons) == len(set(f.reasons))


# ── Helpers ──────────────────────────────────────────────────────────────────

def test_severity_ordering():
    assert severity_rank(TriageLevel.EMERGENCY) > severity_rank(TriageLevel.URGENT)
    assert severity_rank(TriageLevel.URGENT) > severity_rank(TriageLevel.ROUTINE)


def test_max_triage_picks_more_severe():
    assert max_triage(TriageLevel.ROUTINE, TriageLevel.EMERGENCY) is TriageLevel.EMERGENCY
    assert max_triage(TriageLevel.URGENT, TriageLevel.ROUTINE) is TriageLevel.URGENT
    assert max_triage(TriageLevel.URGENT, TriageLevel.URGENT) is TriageLevel.URGENT


# ── Standalone runner ────────────────────────────────────────────────────────

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

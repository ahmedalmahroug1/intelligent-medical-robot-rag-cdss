"""
tests/test_safety_guard.py
--------------------------
Tests for the SafetyGuard — part of File 4. These exercise the combination of the
deterministic rule layer with the (faked) LLM assessment.

They need pydantic (state models) and use the FakeGateway, so run them in your
environment:
    pip install -r requirements.txt
    python -m tests.test_safety_guard

What they prove:
    - rule layer is a floor the LLM cannot lower (rule EMERGENCY + LLM routine
      → EMERGENCY),
    - the LLM can raise the level (rule routine + LLM emergency → EMERGENCY),
    - benign + routine → ROUTINE with no advice,
    - NEVER FAILS OPEN: if the LLM call raises, the guard falls back to rules,
    - self-harm gets dedicated crisis advice,
    - advice falls back to a safe default when the model supplies none,
    - an unparseable LLM level is ignored (the rule floor still applies),
    - the guard does not mutate state.
"""

from __future__ import annotations

import asyncio

from app.core.enums import MessageRole, Route, Sex, TriageLevel
from app.core.state import (
    ConsultationState,
    PatientRecord,
    SessionState,
    VisitState,
)
from app.llm.base import LLMError
from app.safety.guard import _SELF_HARM_ADVICE, SafetyGuard
from app.safety.rules import evaluate_rules
from tests.fakes import FakeGateway


def _state(message_history: list[str] | None = None) -> ConsultationState:
    patient = PatientRecord(patient_id="PT-1", age=67, sex=Sex.MALE,
                            chronic_conditions=["hypertension"])
    visit = VisitState(visit_id="V-1", patient_id="PT-1", route=Route.DIAGNOSIS)
    session = SessionState(session_id="S-1", patient_id="PT-1", visit_id="V-1")
    for m in message_history or []:
        session.add_message(MessageRole.PATIENT, m)
    return ConsultationState(patient=patient, visit=visit, session=session)


def _routine_llm() -> FakeGateway:
    return FakeGateway(structured={"triage_level": "routine", "red_flags": [], "advice": None})


def _emergency_llm() -> FakeGateway:
    return FakeGateway(structured={
        "triage_level": "emergency",
        "red_flags": ["model-identified danger"],
        "advice": "Call emergency services now.",
    })


def test_arabic_red_flags_have_deterministic_coverage() -> None:
    assert evaluate_rules("لا أستطيع التنفس وأشعر باختناق").level is TriageLevel.EMERGENCY
    assert evaluate_rules("عندي ألم في الصدر").level is TriageLevel.URGENT


# ── Combination ──────────────────────────────────────────────────────────────

async def test_rule_emergency_holds_even_if_llm_says_routine():
    guard = SafetyGuard(_routine_llm())
    result = await guard.check(message="I have crushing chest pain", state=_state())
    assert result.triage_level is TriageLevel.EMERGENCY      # rule floor holds


async def test_llm_can_raise_level_above_rules():
    # Message the deterministic layer treats as routine, but the LLM flags.
    guard = SafetyGuard(_emergency_llm())
    result = await guard.check(message="something feels deeply wrong inside", state=_state())
    assert result.triage_level is TriageLevel.EMERGENCY
    assert "model-identified danger" in result.red_flags


async def test_benign_and_routine_is_routine_with_no_advice():
    guard = SafetyGuard(_routine_llm())
    result = await guard.check(message="I have a mild runny nose", state=_state())
    assert result.triage_level is TriageLevel.ROUTINE
    assert result.advice is None
    assert result.red_flags == []


# ── Never fails open ─────────────────────────────────────────────────────────

async def test_guard_falls_back_to_rules_when_llm_fails():
    guard = SafetyGuard(FakeGateway(error=LLMError("LLM is down")))
    # Deterministic layer still catches the emergency.
    result = await guard.check(message="I can't breathe", state=_state())
    assert result.triage_level is TriageLevel.EMERGENCY
    assert result.advice is not None


async def test_guard_does_not_crash_on_llm_failure_for_benign_message():
    guard = SafetyGuard(FakeGateway(error=LLMError("LLM is down")))
    result = await guard.check(message="just a routine question about diet", state=_state())
    assert result.triage_level is TriageLevel.ROUTINE   # rules say nothing, no crash


# ── Advice selection ─────────────────────────────────────────────────────────

async def test_self_harm_uses_crisis_advice():
    guard = SafetyGuard(_routine_llm())
    result = await guard.check(message="I want to die", state=_state())
    assert result.triage_level is TriageLevel.EMERGENCY
    assert result.advice == _SELF_HARM_ADVICE


async def test_arabic_emergency_gets_arabic_deterministic_advice():
    guard = SafetyGuard(_routine_llm())
    result = await guard.check(message="لا أستطيع التنفس", state=_state())
    assert result.triage_level is TriageLevel.EMERGENCY
    assert result.advice is not None and "الطوارئ" in result.advice


async def test_advice_defaults_when_model_gives_none():
    # LLM raises the level but supplies no advice → guard fills a safe default.
    gw = FakeGateway(structured={"triage_level": "urgent", "red_flags": ["x"], "advice": None})
    guard = SafetyGuard(gw)
    result = await guard.check(message="something feels off", state=_state())
    assert result.triage_level is TriageLevel.URGENT
    assert result.advice is not None and "professional" in result.advice.lower()


async def test_unparseable_llm_level_is_ignored_but_rules_apply():
    gw = FakeGateway(structured={"triage_level": "??garbage??", "red_flags": [], "advice": None})
    guard = SafetyGuard(gw)
    # Rules still catch the emergency despite the unusable LLM level.
    result = await guard.check(message="severe bleeding that won't stop", state=_state())
    assert result.triage_level is TriageLevel.EMERGENCY


# ── Purity ───────────────────────────────────────────────────────────────────

async def test_guard_does_not_mutate_state():
    guard = SafetyGuard(_routine_llm())
    state = _state(["earlier message"])
    transcript_len_before = len(state.session.transcript)
    last_safety_before = state.session.last_safety
    await guard.check(message="I have a headache", state=state)
    assert len(state.session.transcript) == transcript_len_before
    assert state.session.last_safety is last_safety_before   # unchanged (None)


# ── Standalone runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import inspect
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            result = t()
            if inspect.isawaitable(result):
                asyncio.run(result)
        except Exception:
            failed += 1
            print(f"[FAIL]  {t.__name__}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[PASS]  {t.__name__}")

    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)

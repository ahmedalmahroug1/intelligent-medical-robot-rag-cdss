"""
app/safety/guard.py
-------------------
The SafetyGuard: the red-flag / emergency check that runs on EVERY patient turn,
before the router, on every path. It can escalate a turn to emergency or urgent
care, and in the assembled system nothing reaches a clinical path without passing
through it.

How it decides (defense-in-depth):
    1. Deterministic rule layer (rules.py) — instant, offline, high-recall.
    2. LLM assessment — judges nuance the rules can't (negation, history,
       hypotheticals) and weighs patient age and comorbidities.
    The final triage is the MORE CAUTIOUS of the two. The rule layer is a floor
    the LLM can raise but never lower.

Critical property — it never fails open:
    If the LLM call fails or returns something unparseable, the guard falls back
    to the deterministic result instead of crashing or silently downgrading. A
    safety check must keep protecting the patient even when the model is
    unavailable. This is why the deterministic layer exists.

Purity:
    check() does NOT mutate state. It returns a SafetyResult; the caller (the
    orchestrator / router wiring built later) decides what to do with it — record
    it on the session, escalate, or proceed. Keeping the guard side-effect-free
    makes it trivial to test and reason about.

Contract note:
    `message` is the CURRENT patient turn and is assumed NOT yet appended to
    `state.session.transcript` — the transcript passed in is the prior history.
    This keeps the current turn from appearing twice in the prompt.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.config import settings
from app.core.enums import TriageLevel
from app.core.state import ConsultationState, SafetyResult
from app.llm.base import LLMGateway, ModelTier
from app.safety.rules import RuleFinding, evaluate_rules, max_triage

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a clinical safety triage screener. Your ONLY job is to \
judge whether a patient's message signals a medical emergency or an urgent \
situation needing prompt care. You are NOT diagnosing.

Classify the message into exactly one level:
- "emergency": signs of a life-threatening problem needing immediate emergency \
care (e.g. suspected heart attack, stroke, anaphylaxis, severe bleeding, severe \
breathing difficulty, loss of consciousness, active intent to self-harm).
- "urgent": concerning symptoms that should be assessed by a professional within \
hours, but not immediately life-threatening.
- "routine": no red flags; safe to continue a normal consultation.

Rules for judgement:
- Account for negation and context. "I have no chest pain" or "I had a headache \
last year" are NOT current red flags.
- Consider the patient's age and chronic conditions — the same symptom may be \
higher risk in an older patient with heart disease.
- If genuinely uncertain between two levels, choose the more cautious one.
- Do NOT over-escalate clearly benign messages (e.g. "I have a mild runny nose").

Return ONLY a JSON object with exactly these keys:
{
  "triage_level": "emergency" | "urgent" | "routine",
  "red_flags": ["short reason", ...],
  "advice": "one or two sentences on what to do now, or null if routine"
}"""


# Default patient-facing escalation messages, used when the model does not supply
# its own advice (or is unavailable). The self-harm message is handled separately.
_DEFAULT_ADVICE: dict[TriageLevel, str | None] = {
    TriageLevel.EMERGENCY: (
        "Based on what you've described, this may be a medical emergency. Please "
        "contact your local emergency services or go to the nearest emergency "
        "department right now."
    ),
    TriageLevel.URGENT: (
        "What you're describing should be checked by a medical professional soon. "
        "Please seek care within the next few hours rather than waiting."
    ),
    TriageLevel.ROUTINE: None,
}

_SELF_HARM_ADVICE = (
    "It sounds like you may be going through something extremely difficult, and "
    "you deserve immediate support from someone trained to help. Please contact "
    "your local emergency services or a crisis helpline right now, or reach out "
    "to someone you trust who can be with you."
)

_DEFAULT_ADVICE_AR: dict[TriageLevel, str | None] = {
    TriageLevel.EMERGENCY: (
        "قد تكون الأعراض التي وصفتها حالة طبية طارئة. اتصل بخدمات الطوارئ المحلية "
        "أو توجّه إلى أقرب قسم طوارئ الآن."
    ),
    TriageLevel.URGENT: (
        "تحتاج الأعراض التي وصفتها إلى تقييم طبي سريع. اطلب الرعاية خلال الساعات "
        "القليلة القادمة ولا تنتظر حتى موعد اعتيادي."
    ),
    TriageLevel.ROUTINE: None,
}

_SELF_HARM_ADVICE_AR = (
    "يبدو أنك تمر بوقت شديد الصعوبة وتستحق دعماً فورياً. اتصل بخدمات الطوارئ "
    "المحلية أو بخط مساعدة الأزمات الآن، واطلب من شخص تثق به أن يبقى معك."
)

_LEVEL_FROM_TEXT: dict[str, TriageLevel] = {
    "emergency": TriageLevel.EMERGENCY,
    "urgent": TriageLevel.URGENT,
    "routine": TriageLevel.ROUTINE,
}


class SafetyGuard:
    """Runs the red-flag check for a single turn and returns a SafetyResult."""

    def __init__(self, gateway: LLMGateway, *, model: ModelTier = ModelTier.FAST) -> None:
        # FAST by default: triage is high-frequency (every turn) and should be
        # low-latency. The deterministic layer backstops it regardless.
        self._gateway = gateway
        self._model = model

    async def check(self, *, message: str, state: ConsultationState) -> SafetyResult:
        if not settings.ENABLE_SAFETY_GUARD:
            logger.info("SafetyGuard bypassed via configuration (ENABLE_SAFETY_GUARD=False)")
            return SafetyResult(triage_level=TriageLevel.ROUTINE, red_flags=[], advice=None)

        rule_finding = evaluate_rules(
            message,
            age=state.patient.age,
            chronic_conditions=state.patient.chronic_conditions,
        )
        llm_level, llm_reasons, llm_advice = await self._llm_assess(
            message, state, rule_finding
        )

        level = max_triage(rule_finding.level, llm_level)
        reasons = list(dict.fromkeys([*rule_finding.reasons, *llm_reasons]))
        advice = self._advice_for(level, rule_finding, llm_advice, message)

        return SafetyResult(triage_level=level, red_flags=reasons, advice=advice)

    # ── LLM assessment (with graceful fallback) ──────────────────────────────

    async def _llm_assess(
        self,
        message: str,
        state: ConsultationState,
        rule_finding: RuleFinding,
    ) -> tuple[TriageLevel, list[str], str | None]:
        try:
            data = await asyncio.wait_for(
                self._gateway.complete_structured(
                    system=_SYSTEM_PROMPT,
                    user=self._build_user_prompt(message, state, rule_finding),
                    model=self._model,
                ),
                timeout=settings.SAFETY_LLM_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - safety must never fail open
            # Never fail open: drop to rules-only by contributing ROUTINE.
            logger.warning(
                "Safety LLM assessment unavailable; using deterministic rules only: %s",
                exc,
            )
            return TriageLevel.ROUTINE, [], None
        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> tuple[TriageLevel, list[str], str | None]:
        raw = str(data.get("triage_level", "")).strip().lower()
        level = _LEVEL_FROM_TEXT.get(raw)
        if level is None:
            logger.warning(
                "Safety LLM returned unknown triage_level=%r; ignoring its level", raw
            )
            level = TriageLevel.ROUTINE  # rule layer still applies as the floor

        flags = data.get("red_flags") or []
        if not isinstance(flags, list):
            flags = [str(flags)]
        reasons = [str(f) for f in flags]

        advice_raw = data.get("advice")
        advice = str(advice_raw) if advice_raw else None
        return level, reasons, advice

    # ── Advice selection ─────────────────────────────────────────────────────

    def _advice_for(
        self,
        level: TriageLevel,
        rule_finding: RuleFinding,
        llm_advice: str | None,
        message: str,
    ) -> str | None:
        if level is TriageLevel.ROUTINE:
            return None
        arabic = bool(re.search(r"[\u0600-\u06ff]", message))
        # Self-harm gets dedicated, supportive crisis guidance regardless of source.
        if "self_harm" in rule_finding.categories:
            return _SELF_HARM_ADVICE_AR if arabic else _SELF_HARM_ADVICE
        # Escalation wording is deterministic: a model may raise the triage level,
        # but it cannot weaken the action the patient is told to take.
        _ = llm_advice
        return (_DEFAULT_ADVICE_AR if arabic else _DEFAULT_ADVICE)[level]

    # ── Prompt construction ──────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        message: str,
        state: ConsultationState,
        rule_finding: RuleFinding,
    ) -> str:
        p = state.patient
        meds = ", ".join(m.name for m in p.active_medications) or "none"
        conditions = ", ".join(p.chronic_conditions) or "none"
        age = str(p.age) if p.age is not None else "unknown"
        flagged = ", ".join(rule_finding.categories) or "none"
        recent = self._recent_transcript(state)

        return (
            f'Patient message:\n"{message}"\n\n'
            f"Patient context:\n"
            f"  Age: {age}\n"
            f"  Chronic conditions: {conditions}\n"
            f"  Current medications: {meds}\n\n"
            f"Recent conversation (older to newer):\n{recent or 'none'}\n\n"
            f"Keyword pre-screen flagged categories: {flagged}\n\n"
            f"Assess the urgency of the patient message and respond with the JSON object."
        )

    def _recent_transcript(self, state: ConsultationState, limit: int = 6) -> str:
        messages = state.session.transcript[-limit:]
        return "\n".join(f"{m.role.value}: {m.content}" for m in messages)

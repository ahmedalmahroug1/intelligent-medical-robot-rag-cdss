"""
app/safety/rules.py
-------------------
The deterministic red-flag layer: pattern-based detection of emergency and
urgent presentations. This is the safety NET that sits under the LLM assessment.

Why a deterministic layer at all (and why it matters for the defense):
    Emergency detection is the single most safety-critical decision in the
    system. Relying on an LLM alone for it is fragile — a model can miss a sign,
    be talked out of a concern, or be unavailable. This layer runs in-process,
    instantly, offline, and cannot be reasoned away. The SafetyGuard combines it
    with the LLM assessment and takes the MORE CAUTIOUS of the two, so this layer
    acts as a floor the LLM can raise but never lower.

Design stance — high recall, deliberately:
    These patterns are tuned to catch danger, accepting some false positives. The
    cost of a false positive is advising someone to seek care they may not need;
    the cost of a false negative can be a missed emergency. For a safety net that
    asymmetry is the right call. The LLM layer adds nuance (negation, context),
    but because the guard takes the cautious maximum, a deterministic hit is not
    suppressed by the LLM. A narrow deterministic context filter suppresses only
    explicit local negations ("no chest pain", "لا يوجد ألم في الصدر"); ambiguous
    wording remains flagged.

Limitations that must be communicated clearly to users:
    - The local negation filter handles short explicit phrases, not arbitrary
      discourse. Complex temporal, hypothetical, or contradictory accounts still
      depend on the LLM assessment and should be reviewed conservatively.
    - This rule set is an engineering starting point. A real deployment must have
      it reviewed and expanded by a qualified clinician.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.enums import TriageLevel


# Severity ordering for TriageLevel. Kept here (in the safety layer) rather than
# on the enum to avoid changing the already-tested core models; it is a safety
# concern that this module and the guard own.
_SEVERITY: dict[TriageLevel, int] = {
    TriageLevel.ROUTINE: 0,
    TriageLevel.URGENT: 1,
    TriageLevel.EMERGENCY: 2,
}


def severity_rank(level: TriageLevel) -> int:
    """Numeric severity, higher = more urgent."""
    return _SEVERITY[level]


def max_triage(a: TriageLevel, b: TriageLevel) -> TriageLevel:
    """The more severe of two triage levels (the cautious choice)."""
    return a if severity_rank(a) >= severity_rank(b) else b


@dataclass(frozen=True)
class RedFlagRule:
    pattern: re.Pattern[str]
    level: TriageLevel
    reason: str
    category: str
    allow_direct_negation: bool = True


@dataclass(frozen=True)
class RuleFinding:
    """Outcome of the deterministic check for one message."""
    level: TriageLevel
    reasons: tuple[str, ...]
    categories: tuple[str, ...]


def _c(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_NEGATION_BEFORE = re.compile(
    r"(?:\b(?:no|not|without|deny|denies|denied|never)\b(?:\s+\w+){0,5}\s*|"
    r"(?:لا\s+يوجد|لا|ليس|لست|بدون|لم|ما\s+عندي(?:ش)?|مافيش)"
    r"(?:\s+\S+){0,5}\s*)$",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r"[.!?؟،;:\n]|\b(?:but|however|although|لكن|بس)\b", re.IGNORECASE)


def _directly_negated(text: str, match: re.Match[str]) -> bool:
    """Suppress only a short, explicit negation immediately before one hit.

    The matched emergency phrase itself is never inspected as a negation prefix,
    so dangerous expressions such as "not breathing" and "don't want to live"
    remain positive red flags. A clause boundary resets the context, preventing a
    negation in an earlier sentence from suppressing a later symptom.
    """
    prefix = text[max(0, match.start() - 80): match.start()]
    parts = _CLAUSE_BREAK.split(prefix)
    local = parts[-1] if parts else prefix
    return bool(_NEGATION_BEFORE.search(local))


# The rule set. EMERGENCY = life-threatening, seek emergency care now.
# URGENT = should be assessed by a professional within hours.
# Ordering does not matter; the finding takes the maximum severity matched.
_RULES: list[RedFlagRule] = [
    # ── EMERGENCY ────────────────────────────────────────────────────────────
    RedFlagRule(
        _c(r"suicid|kill myself|want to die|end my life|(hurt|harm)\s+myself|"
           r"no reason to live|don'?t want to live|انتحار|أقتل نفسي|اقتل نفسي|"
           r"أريد أن أموت|اريد ان اموت|إيذاء نفسي|ايذاء نفسي"),
        TriageLevel.EMERGENCY, "Possible self-harm or suicidal ideation", "self_harm",
    ),
    RedFlagRule(
        _c(r"can'?t breathe|can\s?not breathe|cannot breathe|stopped breathing|"
           r"not breathing|choking|لا أستطيع التنفس|لا استطيع التنفس|"
           r"لا أقدر على التنفس|اختناق|أختنق|اختنق"),
        TriageLevel.EMERGENCY, "Severe breathing difficulty", "respiratory",
    ),
    RedFlagRule(
        _c(r"crushing chest|chest pain.*(radiat|arm|jaw)|"
           r"pressure in (my |the )?chest|tightness in (my |the )?chest|"
           r"ألم (شديد )?في الصدر.*(الذراع|الفك)|ضغط في الصدر|ضيق في الصدر"),
        TriageLevel.EMERGENCY, "Possible cardiac event", "cardiac",
    ),
    RedFlagRule(
        _c(r"face drooping|slurred speech|sudden numbness|weakness on one side|"
           r"sudden confusion|can'?t move (my )?(arm|leg|face)|تدلي الوجه|"
           r"ثقل الكلام|خدر مفاجئ|ضعف مفاجئ.*(جهة|جانب)|لا أستطيع تحريك.*(ذراع|رجل|وجهي)"),
        TriageLevel.EMERGENCY, "Possible stroke (FAST signs)", "neuro",
    ),
    RedFlagRule(
        _c(r"worst headache|thunderclap headache|sudden severe headache|"
           r"أسوأ صداع|اسوأ صداع|صداع مفاجئ شديد"),
        TriageLevel.EMERGENCY, "Possible intracranial emergency", "neuro",
    ),
    RedFlagRule(
        _c(r"unconscious|unresponsive|passed out|fainted|seizure|convulsion|"
           r"فاقد الوعي|فقد الوعي|إغماء|اغماء|تشنج|نوبة صرع"),
        TriageLevel.EMERGENCY, "Loss of consciousness or seizure", "consciousness",
    ),
    RedFlagRule(
        _c(r"severe bleeding|won'?t stop bleeding|coughing up blood|"
           r"vomiting blood|blood in (my )?vomit|نزيف شديد|النزيف لا يتوقف|"
           r"قيء دم|أتقيأ دماً|اتقيأ دما|سعال دم"),
        TriageLevel.EMERGENCY, "Severe bleeding", "hemorrhage",
    ),
    RedFlagRule(
        _c(r"throat (is )?closing|tongue swelling|anaphylax|انغلاق الحلق|"
           r"حلقي ينغلق|تورم اللسان|حساسية مفرطة"),
        TriageLevel.EMERGENCY, "Possible anaphylaxis", "allergy",
    ),
    RedFlagRule(
        _c(r"overdose|took too many (?:pills|tablets)|swallowed (?:a lot of|many) "
           r"(?:pills|tablets)|poison(?:ed|ing)|جرعة زائدة|تناولت حبوب كثيرة|"
           r"ابتلعت حبوب|تسمم"),
        TriageLevel.EMERGENCY, "Possible overdose or poisoning", "toxicology",
    ),
    RedFlagRule(
        _c(r"sudden (?:loss of vision|vision loss|blindness)|can'?t see suddenly|"
           r"فقدان مفاجئ (?:للبصر|للنظر)|فقدت بصري فجأة|لا أرى فجأة"),
        TriageLevel.EMERGENCY, "Sudden loss of vision", "vision",
    ),
    RedFlagRule(
        _c(r"(?:pregnan\w*|postpartum|after giving birth).{0,45}(?:heavy|severe) bleeding|"
           r"(?:heavy|severe) bleeding.{0,45}(?:pregnan\w*|postpartum|after giving birth)|"
           r"(?:حامل|بعد الولادة).{0,45}نزيف شديد|نزيف شديد.{0,45}(?:حامل|بعد الولادة)"),
        TriageLevel.EMERGENCY, "Severe pregnancy or postpartum bleeding", "obstetric",
    ),
    RedFlagRule(
        _c(r"(?:fever|high temperature).{0,45}(?:stiff neck|confusion|purple rash)|"
           r"(?:stiff neck|purple rash).{0,45}(?:fever|high temperature)|"
           r"(?:حمى|حرارة).{0,45}(?:تيبس الرقبة|ارتباك|طفح بنفسجي)|"
           r"(?:تيبس الرقبة|طفح بنفسجي).{0,45}(?:حمى|حرارة)"),
        TriageLevel.EMERGENCY, "Possible severe infection or meningitis", "infection",
    ),
    # ── URGENT ───────────────────────────────────────────────────────────────
    RedFlagRule(
        _c(r"chest pain|ألم الصدر|ألم في الصدر|الم في الصدر|وجع الصدر"),
        TriageLevel.URGENT, "Chest pain reported", "cardiac",
    ),
    RedFlagRule(
        _c(r"shortness of breath|short of breath|difficulty breathing|"
           r"trouble breathing|ضيق التنفس|صعوبة التنفس|صعوبة في التنفس"),
        TriageLevel.URGENT, "Breathing difficulty", "respiratory",
    ),
    RedFlagRule(
        _c(r"severe pain|excruciating|worst pain of my life|ألم شديد|الم شديد|أشد ألم"),
        TriageLevel.URGENT, "Severe pain", "pain",
    ),
    RedFlagRule(
        _c(r"high fever|very high temperature|حمى شديدة|حرارة عالية|حرارتي مرتفعة جداً"),
        TriageLevel.URGENT, "High fever", "infection",
    ),
    RedFlagRule(
        _c(r"persistent vomiting|can'?t keep (?:food|fluids|water) down|"
           r"not urinating|very little urine|قيء مستمر|لا أستطيع الاحتفاظ (?:بالطعام|بالسوائل)|"
           r"انقطاع البول|بول قليل جداً"),
        TriageLevel.URGENT, "Possible dehydration or persistent vomiting", "dehydration",
    ),
    RedFlagRule(
        _c(r"black (?:tarry )?stool|blood in (?:my )?(?:stool|urine)|"
           r"براز أسود|دم في (?:البراز|البول)"),
        TriageLevel.URGENT, "Possible internal bleeding", "hemorrhage",
    ),
    RedFlagRule(
        _c(r"pregnan\w*.{0,35}(?:bleeding|severe abdominal pain)|"
           r"(?:bleeding|severe abdominal pain).{0,35}pregnan\w*|"
           r"حامل.{0,35}(?:نزيف|ألم شديد في البطن)|"
           r"(?:نزيف|ألم شديد في البطن).{0,35}حامل"),
        TriageLevel.URGENT, "Pregnancy-related warning symptom", "obstetric",
    ),
    RedFlagRule(
        _c(r"new confusion|suddenly confused|acute confusion|ارتباك جديد|تشوش مفاجئ"),
        TriageLevel.URGENT, "Acute confusion", "neuro",
    ),
]


def evaluate_rules(
    text: str,
    *,
    age: int | None = None,
    chronic_conditions: tuple[str, ...] | list[str] = (),
) -> RuleFinding:
    """
    Run every rule against the message and return the combined finding.

    The finding's level is the maximum severity of all matched rules; reasons and
    categories collect every match (deduplicated, order preserved). A message with
    no matches returns a ROUTINE finding with empty reasons.
    """
    level = TriageLevel.ROUTINE
    reasons: list[str] = []
    categories: list[str] = []

    for rule in _RULES:
        match = rule.pattern.search(text)
        if match and rule.allow_direct_negation and _directly_negated(text, match):
            continue
        if match:
            level = max_triage(level, rule.level)
            reasons.append(rule.reason)
            categories.append(rule.category)

    # A fever in an infant recorded as age 0 years deserves prompt assessment
    # even when the message does not use the words "high fever". This is a narrow
    # vulnerability floor; broader risk nuance stays with the LLM layer.
    if age == 0 and re.search(r"\bfever\b|حمى|حرارة", text, re.IGNORECASE):
        level = max_triage(level, TriageLevel.URGENT)
        reasons.append("Fever reported in an infant")
        categories.append("vulnerable_age")

    # Consume the context parameter intentionally: the rule API is ready for
    # clinician-reviewed condition-specific floors without changing callers.
    _ = chronic_conditions

    return RuleFinding(
        level=level,
        reasons=tuple(dict.fromkeys(reasons)),       # dedupe, preserve order
        categories=tuple(dict.fromkeys(categories)),
    )

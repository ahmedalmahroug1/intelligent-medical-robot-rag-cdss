"""
app/concierge/prompts.py
------------------------
The concierge's persona, prompts, bilingual templates, and a deterministic red-flag
pre-check. The persona is strict by design: the concierge is a FRONT DESK, not a doctor —
it never diagnoses, advises, or reassures about an outcome. All clinical content comes from
the guard-protected CDSS and is only relayed/translated by the concierge.

Greeting / identity / failover / escalation lines are DETERMINISTIC templates (so safety-
critical wording is never hallucinated); the LLM is used only for (a) the structured intake
extraction and (b) faithful translation of the CDSS reply.
"""
from __future__ import annotations

# Shared instructions for the patient-facing Concierge. It hard-forbids clinical content.
PERSONA = (
    "You are the patient's companion and interviewer — the only person the patient sees. "
    "You speak naturally and you REMEMBER the whole conversation. You may be warm, lightly "
    "humorous, and human. You NEVER invent a diagnosis, name a disease, or tell the patient what "
    "is wrong with them — a medical system does the clinical reasoning silently behind you. You "
    "collect the information that matters and you may ask your own questions. You speak Arabic and "
    "English and reply in the patient's language."
)

EXTRACT_SYS = (
    PERSONA + "\n\n"
    "Task: from the patient's message (Arabic or English), produce a STRUCTURED intake record. "
    "Translate all clinical content to English using common medical terms. Detect the message "
    "language. Extract ONLY what the patient actually stated — do not infer a diagnosis. "
    "Set is_clinical=false when the message has NO new clinical information — e.g. a greeting, "
    "thanks, a language preference, asking who you are, frustration, or small-talk; set "
    "is_clinical=true when it reports symptoms, history, medications, or allergies. "
    "Return ONLY JSON with these keys: original_text (verbatim), language ('ar' or 'en'), "
    "is_clinical (true|false), chief_complaint (short English phrase), symptoms (English list), "
    "duration (e.g. '2 days'), severity (e.g. '7/10' or 'mild'), location (e.g. 'left temple'), "
    "associated_symptoms (English list), medications (list), known_conditions (list), "
    "allergies (list), family_history (list), age (number or null), sex ('male'|'female'|'')."
)


def question_sys(language: str) -> str:
    """The concierge composes the NEXT interview question itself. It is given the reasoning
    engine's suggested question and the fields it still needs, and is free to rephrase, merge,
    add, or replace — it OWNS the conversation. Reception voice; never diagnoses or reassures."""
    lang_name = "Arabic" if language == "ar" else "English"
    return (
        PERSONA + "\n\n"
        f"Ask the patient your NEXT question, in {lang_name}, naturally and warmly — like a person, "
        "not a form. You have the conversation so far (do NOT repeat a question you already asked, "
        "and do NOT greet again), plus the medical system's suggested question and the details "
        "still needed; you may rephrase it, merge it, add your own, or ask for the most useful "
        "missing detail instead. Ask ONE clear question. No diagnosis. Output ONLY the question."
    )


def present_sys(language: str) -> str:
    """The concierge PRESENTS the CDSS's finished assessment (it does not invent it). It relays
    the system's working diagnosis as a possibility to confirm with a clinician — adding no new
    medical claim of its own."""
    lang_name = "Arabic" if language == "ar" else "English"
    return (
        PERSONA + "\n\n"
        f"The medical system has finished its assessment (given to you below). Present it to the "
        f"patient in {lang_name}, warmly and clearly, as a POSSIBLE explanation to confirm with a "
        "clinician — NOT a certainty. Mention the leading consideration and any suggested tests. "
        "Add NO medical claim beyond what you are given. End by advising them to confirm with a "
        "licensed clinician. Keep it brief."
    )


def present_fallback(language: str) -> str:
    if language == "ar":
        return ("أنهى النظام تقييمه. هذا احتمال أوّلي ينبغي تأكيده مع طبيب مختص. "
                "يُرجى مراجعة طبيب مرخّص.")
    return ("The system has finished its assessment. This is a preliminary possibility to confirm "
            "with a clinician. Please see a licensed clinician.")


def bridge_to_system(language: str) -> str:
    """Said when the shared sufficiency is met (Stage A end; Stage B replaces this with the
    presented working diagnosis/report)."""
    if language == "ar":
        return ("شكراً، لديّ معلومات كافية الآن. دعني أراجعها مع النظام الطبي وأعود إليك.")
    return ("Thank you, I have enough information now. Let me review it with the medical system "
            "and get back to you.")


# Deterministic fallback questions (used only if the LLM question call fails).
_FIELD_QUESTION: dict[str, tuple[str, str]] = {
    "duration": ("منذ متى تشعر بهذا؟", "How long have you been feeling this?"),
    "severity": ("ما شدّته على مقياس من 1 إلى 10؟", "How severe is it, on a scale of 1 to 10?"),
    "location": ("أين تشعر به بالضبط؟", "Where exactly do you feel it?"),
}


def fallback_question(language: str, missing: list[str], suggested: str) -> str:
    if missing and missing[0] in _FIELD_QUESTION:
        ar, en = _FIELD_QUESTION[missing[0]]
        return ar if language == "ar" else en
    if suggested:
        return suggested
    return "هل هناك المزيد تودّ إخباري به؟" if language == "ar" else \
           "Is there anything else you'd like to tell me?"


def smalltalk_sys(language: str) -> str:
    """System prompt for a brief, warm, NON-clinical reception reply (e.g. to a greeting, a
    language request, or 'are you a doctor?'). Reception only — never medical content."""
    lang_name = "Arabic" if language == "ar" else "English"
    return (
        PERSONA + "\n\n"
        f"This message is small-talk or a personal/meta question, not new clinical information. "
        f"Reply in {lang_name} naturally and briefly, like a real person, using the conversation "
        "and what you know about the patient. You have ALREADY greeted them — do NOT greet again or "
        "repeat their name every time. Answer what they actually asked (their name, who you are, a "
        "joke, a language switch). Do not give a diagnosis or medical advice. When it feels natural, "
        "gently bring the focus back to how they're feeling."
    )


def smalltalk_fallback(language: str) -> str:
    if language == "ar":
        return ("أنا هنا لمساعدتك واستقبال معلوماتك، والتقييم الطبي يقوم به النظام. "
                "متى أحببت، صف لي ما تشعر به ومنذ متى.")
    return ("I'm here to help and take your information; the medical assessment is handled by "
            "the system. Whenever you're ready, tell me what you're feeling and for how long.")


def translate_sys(language: str) -> str:
    """System prompt to faithfully translate the CDSS reply into the patient's language."""
    target = "Arabic" if language == "ar" else "English"
    return (
        f"Translate the medical assistant's message into clear, natural {target}. Keep it "
        "faithful: do not add, remove, or soften any medical content or instruction. Output "
        "ONLY the translation, nothing else."
    )


# ── deterministic bilingual templates ────────────────────────────────────────
def greeting() -> str:
    return ("Hello, I'm the clinic assistant. May I have your name?\n"
            "مرحباً، أنا مساعد العيادة. ما اسمك من فضلك؟")


IDENTITY_SYS = (
    PERSONA + "\n\n"
    "The patient may or may not have given their name. If the message contains the patient's "
    "NAME, extract it and set has_name=true. If it does NOT contain a name — a greeting, thanks, a "
    "request like 'speak in english', a question, or small-talk — set has_name=false and name=''. "
    "Detect the message language. Return ONLY JSON: {\"name\": str, \"has_name\": true|false, "
    "\"language\": \"ar\"|\"en\"}."
)


def ask_name(language: str) -> str:
    if language == "ar":
        return "تشرّفت بك. ما اسمك من فضلك؟"
    return "Nice to meet you. May I have your name, please?"


def ask_language(name: str = "") -> str:
    """Bilingual, deterministic: asked ONCE, on the very first contact with a patient whose
    record has no saved language. The answer (or the script of their first message) locks the
    conversation language until they explicitly ask to switch."""
    who_ar = f" يا {name}" if name else ""
    who_en = f", {name}" if name else ""
    return (f"أهلاً{who_ar}! بأي لغة تفضّل أن نتابع — العربية أم الإنجليزية؟\n"
            f"Welcome{who_en}! Which language would you prefer — Arabic or English?")


def opening(name: str, language: str, *, returning: bool, last_topic: str = "") -> str:
    """The channel opening for a patient whose language is already known (saved on their
    record) — one template shared by the website and the robot, in THEIR language."""
    if language == "ar":
        if returning:
            extra = f" آخر مرّة تحدّثنا عن {last_topic}." if last_topic else ""
            return f"أهلاً بعودتك يا {name}.{extra} كيف تشعر اليوم؟"
        return f"تشرّفت بك يا {name}. صف لي ما تشعر به اليوم."
    if returning:
        extra = f" Last time we talked about {last_topic}." if last_topic else ""
        return f"Welcome back, {name}.{extra} How are you feeling today?"
    return f"Nice to meet you, {name}. Please describe what you're feeling today."


def greet_known(name: str, language: str) -> str:
    if language == "ar":
        return f"أهلاً {name}، سعيد بعودتك. صف لي ما تشعر به اليوم."
    return f"Welcome back, {name}. Please describe what you're feeling today."


def greet_unknown(name: str, language: str) -> str:
    if language == "ar":
        return (f"تشرّفت بك يا {name}. سأُنشئ لك ملفاً جديداً. "
                "صف لي ما تشعر به ومنذ متى.")
    return (f"Nice to meet you, {name}. I'll create a new profile for you. "
            "Please describe what you're feeling and for how long.")


def failover(language: str) -> str:
    if language == "ar":
        return ("أعتذر، أواجه صعوبة في الوصول للنظام الطبي الآن. سجّلتُ ما أخبرتني به، "
                "ولنحاول مجدداً بعد لحظات. إن كنت قلقاً، فلا تتردد في مراجعة طبيب.")
    return ("I'm sorry, I'm having trouble reaching the medical system right now. I've noted "
            "what you told me; let's try again in a moment. If you're worried, please see a "
            "clinician.")


def escalation(language: str) -> str:
    if language == "ar":
        return ("قد تكون هذه حالة طارئة. يُرجى التوجّه إلى الطوارئ أو الاتصال برقم الطوارئ "
                "المحلي فوراً.")
    return ("This may be an emergency. Please seek emergency care or call your local emergency "
            "number now.")


# ── deterministic red-flag pre-check (defense-in-depth, before the CDSS guard) ─
# Conservative bilingual surface terms. The CDSS SafetyGuard is the real check; this only
# catches an emergency phrased in Arabic in case translation later softens it.
RED_FLAG_TERMS: tuple[str, ...] = (
    # English
    "chest pain", "can't breathe", "cant breathe", "difficulty breathing",
    "shortness of breath", "severe bleeding", "unconscious", "passed out", "stroke",
    "suicid", "kill myself", "choking", "anaphylaxis", "blue lips",
    # Arabic
    "ألم في الصدر", "ألم صدر", "وجع صدر", "ما اقدر اتنفس", "لا أستطيع التنفس",
    "ضيق تنفس", "ضيق في التنفس", "صعوبة تنفس", "صعوبة في التنفس", "نزيف شديد",
    "فقدان وعي", "فقدت الوعي", "اختناق", "انتحار", "سكتة",
)


def has_red_flag(text: str) -> bool:
    low = text.lower()
    return any(term in low for term in RED_FLAG_TERMS)

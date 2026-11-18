"""
app/concierge/identity.py
-------------------------
Stage-0 patient identification by typed name. `PatientRecord` has no name field (and we do
not modify the CDSS state), so the concierge keeps its OWN lightweight name -> patient_id
registry, persisted as JSON. A known name resolves to an existing patient id (history is then
preserved by `ConsultationService.start(patient_id=...)`); an unknown name is registered to a
new id. This is the seam that the face-recognition stage (Stage 1) will replace/augment.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

_DEFAULT_PATH = "data/concierge_identities.json"

_NAME_LEADINS = (
    "my name is", "i am", "i'm", "this is", "name is", "call me",
    "اسمي", "أنا", "انا", "اسمي هو", "يسمونني",
)


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def parse_name(text: str) -> str:
    """Pull a plausible name out of a free-text message ('my name is Patient' -> 'Patient')."""
    t = text.strip()
    low = t.lower()
    for lead in _NAME_LEADINS:
        if low.startswith(lead):
            t = t[len(lead):].strip(" :،,.")
            break
    # Keep it short: a name is a few words, not a sentence.
    return " ".join(t.split()[:3]).strip(" :،,.?!")


def detect_language(text: str) -> str:
    """'ar' if the text contains Arabic letters, else 'en'."""
    return "ar" if re.search(r"[؀-ۿ]", text) else "en"


# ── explicit language choice / switch (DETERMINISTIC on purpose) ─────────────
# The conversation language is LOCKED after it is chosen; the whole point of the lock is that
# the model flip-flops languages mid-dialogue — so the UNLOCK must never depend on the model.
# Two shapes are recognised: a whole-message choice (the answer to "which language?") and a
# narrow set of explicit in-sentence switch commands.

_CHOICE_AR = {
    "عربي", "عربى", "عربية", "العربية", "العربيه", "بالعربي", "بالعربية", "arabic", "ar",
}
_CHOICE_EN = {
    "english", "en", "انجليزي", "إنجليزي", "انكليزي", "انجليزية", "الانجليزية", "الإنجليزية",
    "الانجليزي", "الإنجليزي", "بالانجليزي", "بالانجليزية", "بالإنجليزية", "بالإنجليزي",
}
_SWITCH_AR = (
    "تكلم عربي", "تكلمي عربي", "احكي عربي", "حكي عربي", "كلمني عربي", "تحدث بالعربية",
    "تحدث بالعربي", "بالعربية من فضلك", "غير اللغة الى العربية", "غير اللغة إلى العربية",
    "speak arabic", "switch to arabic", "in arabic", "use arabic", "arabic please",
)
_SWITCH_EN = (
    "تكلم انجليزي", "تكلم إنجليزي", "كلمني انجليزي", "كلمني إنجليزي", "تحدث بالانجليزية",
    "تحدث بالإنجليزية", "بالانجليزية من فضلك", "بالإنجليزية من فضلك",
    "غير اللغة الى الانجليزية", "غير اللغة إلى الإنجليزية",
    "speak english", "switch to english", "in english", "use english", "english please",
    "talk english",
)


def explicit_language_switch(text: str) -> str | None:
    """'ar'/'en' when the message EXPLICITLY chooses or asks to switch the conversation
    language; None otherwise (including for ordinary messages in either script)."""
    low = re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" ?!.,؟،")
    if not low:
        return None
    if low in _CHOICE_AR:
        return "ar"
    if low in _CHOICE_EN:
        return "en"
    for phrase in _SWITCH_AR:
        if phrase in low:
            return "ar"
    for phrase in _SWITCH_EN:
        if phrase in low:
            return "en"
    return None


class IdentityRegistry:
    """A file-backed name -> patient_id map. Minimal on purpose."""

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._map: dict[str, str] = {}
        if self._path.exists():
            try:
                self._map = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt registry starts empty, never crashes
                self._map = {}

    def lookup(self, name: str) -> str | None:
        """Return the patient_id for a known name, or None."""
        return self._map.get(_norm(name))

    def register(self, name: str, patient_id: str | None = None) -> str:
        """Register a (new) name, returning its patient_id (created if not supplied)."""
        key = _norm(name)
        if key in self._map:
            return self._map[key]
        pid = patient_id or uuid.uuid4().hex
        self._map[key] = pid
        self._save()
        return pid

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._map, ensure_ascii=False, indent=1),
                              encoding="utf-8")


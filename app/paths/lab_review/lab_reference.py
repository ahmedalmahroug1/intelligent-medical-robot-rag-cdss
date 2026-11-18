"""
app/paths/lab_review/lab_reference.py
-------------------------------------
The DETERMINISTIC clinical knowledge behind the LAB_REVIEW path — no LLM. This is
deliberately a small, reviewable, hand-curated reference (the same stance as the
TreatmentPath safety overlay: never trust the model alone for a clinical judgement).

Three tables + four pure functions:
  * LAB_ALIASES         — the many ways a lab is written  -> one canonical name.
  * REFERENCE_RANGES    — canonical lab -> (low, high) adult bounds, so a raw value
                          becomes "low" / "normal" / "high". Sex-agnostic for v1.
  * DISEASE_LAB_PATTERNS- condition -> the lab pattern that SUPPORTS it
                          ({lab: "low"|"high"|"normal"}).
  * parse_labs / classify_value / interpret / recommend_labs.

Adding a condition is a single row in DISEASE_LAB_PATTERNS (plus any new lab in
REFERENCE_RANGES + LAB_ALIASES). A pattern only ever applies to its OWN condition, so
expanding the table cannot create spurious matches for other conditions.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

LOW, NORMAL, HIGH = "low", "normal", "high"

# ── canonical lab names & aliases ────────────────────────────────────────────
LAB_ALIASES: dict[str, str] = {
    # haematology
    "hb": "Hb", "hgb": "Hb", "hemoglobin": "Hb", "haemoglobin": "Hb",
    "mcv": "MCV", "mean corpuscular volume": "MCV",
    "ferritin": "Ferritin",
    "iron": "Serum Iron", "serum iron": "Serum Iron",
    "tibc": "TIBC", "total iron binding capacity": "TIBC",
    "transferrin saturation": "Transferrin saturation", "tsat": "Transferrin saturation",
    "rdw": "RDW",
    "wbc": "WBC", "white blood cells": "WBC", "white blood cell count": "WBC",
    "leukocytes": "WBC", "white cell count": "WBC",
    "platelets": "Platelets", "plt": "Platelets", "platelet count": "Platelets",
    "reticulocyte count": "Reticulocyte count", "reticulocytes": "Reticulocyte count",
    "retic": "Reticulocyte count",
    "vitamin b12": "Vitamin B12", "b12": "Vitamin B12", "cobalamin": "Vitamin B12",
    "folate": "Folate", "folic acid": "Folate",
    "ldh": "LDH", "lactate dehydrogenase": "LDH",
    "haptoglobin": "Haptoglobin",
    "total bilirubin": "Total bilirubin", "bilirubin": "Total bilirubin", "tbil": "Total bilirubin",
    # endocrine / metabolic
    "hba1c": "HbA1c", "a1c": "HbA1c", "glycated hemoglobin": "HbA1c",
    "fasting glucose": "Fasting glucose", "fbs": "Fasting glucose",
    "fpg": "Fasting glucose", "glucose": "Fasting glucose",
    "tsh": "TSH",
    "t4": "T4", "free t4": "T4", "ft4": "T4",
    "t3": "T3", "free t3": "T3", "ft3": "T3",
    "uric acid": "Uric acid", "urate": "Uric acid",
    "ldl": "LDL", "ldl cholesterol": "LDL",
    "total cholesterol": "Total cholesterol", "cholesterol": "Total cholesterol",
    # renal / electrolytes
    "creatinine": "Creatinine", "cr": "Creatinine",
    "bun": "BUN", "urea": "BUN", "blood urea nitrogen": "BUN",
    "egfr": "eGFR", "gfr": "eGFR",
    "sodium": "Sodium", "na": "Sodium",
    "potassium": "Potassium", "k": "Potassium",
    "calcium": "Calcium", "ca": "Calcium",
    # hepatic
    "alt": "ALT", "sgpt": "ALT", "alanine aminotransferase": "ALT",
    "ast": "AST", "sgot": "AST", "aspartate aminotransferase": "AST",
    "alp": "ALP", "alkaline phosphatase": "ALP",
    "ggt": "GGT", "gamma gt": "GGT",
    # cardiac
    "troponin": "Troponin", "trop": "Troponin", "troponin i": "Troponin",
    "troponin t": "Troponin", "hs-troponin": "Troponin",
    "bnp": "BNP", "nt-probnp": "BNP", "ntprobnp": "BNP", "pro-bnp": "BNP",
    # pancreatic / inflammatory
    "lipase": "Lipase",
    "amylase": "Amylase",
    "crp": "CRP", "c-reactive protein": "CRP",
    "esr": "ESR",
    "procalcitonin": "Procalcitonin", "pct": "Procalcitonin",
    "lactate": "Lactate", "lactic acid": "Lactate",
}

# ── reference ranges: value < low => "low", value > high => "high" ───────────
# Adult, sex-agnostic defaults for v1 (a documented simplification — real ranges
# vary by sex/lab/units; extend per deployment). For markers that are only
# clinically meaningful when ELEVATED (troponin, BNP, …) the low bound is 0.
REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "Hb": (12.0, 17.0),                  # g/dL
    "MCV": (80.0, 100.0),                # fL
    "Ferritin": (30.0, 300.0),           # ng/mL
    "Serum Iron": (60.0, 170.0),         # µg/dL
    "TIBC": (250.0, 450.0),              # µg/dL
    "Transferrin saturation": (20.0, 50.0),  # %
    "RDW": (11.5, 14.5),                 # %
    "WBC": (4.0, 11.0),                  # ×10^9/L
    "Platelets": (150.0, 450.0),         # ×10^9/L
    "Reticulocyte count": (0.5, 2.5),    # %
    "Vitamin B12": (200.0, 900.0),       # pg/mL
    "Folate": (3.0, 20.0),               # ng/mL
    "LDH": (140.0, 280.0),               # U/L
    "Haptoglobin": (30.0, 200.0),        # mg/dL
    "Total bilirubin": (0.1, 1.2),       # mg/dL
    "HbA1c": (4.0, 5.7),                 # %
    "Fasting glucose": (70.0, 100.0),    # mg/dL
    "TSH": (0.4, 4.0),                   # mIU/L
    "T4": (0.8, 1.8),                    # ng/dL (free T4)
    "T3": (2.3, 4.2),                    # pg/mL (free T3)
    "Uric acid": (3.5, 7.2),             # mg/dL
    "LDL": (0.0, 130.0),                 # mg/dL
    "Total cholesterol": (0.0, 200.0),   # mg/dL
    "Creatinine": (0.6, 1.3),            # mg/dL
    "BUN": (7.0, 20.0),                  # mg/dL
    "eGFR": (90.0, 120.0),               # mL/min/1.73m² (low = impaired)
    "Sodium": (135.0, 145.0),            # mmol/L
    "Potassium": (3.5, 5.0),             # mmol/L
    "Calcium": (8.5, 10.5),              # mg/dL
    "ALT": (7.0, 56.0),                  # U/L
    "AST": (10.0, 40.0),                 # U/L
    "ALP": (44.0, 147.0),                # U/L
    "GGT": (8.0, 61.0),                  # U/L
    "Troponin": (0.0, 0.04),             # ng/mL
    "BNP": (0.0, 100.0),                 # pg/mL
    "Lipase": (10.0, 140.0),             # U/L
    "Amylase": (30.0, 110.0),            # U/L
    "CRP": (0.0, 5.0),                   # mg/L
    "ESR": (0.0, 20.0),                  # mm/hr
    "Procalcitonin": (0.0, 0.5),         # ng/mL
    "Lactate": (0.5, 2.2),               # mmol/L
}

# ── condition -> the lab pattern that SUPPORTS it ────────────────────────────
DISEASE_LAB_PATTERNS: dict[str, dict[str, str]] = {
    # ── anaemias ──
    "Iron deficiency anemia": {
        "Hb": LOW, "MCV": LOW, "Ferritin": LOW, "Serum Iron": LOW,
        "TIBC": HIGH, "Transferrin saturation": LOW, "RDW": HIGH},
    "Thalassemia": {
        "Hb": LOW, "MCV": LOW, "Ferritin": NORMAL, "RDW": NORMAL},
    "Anemia of chronic disease": {
        "Hb": LOW, "MCV": NORMAL, "Ferritin": HIGH, "Serum Iron": LOW, "TIBC": LOW},
    "Vitamin B12 deficiency anemia": {
        "Hb": LOW, "MCV": HIGH, "Vitamin B12": LOW},
    "Folate deficiency anemia": {
        "Hb": LOW, "MCV": HIGH, "Folate": LOW},
    "Megaloblastic anemia": {"Hb": LOW, "MCV": HIGH},
    "Hemolytic anemia": {
        "Hb": LOW, "Reticulocyte count": HIGH, "Total bilirubin": HIGH,
        "LDH": HIGH, "Haptoglobin": LOW},
    "Aplastic anemia": {"Hb": LOW, "WBC": LOW, "Platelets": LOW},
    "Pancytopenia": {"Hb": LOW, "WBC": LOW, "Platelets": LOW},
    "Anemia": {"Hb": LOW},
    "Polycythemia vera": {"Hb": HIGH, "Platelets": HIGH},
    "Leukemia": {"WBC": HIGH, "Hb": LOW, "Platelets": LOW},
    # ── endocrine / metabolic ──
    "Diabetes mellitus": {"HbA1c": HIGH, "Fasting glucose": HIGH},
    "Diabetic ketoacidosis": {"Fasting glucose": HIGH, "HbA1c": HIGH},
    "Hyperthyroidism": {"TSH": LOW, "T4": HIGH, "T3": HIGH},
    "Hypothyroidism": {"TSH": HIGH, "T4": LOW},
    "Subclinical hypothyroidism": {"TSH": HIGH, "T4": NORMAL},
    "Gout": {"Uric acid": HIGH},
    "Hyperlipidemia": {"LDL": HIGH, "Total cholesterol": HIGH},
    "Hypercalcemia": {"Calcium": HIGH},
    # ── renal ──
    "Acute kidney injury": {"Creatinine": HIGH, "BUN": HIGH, "eGFR": LOW},
    "Chronic kidney disease": {"Creatinine": HIGH, "eGFR": LOW, "Hb": LOW},
    # ── hepatic ──
    "Hepatitis": {"ALT": HIGH, "AST": HIGH, "Total bilirubin": HIGH},
    "Cholestasis": {"ALP": HIGH, "GGT": HIGH, "Total bilirubin": HIGH},
    # ── cardiac ──
    "Possible NSTEMI / STEMI": {"Troponin": HIGH},
    "Myocardial infarction": {"Troponin": HIGH},
    "Heart failure": {"BNP": HIGH},
    # ── GI ──
    "Acute pancreatitis": {"Lipase": HIGH, "Amylase": HIGH},
    # ── infection / inflammation ──
    "Bacterial infection": {"WBC": HIGH, "CRP": HIGH, "ESR": HIGH},
    "Pneumonia": {"WBC": HIGH, "CRP": HIGH},
    "Sepsis": {"WBC": HIGH, "CRP": HIGH, "Procalcitonin": HIGH, "Lactate": HIGH},
    "Acute inflammation": {"ESR": HIGH, "CRP": HIGH},
    "Viral infection": {"WBC": LOW},
    # ── electrolytes ──
    "Hyperkalemia": {"Potassium": HIGH},
    "Hypokalemia": {"Potassium": LOW},
    "Hyponatremia": {"Sodium": LOW},
    "Hypernatremia": {"Sodium": HIGH},
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


_CANON_LOWER = {name.lower(): name for name in REFERENCE_RANGES}


def canonical_lab(name: str) -> str | None:
    """Map any spelling of a lab to its canonical name, or None if unknown."""
    s = re.sub(r"\s+", " ", str(name).strip().lower())
    if s in LAB_ALIASES:
        return LAB_ALIASES[s]
    if s in _CANON_LOWER:
        return _CANON_LOWER[s]
    return None


# A scan regex over every known spelling (longest first so "serum iron" beats "iron").
# A leading word-boundary stops a short symbol (K, Na, Ca, Cr) matching inside another
# word ("weak 5" must NOT read as Potassium=5).
_ALIAS_KEYS = sorted(set(LAB_ALIASES) | set(_CANON_LOWER), key=len, reverse=True)
_SCAN_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ALIAS_KEYS) + r")\s*[=:]?\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_labs(data) -> dict[str, float]:
    """Parse lab results from a dict OR free text into {canonical name: value}.

    Accepts: a dict ({"Hb": 8.2}), a JSON string, or lines like "Hb = 8.2",
    "MCV: 68", "ferritin 4". Unknown lab names and non-numeric values are dropped.
    """
    if isinstance(data, dict):
        items: list[tuple[str, object]] = list(data.items())
    else:
        text = str(data).strip()
        items = []
        try:
            obj = json.loads(text)
        except Exception:  # noqa: BLE001 - not JSON, fall through to regex
            obj = None
        if isinstance(obj, dict):
            items = list(obj.items())
        else:
            items = [(m.group(1), m.group(2)) for m in _SCAN_RE.finditer(text)]

    out: dict[str, float] = {}
    for raw_name, raw_value in items:
        canon = canonical_lab(str(raw_name))
        if canon is None:
            continue
        try:
            out[canon] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return out


def classify_value(lab: str, value: float) -> str | None:
    """A canonical lab value -> 'low' | 'normal' | 'high', or None if no range."""
    rng = REFERENCE_RANGES.get(lab)
    if rng is None:
        return None
    low, high = rng
    if value < low:
        return LOW
    if value > high:
        return HIGH
    return NORMAL


_PATTERN_KEYS = list(DISEASE_LAB_PATTERNS)


def match_condition(name: str) -> str | None:
    """Map a differential condition name to a DISEASE_LAB_PATTERNS key (prefer the
    most specific match), or None if the labs reference does not cover it."""
    s = _norm(name)
    if not s:
        return None
    for key in _PATTERN_KEYS:
        if _norm(key) == s:
            return key
    best: str | None = None
    for key in _PATTERN_KEYS:
        ks = _norm(key)
        if ks in s or s in ks:
            if best is None or len(ks) > len(_norm(best)):
                best = key
    return best


def interpret(labs: dict[str, float], conditions):
    """Score each condition against the entered labs, deterministically.

    Returns (classified, per_condition) where:
      * classified    = {canonical lab: 'low'|'normal'|'high'}
      * per_condition = {condition: {matched_kb, kb_condition, matches,
                         contradictions, support}} with support = matches - contradictions.
    A condition the labs reference does not cover is returned with matched_kb=False
    and is left untouched by the caller (open-world safe).
    """
    classified = {lab: classify_value(lab, v) for lab, v in labs.items()
                  if classify_value(lab, v) is not None}
    per_condition: dict[str, dict] = {}
    for cond in conditions:
        key = match_condition(cond)
        if key is None:
            per_condition[cond] = {"matched_kb": False}
            continue
        pattern = DISEASE_LAB_PATTERNS[key]
        matches, contradictions = [], []
        for lab, expected_dir in pattern.items():
            actual = classified.get(lab)
            if actual is None:
                continue
            (matches if actual == expected_dir else contradictions).append(lab)
        per_condition[cond] = {
            "matched_kb": True, "kb_condition": key,
            "matches": matches, "contradictions": contradictions,
            "support": len(matches) - len(contradictions),
        }
    return classified, per_condition


def recommend_labs(conditions) -> list[tuple[str, list[str]]]:
    """Rank the labs worth ordering for a differential: each covered condition's
    pattern labs, ordered by how many of the conditions they help discriminate."""
    used: dict[str, list[str]] = defaultdict(list)
    for cond in conditions:
        key = match_condition(cond)
        if key is None:
            continue
        for lab in DISEASE_LAB_PATTERNS[key]:
            used[lab].append(cond)
    return sorted(used.items(), key=lambda kv: (-len(kv[1]), kv[0]))

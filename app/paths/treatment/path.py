"""
app/paths/treatment/path.py
---------------------------
The TREATMENT path: management guidance for an ALREADY-KNOWN condition (how to
treat / manage / medicate), as opposed to working out what is wrong (DIAGNOSIS).
It is an ordinary `Path` peer — same contract, same router, no special status.

Shape (one structured LLM call, like the diagnosis cheap loop):
    1. Resolve the condition to manage — prefer the converged working diagnosis on
       the blackboard, else let the model read it from the patient's message.
    2. One `complete_typed(TreatmentResult)` call produces general management
       guidance, medication OPTIONS, and red-flags, given the patient's record.
    3. A DETERMINISTIC safety overlay cross-checks every proposed medication
       against the patient's allergies and active medications — never trusting the
       model to have caught a conflict — and forces a clinician-confirmation note.
    4. The patient-facing reply is composed deterministically (so the
       safety-critical wording cannot be hallucinated), always framed as general
       information that a licensed clinician must confirm — this path never
       prescribes.

Like every path, it runs BELOW the safety guard: emergencies are intercepted and
escalated before a turn ever reaches here. If the patient is actually describing
new symptoms (a diagnosis question), the model flags it and the path hands control
back to the router via a SWITCH to DIAGNOSIS.

Scope/limitation (documented honestly): the safety overlay is a substring
cross-check against the record, NOT a curated drug-interaction database. It catches
the obvious allergy/duplication cases and always defers to a clinician; it is not a
substitute for a real interaction checker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.enums import Route
from app.llm.base import LLMGateway, ModelTier
from app.paths.treatment.schemas import TreatmentResult
from app.routing.contract import Path, PathResult, Transition

if TYPE_CHECKING:  # type-only; attributes are read at runtime
    from app.core.state import ConsultationState

_CLARIFY = "Which condition would you like guidance on managing?"

_HANDOFF_TO_DIAGNOSIS = (
    "That sounds like something new to look into rather than managing a known "
    "condition — let's work through it properly."
)

_DISCLAIMER = (
    "This is general information, not a prescription. A licensed clinician should "
    "confirm any treatment before you start it."
)

_SYSTEM_PROMPT = """You are the treatment-guidance component of a medical \
decision-support assistant. The patient wants help MANAGING an already-known \
condition — not a new diagnosis. Given the condition and the patient's record \
(age, sex, chronic conditions, current medications, allergies), provide safe, \
general management guidance: practical self-care steps, the medication OPTIONS a \
clinician might consider (each with its purpose and the key caution), and clear \
red-flags for when to seek urgent care. You are NOT prescribing — a licensed \
clinician confirms any treatment. Do NOT recommend a medication the patient is \
allergic to. If the condition to manage is not clear, leave "condition" empty. If \
the patient is actually describing NEW symptoms to be diagnosed, set \
"new_symptoms" to true. Return ONLY a JSON object with exactly these keys:
{
  "condition": "the condition being managed, or empty if unclear",
  "summary": "a short management overview in plain language",
  "self_care": ["practical non-drug steps"],
  "medication_options": [{"name": "...", "purpose": "...", "caution": "..."}],
  "red_flags": ["when to seek urgent care"],
  "clinician_note": "one line a reviewing clinician should know",
  "new_symptoms": false
}"""


class TreatmentPath(Path):
    """Management guidance for a known condition, as a router-registered peer."""

    route = Route.TREATMENT

    def __init__(self, gateway: LLMGateway, *, model: ModelTier = ModelTier.REASONING) -> None:
        self._gateway = gateway
        self._model = model

    async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
        condition = self._known_condition(state)
        result = await self._gateway.complete_typed(
            TreatmentResult,
            system=_SYSTEM_PROMPT,
            user=self._build_user_prompt(turn, state, condition),
            model=self._model,
        )

        # The patient is really describing something to diagnose → hand back.
        if result.new_symptoms and not condition:
            return PathResult(
                reply=_HANDOFF_TO_DIAGNOSIS,
                transition=Transition.switch(Route.DIAGNOSIS),
            )

        if not result.condition and not condition:
            return PathResult(reply=_CLARIFY, transition=Transition.stay())

        flags = self._safety_flags(result, state)
        reply = self._compose_reply(result, condition, flags)
        return PathResult(reply=reply, transition=Transition.stay())

    # ── The deterministic safety overlay (does not trust the model) ────────────

    def _safety_flags(self, result: TreatmentResult, state: ConsultationState) -> list[str]:
        """Cross-check each proposed medication against the patient's allergies and
        active medications. Substring match in either direction catches the obvious
        conflicts; it is intentionally conservative, not a drug-interaction DB."""
        patient = state.patient
        allergens = [a.substance.strip().lower() for a in patient.allergies if a.substance.strip()]
        active = [m.name.strip().lower() for m in patient.active_medications if m.name.strip()]

        flags: list[str] = []
        for med in result.medication_options:
            name = med.name.strip().lower()
            if not name:
                continue
            for allergen in allergens:
                if allergen in name or name in allergen:
                    flags.append(
                        f"{med.name}: the record lists an allergy to {allergen} — do not take."
                    )
            for current in active:
                if current in name or name in current:
                    flags.append(
                        f"{med.name}: the patient already takes {current} — check for "
                        "duplication or interaction."
                    )
        return flags

    # ── Deterministic reply composition (safety wording never hallucinated) ────

    def _compose_reply(
        self, result: TreatmentResult, fallback_condition: str | None, flags: list[str]
    ) -> str:
        condition = result.condition or fallback_condition or "your condition"
        lines = [f"Here is some general guidance on managing {condition}."]
        if result.summary:
            lines.append(result.summary)
        if result.self_care:
            lines.append("Self-care steps: " + "; ".join(result.self_care) + ".")
        if result.medication_options:
            meds = "; ".join(
                f"{m.name} ({m.purpose})" if m.purpose else m.name
                for m in result.medication_options if m.name
            )
            if meds:
                lines.append("Medication options a clinician might consider: " + meds + ".")
        if flags:
            lines.append(
                "Important safety check based on your record — " + " ".join(flags)
                + " Please do not start these without confirming with a clinician first."
            )
        if result.red_flags:
            lines.append("Seek prompt medical care if: " + "; ".join(result.red_flags) + ".")
        lines.append(_DISCLAIMER)
        return " ".join(lines)

    # ── Prompt construction / context ──────────────────────────────────────────

    def _known_condition(self, state: ConsultationState) -> str | None:
        wd = state.visit.working_diagnosis
        if wd is not None and wd.condition:
            return wd.condition
        return None

    def _build_user_prompt(
        self, turn: str, state: ConsultationState, condition: str | None
    ) -> str:
        patient = state.patient
        age = str(patient.age) if patient.age is not None else "unknown"
        conditions = ", ".join(patient.chronic_conditions) or "none recorded"
        meds = ", ".join(m.name for m in patient.active_medications) or "none recorded"
        allergies = ", ".join(a.substance for a in patient.allergies) or "none recorded"
        known = f"Known condition to manage: {condition}\n\n" if condition else ""
        return (
            "Patient context:\n"
            f"  Age: {age}\n"
            f"  Sex: {patient.sex.value}\n"
            f"  Chronic conditions: {conditions}\n"
            f"  Current medications: {meds}\n"
            f"  Allergies: {allergies}\n\n"
            + known
            + f'Patient\'s message:\n"{turn}"\n\n'
            "Respond with the JSON object."
        )

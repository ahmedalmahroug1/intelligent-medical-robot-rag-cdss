"""
app/paths/lab_review/path.py
----------------------------
The LAB_REVIEW path: closes the diagnostic loop with objective data. It runs on the
existing `Route.TEST_REVIEW` (the classifier already routes "here are my lab results"
to it) and is an ordinary `Path` peer — same contract, same router, no special status.

Shape (mirrors TreatmentPath — deterministic core, deterministic reply, no LLM in the
clinical judgement):
    1. Parse lab values from the patient's message (deterministic; an optional single
       LLM call only as a FALLBACK extractor for messy free text — never to interpret).
    2. NO values yet -> recommend the labs that would discriminate the current
       differential (deterministic, via lab_reference) and ask the patient to report them.
    3. Values present -> store them on the visit, interpret them DETERMINISTICALLY against
       lab_reference, nudge each covered hypothesis's confidence up (supported) or down
       (contradicted), re-rank the differential, update the working diagnosis IN PLACE
       (no schema change), and explain — always framed as informational, never a
       prescription.

Conditions the labs reference does not cover are left untouched (open-world safe). This
path never prescribes, never escalates, never approves — the guard and approval gate are
unchanged and sit above it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.enums import Route
from app.core.state import Hypothesis, RecommendedTest
from app.llm.base import LLMError, LLMGateway, ModelTier
from app.paths.lab_review.lab_reference import (
    classify_value, interpret, parse_labs, recommend_labs,
)
from app.paths.lab_review.schemas import LabParse
from app.routing.contract import Path, PathResult, Transition

if TYPE_CHECKING:  # type-only; attributes read at runtime
    from app.core.state import ConsultationState

_CLAMP = (0.02, 0.98)
_DEFAULT_STEP = 0.08          # confidence nudge per net supporting lab

_DISCLAIMER = (
    "These results are interpreted automatically for information only — not a diagnosis "
    "or prescription. A licensed clinician must review them before any decision."
)

_NO_CONTEXT = (
    "Share your lab results (e.g. \"Hb 8.2, MCV 68, Ferritin 4\") and I'll interpret "
    "them, or tell me your symptoms first so we can work out what to check."
)

_PARSE_SYS = (
    "Extract laboratory results from the patient's message as numeric values. Return "
    "ONLY JSON: {\"values\": {\"<test name>\": <number>}}. Use the test's common name "
    "(e.g. Hb, MCV, Ferritin, HbA1c, TSH). Omit anything that is not a lab value."
)


class LabReviewPath(Path):
    """Recommend → parse → interpret → update, on Route.TEST_REVIEW. Informational only."""

    route = Route.TEST_REVIEW

    def __init__(
        self, gateway: LLMGateway, *, model: ModelTier = ModelTier.FAST,
        step: float = _DEFAULT_STEP,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._step = step

    async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
        labs = parse_labs(turn)
        if not labs:
            labs = await self._llm_fallback_parse(turn)
        if not labs:
            return self._recommend_or_clarify(state)
        return self._interpret_and_update(state, labs)

    # ── Step 1: recommend labs (deterministic) ────────────────────────────────
    def _recommend_or_clarify(self, state: ConsultationState) -> PathResult:
        conditions = self._candidate_conditions(state)
        recs = recommend_labs(conditions) if conditions else []
        if not recs:
            return PathResult(reply=_NO_CONTEXT, transition=Transition.stay())

        tests = [
            RecommendedTest(
                name=lab,
                rationale="Helps distinguish: " + ", ".join(dict.fromkeys(conds)) + ".",
                discriminates=list(dict.fromkeys(conds)),
            )
            for lab, conds in recs
        ]
        state.visit.recommended_tests = tests

        lab_list = ", ".join(t.name for t in tests)
        reply = (
            "Based on the current differential, these labs would help narrow it down: "
            f"{lab_list}. Run them and share the values (e.g. \"Hb 8.2, MCV 68\") and "
            "I'll interpret them. " + _DISCLAIMER
        )
        return PathResult(reply=reply, transition=Transition.stay())

    # ── Steps 2-4: store, interpret, update, explain (deterministic) ───────────
    def _interpret_and_update(
        self, state: ConsultationState, labs: dict[str, float]
    ) -> PathResult:
        # 2. store (reassign so validate_assignment / persistence sees the change)
        state.visit.lab_results = {**state.visit.lab_results, **labs}

        differential = list(state.visit.differential)
        wd = state.visit.working_diagnosis
        conditions = [h.condition for h in differential]
        if wd is not None and not any(self._same(h.condition, wd.condition) for h in differential):
            conditions.append(wd.condition)

        # 3. deterministic interpretation
        classified, per_condition = interpret(labs, conditions)

        # 4. nudge confidence for covered hypotheses; re-rank
        old_conf = {h.condition: float(h.confidence) for h in differential}
        updated: list[Hypothesis] = []
        for h in differential:
            info = per_condition.get(h.condition, {})
            if info.get("matched_kb"):
                new_conf = self._clamp(float(h.confidence) + self._step * self._support(info))
                updated.append(h.model_copy(update={
                    "confidence": round(new_conf, 4),
                    "supporting": list(dict.fromkeys([*h.supporting,
                        *(f"{lab} consistent" for lab in info["matches"])])),
                    "against": list(dict.fromkeys([*h.against,
                        *(f"{lab} not consistent" for lab in info["contradictions"])])),
                }))
            else:
                updated.append(h)
        updated.sort(key=lambda h: h.confidence, reverse=True)
        state.visit.differential = updated

        lead_changed = bool(updated) and bool(differential) and \
            not self._same(updated[0].condition, differential[0].condition)

        # update the working diagnosis IN PLACE (schema unchanged), if one exists
        if wd is not None:
            if updated:
                lead = updated[0]
                note = self._wd_note(per_condition, lead.condition)
                state.visit.working_diagnosis = wd.model_copy(update={
                    "condition": lead.condition,
                    "confidence": lead.confidence,
                    "differential": updated,
                    "reasoning": (wd.reasoning + " " + note).strip(),
                })
            else:
                # no differential to re-rank: adjust the working condition itself
                info = per_condition.get(wd.condition, {})
                if info.get("matched_kb"):
                    new_conf = self._clamp(float(wd.confidence) + self._step * self._support(info))
                    state.visit.working_diagnosis = wd.model_copy(update={
                        "confidence": round(new_conf, 4),
                        "reasoning": (wd.reasoning + " " + self._wd_note(per_condition, wd.condition)).strip(),
                    })

        reply = self._compose_reply(labs, classified, per_condition, updated, old_conf, lead_changed)
        return PathResult(reply=reply, transition=Transition.stay())

    # ── deterministic explanation ─────────────────────────────────────────────
    def _compose_reply(self, labs, classified, per_condition, updated, old_conf, lead_changed) -> str:
        readings = "; ".join(
            f"{lab} {self._fmt(labs[lab])} ({classified[lab]})"
            for lab in labs if lab in classified
        )
        lines = ["Here is what these labs suggest."]
        if readings:
            lines.append("Readings: " + readings + ".")

        for h in updated[:3]:
            info = per_condition.get(h.condition, {})
            if not info.get("matched_kb"):
                continue
            ms, cs = info["matches"], info["contradictions"]
            if not ms and not cs:
                continue
            if ms and not cs:
                verdict = "supported by " + ", ".join(ms)
            elif cs and not ms:
                verdict = "argued against by " + ", ".join(cs)
            else:
                verdict = "mixed (" + ", ".join(ms) + " support; " + ", ".join(cs) + " against)"
            before = old_conf.get(h.condition)
            change = f" — confidence {self._pct(before)} → {self._pct(h.confidence)}" if before is not None else ""
            lines.append(f"{h.condition}: {verdict}{change}.")

        if lead_changed and updated:
            lines.append(f"Most likely now: {updated[0].condition}.")
        lines.append(_DISCLAIMER)
        return " ".join(lines)

    # ── helpers ────────────────────────────────────────────────────────────────
    async def _llm_fallback_parse(self, turn: str) -> dict[str, float]:
        """Only reached when the deterministic parser found nothing. Re-validates the
        model's output through parse_labs, so the LLM never bypasses the reference."""
        try:
            result = await self._gateway.complete_typed(
                LabParse, system=_PARSE_SYS, user=turn, model=self._model
            )
        except LLMError:
            return {}
        return parse_labs(result.values)

    def _candidate_conditions(self, state: ConsultationState) -> list[str]:
        conditions = [h.condition for h in state.visit.differential]
        wd = state.visit.working_diagnosis
        if wd is not None and wd.condition and not conditions:
            conditions = [wd.condition]
        return conditions

    def _wd_note(self, per_condition: dict, condition: str) -> str:
        info = per_condition.get(condition, {})
        if not info.get("matched_kb"):
            return ""
        ms, cs = info.get("matches", []), info.get("contradictions", [])
        if ms and not cs:
            return f"Lab results support {condition} ({', '.join(ms)})."
        if cs:
            return f"Lab results are partly inconsistent with {condition} ({', '.join(cs)})."
        return ""

    @staticmethod
    def _support(info: dict) -> int:
        """Net lab support for a condition. Contradictions are weighted ×2: a lab that
        should be X but is Y is stronger evidence AGAINST than a (often non-specific)
        match is FOR — so a single discriminating contradiction lowers confidence."""
        return len(info["matches"]) - 2 * len(info["contradictions"])

    @staticmethod
    def _same(a: str, b: str) -> bool:
        return a.strip().casefold() == b.strip().casefold()

    @staticmethod
    def _clamp(p: float) -> float:
        return max(_CLAMP[0], min(_CLAMP[1], p))

    @staticmethod
    def _fmt(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else str(v)

    @staticmethod
    def _pct(p) -> str:
        return f"{round(float(p) * 100)}%"

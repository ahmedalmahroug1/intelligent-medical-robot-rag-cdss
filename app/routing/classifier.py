"""
app/routing/classifier.py
-------------------------
Turn CLASSIFICATION: deciding WHICH path should handle a turn when no path is
already active and no suspended path is waiting to resume. This is the router's
fallback when it cannot simply continue an in-progress path.

`TurnClassifier` is the interface; `LLMClassifier` is the production
implementation (one FAST-tier LLM call). Keeping classification behind an
interface is the same move the rest of the system makes: the router depends on
the abstraction, so a turn can be classified by a deterministic fake in tests
(no network, no key, no cost) while production uses the model. Without it the
router could not be tested offline.

Safety / UX stance — never guess wildly
    If the model fails, or returns something that is not one of the AVAILABLE
    routes, the classifier returns Route.UNKNOWN. The router turns UNKNOWN into a
    short clarifying question rather than dispatching to an arbitrary path.
    Mis-routing a patient is worse than asking them to say a little more. This is
    the routing analogue of the guard's "never fail open": when unsure, fall back
    to the safe behaviour (ask), not to a confident-but-wrong one.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from app.core.enums import Route
from app.llm.base import LLMError, LLMGateway, ModelTier

if TYPE_CHECKING:  # type-only; not imported at runtime
    from app.core.state import ConsultationState

logger = logging.getLogger(__name__)


class TurnClassifier(ABC):
    """Maps a patient turn to the Route that should handle it."""

    @abstractmethod
    async def classify(
        self,
        *,
        turn: str,
        state: ConsultationState,
        available: Iterable[Route],
    ) -> Route:
        """
        Return the best Route for `turn`, chosen from `available` (the routes the
        router actually has paths for). Return Route.UNKNOWN if none clearly fits
        or classification is not possible.
        """
        ...


# One-line descriptions telling the model what each route is for. Only the routes
# that are actually AVAILABLE are shown to the model, so it can never pick a path
# the router has no handler for. Kept here (a routing concern) rather than on the
# Route enum, to avoid coupling the core vocabulary to prompt wording.
_ROUTE_DESCRIPTIONS: dict[Route, str] = {
    Route.DIAGNOSIS: (
        "The patient is describing symptoms or a health problem and wants help "
        "understanding what might be causing it."
    ),
    Route.TREATMENT: (
        "The patient is asking how to treat, manage, or take medication for an "
        "already-known condition or diagnosis."
    ),
    Route.QA: (
        "The patient is asking a general medical question that is not about their "
        "own active diagnostic work-up."
    ),
    Route.FOLLOW_UP: (
        "The patient is following up on a previous visit, or reporting how they "
        "have been since an earlier consultation."
    ),
    Route.TEST_REVIEW: (
        "The patient is reporting, or asking about, the result of a lab test, "
        "scan, or other medical investigation."
    ),
}

_SYSTEM_PROMPT = """You are a router for a medical assistant. Your ONLY job is to \
decide which kind of conversation the patient's latest message belongs to, so it \
can be sent to the right handler. You are NOT answering the medical question.

You will be given the available conversation types (with descriptions) and the \
patient's message. Choose exactly one type — by its key — that best fits the \
message. If none clearly fits, or the message is too vague to tell, choose \
"unknown" so the patient can be asked to clarify.

Return ONLY a JSON object with exactly this shape:
{"route": "<one of the provided keys, or 'unknown'>"}"""


class LLMClassifier(TurnClassifier):
    """Classifies a turn with a single FAST-tier LLM call."""

    def __init__(self, gateway: LLMGateway, *, model: ModelTier = ModelTier.FAST) -> None:
        self._gateway = gateway
        self._model = model

    async def classify(
        self,
        *,
        turn: str,
        state: ConsultationState,
        available: Iterable[Route],
    ) -> Route:
        options = [r for r in available if r is not Route.UNKNOWN]
        if not options:
            return Route.UNKNOWN

        try:
            data = await self._gateway.complete_structured(
                system=_SYSTEM_PROMPT,
                user=self._build_user_prompt(turn, state, options),
                model=self._model,
            )
        except LLMError as exc:
            logger.warning("Turn classification unavailable; will ask to clarify: %s", exc)
            return Route.UNKNOWN

        return self._parse(data, options)

    # ── Internals ────────────────────────────────────────────────────────────

    def _build_user_prompt(
        self, turn: str, state: ConsultationState, options: list[Route]
    ) -> str:
        type_lines = "\n".join(
            f'- "{r.value}": {_ROUTE_DESCRIPTIONS.get(r, r.value)}' for r in options
        )
        recent = self._recent_transcript(state)
        return (
            "Available conversation types:\n"
            f"{type_lines}\n\n"
            "Recent conversation (older to newer):\n"
            f"{recent or 'none'}\n\n"
            f'Patient\'s latest message:\n"{turn}"\n\n'
            "Respond with the JSON object naming the best-fitting type."
        )

    def _recent_transcript(self, state: ConsultationState, limit: int = 6) -> str:
        messages = state.session.transcript[-limit:]
        return "\n".join(f"{m.role.value}: {m.content}" for m in messages)

    def _parse(self, data: dict[str, Any], options: list[Route]) -> Route:
        raw = str(data.get("route", "")).strip().lower()
        by_value = {r.value: r for r in options}
        route = by_value.get(raw)
        if route is None:
            # Either the model said "unknown" (fine) or returned something we
            # cannot service (log it). Both degrade to a clarifying question.
            if raw and raw != Route.UNKNOWN.value:
                logger.warning("Classifier returned unusable route=%r; clarifying", raw)
            return Route.UNKNOWN
        return route

"""
app/llm/split_gateway.py
------------------------
A gateway that routes each call to a DIFFERENT backend depending on its ModelTier.

Why this exists: the diagnostic reasoning is a handful of calls per case, but the
auxiliary traffic (patient simulator, turn classifier, extraction) is the bulk of
the calls. On a metered/free API the auxiliary traffic exhausts the daily quota
long before the reasoning does. Splitting by tier lets the cheap, high-frequency
FAST work run on a free LOCAL model (Ollama, unlimited) while only the scarce
REASONING calls hit the paid/rate-limited API.

It is a pure delegator: it implements the same LLMGateway interface and forwards
each primitive to `fast` or `reasoning` based on the call's `model` tier. It adds
no policy of its own beyond that choice, so retries/logging/etc. stay in the
concrete gateways underneath.

A bonus for evaluation rigour: with the patient + routing held on ONE fixed local
model, comparing several API reasoning models varies only the thing under test.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from app.llm.base import LLMGateway, LLMMessage, ModelTier


class SplitTierGateway(LLMGateway):
    """Route FAST-tier calls to one gateway and REASONING-tier calls to another."""

    def __init__(
        self,
        *,
        fast: LLMGateway,
        reasoning: LLMGateway,
        default_tier: ModelTier = ModelTier.REASONING,
    ) -> None:
        self._fast = fast
        self._reasoning = reasoning
        self._default_tier = default_tier

    def _pick(self, model: ModelTier | None) -> LLMGateway:
        tier = model or self._default_tier
        return self._fast if tier == ModelTier.FAST else self._reasoning

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> dict[str, Any]:
        return await self._pick(model).complete_structured(
            system=system, user=user, history=history, model=model
        )

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> str:
        return await self._pick(model).complete_text(
            system=system, user=user, history=history, model=model
        )

    async def stream_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._pick(model).stream_text(
            system=system, user=user, history=history, model=model
        ):
            yield chunk

"""
app/llm/openai_gateway.py
-------------------------
The OpenAI-backed implementation of LLMGateway.

This is the ONLY file in the system that imports `openai`. Everything else
depends on the LLMGateway interface, so replacing the provider means adding a
sibling implementation here — no agent changes.

What this single choke point centralises (the reason agents must not call OpenAI
directly, as they did in V2):
    - model selection via ModelTier → concrete model name,
    - timeout and a retry loop (retries handled here, in one place, and logged),
    - uniform error wrapping into LLMError,
    - token-usage logging for cost visibility.

Note on retries: only the non-streaming calls retry. A failed stream cannot be
retried cleanly because partial output may already have been yielded to the
caller, so streaming makes a single attempt and surfaces failure as LLMError.

Note on prompt caching: OpenAI caches long, repeated prompt prefixes
automatically for supported models, so a large static system prompt benefits
without any special handling here.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Sequence
from collections.abc import Mapping

import re

from openai import APITimeoutError, AsyncOpenAI, OpenAIError, RateLimitError

from app.config import settings
from app.llm.base import LLMError, LLMGateway, LLMMessage, ModelTier

logger = logging.getLogger(__name__)

# Providers phrase the wait differently in the 429 body: Groq "try again in 2.78s",
# Google "Please retry in 46.8s" / retryDelay '46s'. Match all of them.
_RETRY_AFTER_RE = re.compile(r"(?:try again in|retry in|retryDelay['\":\s]+)\s*([\d.]+)s?")


def _retry_after_seconds(exc: Exception) -> float | None:
    """Seconds to wait before retrying a rate-limited call, from the provider's own
    signal: the `retry-after` header first, else the "try again / retry in Xs" hint
    OpenAI-compatible hosts (Groq, Google) put in the 429 body. None if not present."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
        except Exception:  # noqa: BLE001 - some error responses have no usable headers
            header = None
        if header:
            try:
                return float(header)
            except ValueError:
                pass
    m = _RETRY_AFTER_RE.search(str(exc))
    return float(m.group(1)) if m else None


class OpenAIGateway(LLMGateway):
    """LLMGateway backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_tier: ModelTier = ModelTier.REASONING,
        timeout: float | None = None,
        max_retries: int | None = None,
        temperature: float | None = None,
        backoff_s: float = 0.5,
        models: Mapping[ModelTier, str] | None = None,
    ) -> None:
        key = api_key if api_key is not None else settings.OPENAI_API_KEY
        if not key:
            raise ValueError(
                "No API key set. Pass api_key=... or set the provider env var "
                "(e.g. OPENAI_API_KEY / GROQ_API_KEY)."
            )
        # base_url lets this same gateway talk to any OpenAI-COMPATIBLE endpoint
        # (Groq, OpenRouter, Cerebras, Gemini's OpenAI shim, a local vLLM). The
        # OpenAI SDK is the wire protocol; only the host and key change. None keeps
        # the default OpenAI endpoint.
        self._base_url = base_url if base_url is not None else (settings.LLM_BASE_URL or None)

        self._timeout = settings.LLM_TIMEOUT_S if timeout is None else timeout
        self._max_retries = settings.LLM_MAX_RETRIES if max_retries is None else max_retries
        self._temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        self._backoff = backoff_s
        self._default_tier = default_tier
        # Tier → concrete model name, from settings by default. `models` overrides
        # specific tiers, which is how the evaluation harness compares models (e.g.
        # point REASONING at gpt-4o for one run and gpt-4o-mini for the next)
        # without touching env or settings.
        self._model_for: dict[ModelTier, str] = {
            ModelTier.FAST: settings.MODEL_FAST,
            ModelTier.REASONING: settings.MODEL_REASONING,
        }
        if models:
            self._model_for.update(models)

        # max_retries=0: we run (and log) retries ourselves so all retry policy
        # lives in one place rather than being split with the SDK's own retries.
        self._client = AsyncOpenAI(
            api_key=key, base_url=self._base_url, max_retries=0, timeout=self._timeout
        )

    def model_name_for(self, tier: ModelTier) -> str:
        """The concrete model name a tier maps to (useful for logging/tests)."""
        return self._model_for[tier]

    # ── Public API ───────────────────────────────────────────────────────────

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> dict[str, Any]:
        content = await self._complete(
            self._build(system, user, history), model, json_mode=True
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}") from exc

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> str:
        return await self._complete(
            self._build(system, user, history), model, json_mode=False
        )

    async def stream_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> AsyncIterator[str]:
        model_name = self._resolve(model)
        messages = self._build(system, user, history)
        try:
            stream = await self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self._temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                choices = chunk.choices
                if choices and choices[0].delta and choices[0].delta.content:
                    yield choices[0].delta.content
                # The final chunk carries usage when include_usage is set.
                if getattr(chunk, "usage", None):
                    self._log_usage(model_name, chunk.usage, attempt=1)
        except OpenAIError as exc:
            raise LLMError(f"LLM streaming failed: {exc}") from exc

    # ── Internals ────────────────────────────────────────────────────────────

    def _resolve(self, tier: ModelTier | None) -> str:
        return self._model_for[tier or self._default_tier]

    def _build(
        self,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None,
    ) -> list[dict[str, str]]:
        """Assemble the message list: system prompt, prior turns, current user."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if history:
            messages.extend({"role": m.role, "content": m.content} for m in history)
        messages.append({"role": "user", "content": user})
        return messages

    async def _complete(
        self,
        messages: list[dict[str, str]],
        tier: ModelTier | None,
        *,
        json_mode: bool,
    ) -> str:
        """Non-streaming call with a logged retry loop. Returns the text content."""
        model_name = self._resolve(tier)
        attempts = self._max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": self._temperature,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = await self._client.chat.completions.create(**kwargs)
                self._log_usage(model_name, getattr(resp, "usage", None), attempt)
                return resp.choices[0].message.content or ""
            except OpenAIError as exc:
                last_exc = exc
                # Retrying a request that already consumed the full timeout can
                # double multi-minute structured jobs and still fail for the same
                # reason. Connection resets/rate limits remain retryable, but a
                # timeout is final for this call and the caller can checkpoint or
                # rerun the background task explicitly.
                if isinstance(exc, APITimeoutError):
                    logger.warning("LLM timed out (model=%s); not retrying.", model_name)
                    break
                if attempt < attempts:
                    # On a rate limit, wait exactly as long as the provider asks
                    # (its per-minute window resets), rather than a too-short fixed
                    # backoff that just burns the retry budget. A small cushion is
                    # added; capped so a pathological hint can't hang the run.
                    wait = None
                    if isinstance(exc, RateLimitError):
                        wait = _retry_after_seconds(exc)
                    if wait is not None:
                        delay = min(wait + 0.5, 30.0)
                        logger.info("rate-limited (model=%s); waiting %.1fs (provider hint)",
                                    model_name, delay)
                    else:
                        delay = self._backoff * attempt  # linear backoff for other errors
                        logger.warning("LLM attempt %d/%d failed (model=%s): %s",
                                       attempt, attempts, model_name, exc)
                    await asyncio.sleep(delay)

        raise LLMError(
            f"LLM call failed after {attempts} attempt(s): {last_exc}"
        ) from last_exc

    def _log_usage(self, model_name: str, usage: Any, attempt: int) -> None:
        if usage is not None:
            logger.info(
                "LLM ok model=%s attempt=%d prompt=%s completion=%s total=%s",
                model_name, attempt,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
            )

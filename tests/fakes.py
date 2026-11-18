"""
tests/fakes.py
--------------
Test doubles reused across the suite. Kept dependency-free (only the gateway
interface) so any path or agent can be tested offline, deterministically, with
no network, no API key, and no cost.

FakeGateway implements LLMGateway and returns canned responses:
    - structured: a dict returned on every call, OR a list of dicts returned in
      sequence (so multi-call flows like the interview loop can script distinct
      responses per turn).
    - text: a fixed string for complete_text.
    - chunks: a list of strings yielded by stream_text.

It records every call in `.calls` so tests can assert what was asked and which
model tier was requested.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from app.llm.base import LLMGateway, LLMMessage, ModelTier


class FakeGateway(LLMGateway):
    def __init__(
        self,
        *,
        structured: dict[str, Any] | list[dict[str, Any]] | None = None,
        text: str = "",
        chunks: Sequence[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        if isinstance(structured, list):
            self._queue: list[dict[str, Any]] | None = list(structured)
            self._default: dict[str, Any] = structured[-1] if structured else {}
        else:
            self._queue = None
            self._default = structured or {}
        self._text = text
        self._chunks = list(chunks or [])
        # If set, every call raises this — used to test failure handling (e.g. the
        # safety guard falling back to deterministic rules when the LLM is down).
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"kind": "structured", "user": user, "model": model})
        if self._error:
            raise self._error
        if self._queue:
            return dict(self._queue.pop(0))
        return dict(self._default)

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> str:
        self.calls.append({"kind": "text", "user": user, "model": model})
        if self._error:
            raise self._error
        return self._text

    async def stream_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> AsyncIterator[str]:
        self.calls.append({"kind": "stream", "user": user, "model": model})
        if self._error:
            raise self._error
        for chunk in self._chunks:
            yield chunk

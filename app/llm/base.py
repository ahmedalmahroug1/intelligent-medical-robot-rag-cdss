"""
app/llm/base.py
---------------
The provider-agnostic LLM interface that every agent depends on.

Why an interface (not just a concrete OpenAI class):
    1. Swappability — the concrete provider can change (a different model, a
       different vendor, or a local model on the Pi) by adding one implementation,
       without touching any agent. Agents depend on this abstraction, not on
       OpenAI.
    2. Testability — paths and agents can be tested offline and deterministically
       against a FakeGateway, with no network, no API key, and no cost. This alone
       justifies the interface: without it, testing the diagnostic path would
       require real OpenAI calls.

This module depends only on the standard library and Pydantic — and the Pydantic
dependency is the library itself (BaseModel/ValidationError, used by
complete_typed), NOT any domain model. It defines WHAT a gateway does; the OpenAI
specifics live in openai_gateway.py.

The interface offers three primitives; paths compose them as needed:
    - complete_structured → one JSON object (for agents that return structured
      data, e.g. the differential + next-question reasoner).
    - complete_text       → one plain-text answer (e.g. report prose).
    - stream_text         → text streamed chunk-by-chunk (for the patient-facing
      reply, later surfaced over SSE so words appear as they are generated).

It deliberately does NOT know about domain models (Hypothesis, VisitState, …).
It returns plain dicts/strings; callers validate those into their own models.
That keeps the gateway pure infrastructure with zero coupling to the clinical
layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Literal, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

# Bound to Pydantic models so complete_typed (below) can validate LLM output into
# any caller-supplied schema. This is the ONLY non-stdlib import in this module,
# and it is the Pydantic library itself — NOT any domain model. The gateway still
# knows nothing about Hypothesis/VisitState/etc.; the caller passes its own schema
# and gets an instance of it back.
T = TypeVar("T", bound=BaseModel)


class ModelTier(str, Enum):
    """
    Which class of model to use for a call. The gateway maps each tier to a
    concrete model name (in settings), so an agent expresses intent ("this is
    cheap turn-taking" vs "this is hard reasoning") instead of hard-coding a
    model string. Swapping the underlying model is then a config change.
    """
    FAST = "fast"            # cheap/low-latency: conversational turns, extraction
    REASONING = "reasoning"  # most capable: diagnosis, evidence synthesis


@dataclass(frozen=True)
class LLMMessage:
    """
    One prior conversational turn passed as history. Immutable.

    A small gateway-level type kept separate from the domain `Message` on purpose:
    the gateway is infrastructure and must not couple to the clinical model. The
    diagnostic path adapts its transcript into these when it needs history.
    """
    role: Literal["user", "assistant"]
    content: str


class LLMError(RuntimeError):
    """
    Any failure while talking to the LLM (timeout, transport, bad output),
    wrapping the provider's own exception types so callers catch one error and
    stay decoupled from the underlying SDK.
    """


class LLMGateway(ABC):
    """The single interface every agent uses to talk to a language model."""

    @abstractmethod
    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> dict[str, Any]:
        """Return a single parsed JSON object. Raises LLMError on failure or
        if the model does not return valid JSON."""
        ...

    @abstractmethod
    async def complete_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> str:
        """Return a single plain-text completion. Raises LLMError on failure."""
        ...

    @abstractmethod
    def stream_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion as text chunks. Implemented as an async generator,
        so callers do `async for chunk in gateway.stream_text(...)`.
        Raises LLMError if the stream cannot be started or fails.
        """
        ...

    # ── Convenience built on the primitives (concrete, shared by all gateways) ─

    async def complete_typed(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> T:
        """
        Return a validated instance of `schema` (a Pydantic model).

        Thin convenience built ON TOP of `complete_structured`: it makes the call,
        then validates the returned dict into `schema`. This is the single place a
        schema mismatch becomes a clean LLMError, so every path that needs typed
        output gets the same parsing-and-error behaviour for free. The dict
        primitive stays available underneath for the rare caller that must inspect
        or repair raw output before validating.

        This is deliberately a CONCRETE method on the base class rather than one
        abstract method implemented per gateway (as the original plan sketched):
        defining it once means the validation/error policy can never drift between
        the OpenAI gateway and the test fake, and any new gateway inherits it
        automatically. It calls only the abstract `complete_structured`, so it
        works for every implementation.
        """
        data = await self.complete_structured(
            system=system, user=user, history=history, model=model
        )
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMError(
                f"LLM output did not match schema {schema.__name__}: {exc}"
            ) from exc

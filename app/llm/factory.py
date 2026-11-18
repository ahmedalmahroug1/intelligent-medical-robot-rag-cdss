"""
app/llm/factory.py
------------------
The single place the LLM PROVIDER is chosen. Composition roots (the API app, the
evaluation runners) call `make_gateway(...)` instead of constructing a concrete
gateway, so switching between OpenAI, a free OpenAI-compatible host (Groq /
OpenRouter / Cerebras / Gemini), or a local Ollama model is one setting —
`LLM_PROVIDER` — and touches nothing else. Every component still depends only on
the `LLMGateway` interface; this factory is the one seam that knows the concretes.

The provider implementations are imported LAZILY (inside the branches) so that:
  - importing this module never requires the `openai` package or an API key, and
  - selecting Ollama never imports the OpenAI SDK (and vice versa).

OpenAI-compatible providers (Groq/OpenRouter/Cerebras/Gemini) all speak the OpenAI
Chat Completions wire protocol, so they reuse `OpenAIGateway` unchanged — the
factory only supplies the right base URL and reads that provider's API key. This is
this keeps provider endpoint and credential selection in one place, without changes
to the components above the gateway.

`models` (a ModelTier→name map) and any provider-specific keyword (e.g. `api_key`,
`host`) pass straight through, so existing call sites — including the evaluation's
per-config model overrides — keep working unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.config import settings
from app.llm.base import LLMGateway, ModelTier

_OPENAI = {"openai", ""}
_OLLAMA = {"ollama", "local"}

# OpenAI-compatible providers: name -> (base_url, settings-attr holding the API key).
# Each reuses OpenAIGateway with only the host + key changed.
_COMPATIBLE: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
    ),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
}


def make_gateway(
    *,
    provider: str | None = None,
    models: Mapping[ModelTier, str] | None = None,
    **kwargs,
) -> LLMGateway:
    """
    Build the configured LLM gateway.

    provider: "openai", "ollama", or an OpenAI-compatible host name
              ("groq" / "openrouter" / "cerebras" / "gemini"); defaults to
              `settings.LLM_PROVIDER`.
    models:   optional ModelTier→model-name overrides (used by the eval to compare
              models without touching settings).
    **kwargs: forwarded to the chosen gateway (e.g. api_key=, base_url=, host=,
              temperature=).
    """
    name = (provider or settings.LLM_PROVIDER or "openai").strip().lower()

    if name in _OLLAMA:
        from app.llm.ollama_gateway import OllamaGateway
        return OllamaGateway(models=models, **kwargs)

    if name in _COMPATIBLE:
        base_url, key_attr = _COMPATIBLE[name]
        from app.llm.openai_gateway import OpenAIGateway
        # Supply the provider's key + base URL unless the caller overrode them.
        kwargs.setdefault("api_key", getattr(settings, key_attr, "") or None)
        kwargs.setdefault("base_url", base_url)
        if not kwargs.get("api_key"):
            raise ValueError(
                f"LLM_PROVIDER={name!r} needs {key_attr}. Set it in the environment "
                f"(e.g. `set {key_attr}=...`) or pass api_key=..."
            )
        return OpenAIGateway(models=models, **kwargs)

    if name in _OPENAI:
        from app.llm.openai_gateway import OpenAIGateway
        return OpenAIGateway(models=models, **kwargs)

    raise ValueError(
        f"Unknown LLM_PROVIDER {name!r}; expected 'openai', 'ollama', or one of "
        f"{sorted(_COMPATIBLE)}."
    )

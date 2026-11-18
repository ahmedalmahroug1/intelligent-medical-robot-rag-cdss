"""
app/llm/ollama_gateway.py
-------------------------
The Ollama-backed implementation of LLMGateway: runs the system against LOCAL
models served by an Ollama daemon, with no API key and no per-call cost.

It is a peer of `openai_gateway.OpenAIGateway` — both implement the SAME
`LLMGateway` interface, so every component (guard, classifier, diagnosis path,
milestone, evaluation) works unchanged regardless of which one is wired in. The
provider is chosen once, in `app.llm.factory.make_gateway`, from `LLM_PROVIDER`.

Why the native Ollama API (httpx) rather than the OpenAI-compatible shim:
    Ollama exposes `/api/chat` with first-class `format: "json"` (grammar-constrained
    valid JSON), `options.temperature`, and newline-delimited streaming. Using it
    directly keeps full control of those behaviours and depends only on httpx (already
    a project dependency) — no reliance on the OpenAI SDK's expectations.

Model auto-detection (`ollama list`):
    Local model names are not known ahead of time, so empty tier settings are
    resolved from the INSTALLED models. `list_ollama_models()` runs the `ollama list`
    CLI and, if the binary is not on PATH, falls back to the daemon's `/api/tags`
    endpoint (which is exactly what the CLI calls) — same data either way. The
    smallest installed model backs the FAST tier, the largest backs REASONING; both
    are overridable via OLLAMA_MODEL_FAST / OLLAMA_MODEL_REASONING.

Robustness for local models:
    Small local models are less reliable at emitting clean JSON than GPT-4o. Two
    defences: every structured call sets `format: "json"`, and `_extract_json` rescues
    output that wraps the object in prose, markdown fences, or <think>…</think> traces
    (common with reasoning-tuned local models) before giving up with an LLMError.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncIterator, Sequence
from collections.abc import Mapping

import httpx

from app.config import settings
from app.llm.base import LLMError, LLMGateway, LLMMessage, ModelTier

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Installed-model discovery  (`ollama list`, with an /api/tags fallback)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OllamaModel:
    """An installed model, as reported by `ollama list` / `/api/tags`."""
    name: str
    size_bytes: int


_SIZE_UNITS = {
    "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
    "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40,
}


def _parse_size(text: str) -> int:
    """'4.7 GB' -> 4_700_000_000. Returns 0 if it cannot be parsed."""
    m = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)\s*$", text)
    if not m:
        return 0
    value, unit = m.group(1), m.group(2).upper()
    try:
        return int(float(value) * _SIZE_UNITS.get(unit, 1))
    except ValueError:
        return 0


def _parse_cli_table(stdout: str) -> list[OllamaModel]:
    """
    Parse the `ollama list` table. Columns are whitespace-aligned and several
    values contain spaces ('4.7 GB', '2 weeks ago'), so split on runs of 2+ spaces.

        NAME              ID            SIZE      MODIFIED
        llama3.2:3b       a80c4f17acd5  2.0 GB    2 weeks ago
    """
    models: list[OllamaModel] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if not parts or parts[0].upper() == "NAME":
            continue
        name = parts[0].strip()
        size = _parse_size(parts[2]) if len(parts) >= 3 else 0
        if name:
            models.append(OllamaModel(name=name, size_bytes=size))
    return models


def _ollama_binary() -> str | None:
    """Locate the `ollama` CLI: PATH first, then the default Windows install dir."""
    exe = shutil.which("ollama")
    if exe:
        return exe
    win = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
    return win if os.path.isfile(win) else None


def _cli_models() -> list[OllamaModel] | None:
    """Installed models via the `ollama list` CLI, or None if it cannot be used."""
    exe = _ollama_binary()
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "list"], capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("`ollama list` failed (%s); will try the HTTP API", exc)
        return None
    if proc.returncode != 0:
        logger.debug("`ollama list` exited %d; will try the HTTP API", proc.returncode)
        return None
    return _parse_cli_table(proc.stdout)


def _http_models(host: str) -> list[OllamaModel]:
    """Installed models via the daemon's /api/tags (what `ollama list` calls)."""
    try:
        resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("Could not list Ollama models from %s/api/tags: %s", host, exc)
        return []
    out: list[OllamaModel] = []
    for m in data.get("models", []):
        name = m.get("name") or m.get("model")
        if name:
            out.append(OllamaModel(name=name, size_bytes=int(m.get("size", 0) or 0)))
    return out


@lru_cache(maxsize=4)
def list_ollama_models(host: str | None = None) -> tuple[OllamaModel, ...]:
    """
    The installed Ollama models. Tries the `ollama list` CLI first (as requested),
    then falls back to the daemon's HTTP `/api/tags`. Cached per host within a
    process so a config sweep does not re-shell out for every gateway it builds.
    """
    host = host or settings.OLLAMA_HOST
    cli = _cli_models()
    if cli:  # non-empty CLI result wins
        return tuple(cli)
    return tuple(_http_models(host))


def _is_installed(name: str, installed: set[str]) -> bool:
    """Lenient match: exact, or the bare name against an installed ':tag' variant."""
    if name in installed:
        return True
    if ":" not in name:
        return f"{name}:latest" in installed or any(
            i.split(":", 1)[0] == name for i in installed
        )
    return False


def resolve_models(
    *,
    explicit: Mapping[ModelTier, str] | None = None,
    host: str | None = None,
    auto_detect: bool = True,
) -> dict[ModelTier, str]:
    """
    Decide the concrete model name for each tier.

    Precedence: an explicit `models=` override → the OLLAMA_MODEL_* settings →
    (if `auto_detect`) the installed models, smallest→FAST and largest→REASONING. A
    configured-but-not-installed name is reported and replaced by an auto pick rather
    than failing at first call.
    """
    mapping: dict[ModelTier, str | None] = {
        ModelTier.FAST: settings.OLLAMA_MODEL_FAST or None,
        ModelTier.REASONING: settings.OLLAMA_MODEL_REASONING or None,
    }
    for tier, name in (explicit or {}).items():
        if name:
            mapping[tier] = name

    if not auto_detect:
        missing = [t.value for t, n in mapping.items() if not n]
        if missing:
            raise LLMError(
                f"Ollama model not set for tier(s) {missing} and auto_detect is off. "
                "Set OLLAMA_MODEL_FAST / OLLAMA_MODEL_REASONING."
            )
        return {t: n for t, n in mapping.items() if n is not None}  # type: ignore[misc]

    installed = list_ollama_models(host)
    names = {m.name for m in installed}
    by_size = sorted(installed, key=lambda m: m.size_bytes)
    smallest = by_size[0].name if by_size else None
    largest = by_size[-1].name if by_size else None
    auto = {ModelTier.FAST: smallest, ModelTier.REASONING: largest}

    for tier in (ModelTier.FAST, ModelTier.REASONING):
        chosen = mapping[tier]
        if chosen and not _is_installed(chosen, names):
            logger.warning(
                "Ollama %s model %r is not installed; auto-selecting from `ollama list`.",
                tier.value, chosen,
            )
            chosen = None
        if not chosen:
            chosen = auto[tier]
        mapping[tier] = chosen

    if not mapping[ModelTier.FAST] or not mapping[ModelTier.REASONING]:
        raise LLMError(
            "No Ollama models available. Install at least one (e.g. "
            "`ollama pull llama3.2:3b`) or set OLLAMA_MODEL_FAST / OLLAMA_MODEL_REASONING."
        )
    resolved = {t: n for t, n in mapping.items() if n is not None}  # type: ignore[misc]
    logger.info(
        "Ollama models: FAST=%s, REASONING=%s",
        resolved[ModelTier.FAST], resolved[ModelTier.REASONING],
    )
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# JSON rescue for less-reliable local output
# ─────────────────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)


def _extract_json(content: str) -> Any:
    """
    Parse a JSON value from model output, tolerating the wrappers local models add
    (a leading explanation, ```json fences, or <think>…</think> traces). Raises
    json.JSONDecodeError if nothing parseable is found.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    cleaned = _THINK_RE.sub("", content).strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        return json.loads(fence.group(1))
    # Fall back to the outermost {...} (or [...]) span.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(open_c), cleaned.rfind(close_c)
        if 0 <= start < end:
            return json.loads(cleaned[start:end + 1])
    raise json.JSONDecodeError("no JSON value found in model output", content, 0)


# ─────────────────────────────────────────────────────────────────────────────
# The gateway
# ─────────────────────────────────────────────────────────────────────────────

class OllamaGateway(LLMGateway):
    """LLMGateway backed by a local Ollama daemon (native /api/chat)."""

    def __init__(
        self,
        *,
        host: str | None = None,
        default_tier: ModelTier = ModelTier.REASONING,
        timeout: float | None = None,
        max_retries: int | None = None,
        temperature: float | None = None,
        backoff_s: float = 0.5,
        models: Mapping[ModelTier, str] | None = None,
        auto_detect: bool = True,
    ) -> None:
        self._host = (host or settings.OLLAMA_HOST).rstrip("/")
        self._timeout = settings.LLM_TIMEOUT_S if timeout is None else timeout
        self._max_retries = settings.LLM_MAX_RETRIES if max_retries is None else max_retries
        self._temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        self._backoff = backoff_s
        self._default_tier = default_tier
        # Resolve tier→model now (auto-detecting installed models) so a bad/empty
        # configuration fails clearly at construction, not mid-consultation.
        self._model_for = resolve_models(
            explicit=models, host=self._host, auto_detect=auto_detect
        )
        self._client = httpx.AsyncClient(base_url=self._host, timeout=self._timeout)

    def model_name_for(self, tier: ModelTier) -> str:
        """The concrete model name a tier maps to (useful for logging/tests)."""
        return self._model_for[tier]

    @property
    def models(self) -> dict[ModelTier, str]:
        return dict(self._model_for)

    # ── Public API ───────────────────────────────────────────────────────────

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> dict[str, Any]:
        content = await self._chat(
            self._build(system, user, history), model, json_mode=True
        )
        try:
            data = _extract_json(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError(
                f"Ollama returned a JSON {type(data).__name__}, expected an object."
            )
        return data

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> str:
        return await self._chat(self._build(system, user, history), model, json_mode=False)

    async def stream_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> AsyncIterator[str]:
        model_name = self._resolve(model)
        payload = self._payload(model_name, self._build(system, user, history),
                                stream=True, json_mode=False)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    piece = (chunk.get("message") or {}).get("content")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama streaming failed: {exc}") from exc

    # ── Internals ────────────────────────────────────────────────────────────

    def _resolve(self, tier: ModelTier | None) -> str:
        return self._model_for[tier or self._default_tier]

    def _build(
        self, system: str, user: str, history: Sequence[LLMMessage] | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if history:
            messages.extend({"role": m.role, "content": m.content} for m in history)
        messages.append({"role": "user", "content": user})
        return messages

    def _payload(
        self, model_name: str, messages: list[dict[str, str]], *, stream: bool, json_mode: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self._temperature},
        }
        # Thinking-capable local models (notably Qwen3) enable long reasoning traces
        # by default.  The evaluation runner can disable that work for a latency
        # profile without changing prompts or model files:
        #     OLLAMA_THINK=false
        # Leave the variable unset to preserve Ollama's/model's normal default.
        think = os.environ.get("OLLAMA_THINK", "").strip().lower()
        if think in {"1", "true", "yes", "on"}:
            payload["think"] = True
        elif think in {"0", "false", "no", "off"}:
            payload["think"] = False
        elif think in {"low", "medium", "high"}:
            payload["think"] = think
        if json_mode:
            payload["format"] = "json"  # grammar-constrained valid JSON
        return payload

    async def _chat(
        self, messages: list[dict[str, str]], tier: ModelTier | None, *, json_mode: bool,
    ) -> str:
        """Non-streaming call with a logged retry loop. Returns the message content."""
        model_name = self._resolve(tier)
        payload = self._payload(model_name, messages, stream=False, json_mode=json_mode)
        attempts = self._max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                resp = await self._client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                self._log_usage(model_name, data, attempt)
                return (data.get("message") or {}).get("content") or ""
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "Ollama attempt %d/%d failed (model=%s): %s",
                    attempt, attempts, model_name, exc,
                )
                if attempt < attempts:
                    await asyncio.sleep(self._backoff * attempt)  # linear backoff

        raise LLMError(
            f"Ollama call failed after {attempts} attempt(s): "
            f"{type(last_exc).__name__}: {last_exc}"  # httpx timeouts stringify to ''
        ) from last_exc

    def _log_usage(self, model_name: str, data: Mapping[str, Any], attempt: int) -> None:
        # Ollama reports token counts on the final response object.
        if data.get("eval_count") is not None:
            logger.info(
                "Ollama ok model=%s attempt=%d prompt=%s completion=%s",
                model_name, attempt,
                data.get("prompt_eval_count", "?"), data.get("eval_count", "?"),
            )

    async def aclose(self) -> None:
        await self._client.aclose()

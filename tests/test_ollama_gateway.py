"""
tests/test_ollama_gateway.py
----------------------------
Tests for the Ollama provider and the gateway factory. Run:

    python -m tests.test_ollama_gateway

Almost everything runs OFFLINE: model discovery, the smallest→FAST / largest→
REASONING auto-detection, the JSON-rescue parser, structured-output contract, and
provider selection are exercised with stubs (no daemon, no network). One guarded
LIVE test runs a real structured call only if an Ollama server is reachable, so it
verifies the end-to-end path on a machine that actually has Ollama.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import traceback

import httpx

from app.llm.base import LLMError, LLMGateway, ModelTier
from app.llm import ollama_gateway as og
from app.llm.ollama_gateway import (
    OllamaGateway,
    OllamaModel,
    _extract_json,
    _parse_cli_table,
    _parse_size,
    resolve_models,
)
from app.llm.factory import make_gateway

try:
    from app.llm.openai_gateway import OpenAIGateway
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


# ── Model discovery ───────────────────────────────────────────────────────────

def test_parse_size_units() -> None:
    assert _parse_size("2.0 GB") == 2_000_000_000
    assert _parse_size("274 MB") == 274_000_000
    assert _parse_size("4.7 GB") == 4_700_000_000
    assert _parse_size("garbage") == 0


def test_parse_cli_table_skips_header_and_reads_sizes() -> None:
    table = (
        "NAME              ID            SIZE      MODIFIED\n"
        "llama3.2:3b       a80c4f17acd5  2.0 GB    2 weeks ago\n"
        "qwen3:14b         deadbeef0000  9.3 GB    3 weeks ago\n"
    )
    models = _parse_cli_table(table)
    assert [m.name for m in models] == ["llama3.2:3b", "qwen3:14b"]
    assert models[0].size_bytes == 2_000_000_000
    assert models[1].size_bytes == 9_300_000_000


# ── Auto-detection / tier resolution ──────────────────────────────────────────

def _patch_installed(models: tuple[OllamaModel, ...]):
    """Temporarily replace the installed-model lookup; returns a restore callable."""
    original = og.list_ollama_models
    og.list_ollama_models = lambda host=None: models  # type: ignore[assignment]
    return lambda: setattr(og, "list_ollama_models", original)


def test_resolve_models_auto_picks_smallest_fast_largest_reasoning() -> None:
    restore = _patch_installed((
        OllamaModel("phi4-mini:latest", 2_500_000_000),
        OllamaModel("llama3.2:3b", 2_000_000_000),
        OllamaModel("qwen3:14b", 9_300_000_000),
    ))
    try:
        resolved = resolve_models(explicit=None, auto_detect=True)
        assert resolved[ModelTier.FAST] == "llama3.2:3b"     # smallest
        assert resolved[ModelTier.REASONING] == "qwen3:14b"  # largest
    finally:
        restore()


def test_resolve_models_explicit_honored_else_fallback() -> None:
    restore = _patch_installed((
        OllamaModel("llama3.2:3b", 2_000_000_000),
        OllamaModel("qwen3:14b", 9_300_000_000),
    ))
    try:
        # An installed explicit choice is kept; a not-installed one falls back to auto.
        resolved = resolve_models(
            explicit={ModelTier.FAST: "qwen3:14b", ModelTier.REASONING: "gpt-4o"},
            auto_detect=True,
        )
        assert resolved[ModelTier.FAST] == "qwen3:14b"        # explicit + installed
        assert resolved[ModelTier.REASONING] == "qwen3:14b"   # gpt-4o absent → largest
    finally:
        restore()


def test_resolve_models_no_models_raises() -> None:
    restore = _patch_installed(())
    try:
        raised = False
        try:
            resolve_models(explicit=None, auto_detect=True)
        except LLMError:
            raised = True
        assert raised, "no installed models and nothing configured must raise"
    finally:
        restore()


def test_resolve_models_no_auto_detect_requires_explicit() -> None:
    raised = False
    try:
        resolve_models(explicit=None, auto_detect=False)  # nothing set
    except LLMError:
        raised = True
    assert raised
    pinned = resolve_models(
        explicit={ModelTier.FAST: "a", ModelTier.REASONING: "b"}, auto_detect=False,
    )
    assert pinned == {ModelTier.FAST: "a", ModelTier.REASONING: "b"}


# ── JSON rescue ───────────────────────────────────────────────────────────────

def test_extract_json_handles_wrappers() -> None:
    assert _extract_json('{"ok": true}') == {"ok": True}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('<think>hmm let me see</think>\n{"b": 2}') == {"b": 2}
    assert _extract_json('Sure! Here it is: {"c": 3} hope that helps') == {"c": 3}
    raised = False
    try:
        _extract_json("no json at all")
    except Exception:
        raised = True
    assert raised


# ── Gateway construction + structured-output contract (offline) ───────────────

def _offline_gateway() -> OllamaGateway:
    # auto_detect=False + explicit models => no network in the constructor.
    return OllamaGateway(
        models={ModelTier.FAST: "fast-model", ModelTier.REASONING: "reason-model"},
        auto_detect=False,
    )


def test_gateway_is_an_llmgateway_and_maps_tiers() -> None:
    gw = _offline_gateway()
    assert isinstance(gw, LLMGateway)
    assert gw.model_name_for(ModelTier.FAST) == "fast-model"
    assert gw.model_name_for(ModelTier.REASONING) == "reason-model"


def test_payload_can_disable_thinking_from_environment() -> None:
    gw = _offline_gateway()
    previous = os.environ.get("OLLAMA_THINK")
    try:
        os.environ["OLLAMA_THINK"] = "false"
        payload = gw._payload(
            "reason-model",
            [{"role": "user", "content": "test"}],
            stream=False,
            json_mode=False,
        )
        assert payload["think"] is False
    finally:
        if previous is None:
            os.environ.pop("OLLAMA_THINK", None)
        else:
            os.environ["OLLAMA_THINK"] = previous


async def test_complete_structured_enforces_object() -> None:
    gw = _offline_gateway()

    async def good(messages, tier, *, json_mode):
        return '{"diagnosis": "viral", "confidence": 0.6}'

    async def array(messages, tier, *, json_mode):
        return "[1, 2, 3]"

    async def junk(messages, tier, *, json_mode):
        return "the model rambled without json"

    gw._chat = good  # type: ignore[assignment]
    out = await gw.complete_structured(system="s", user="u")
    assert out == {"diagnosis": "viral", "confidence": 0.6}

    for bad in (array, junk):
        gw._chat = bad  # type: ignore[assignment]
        raised = False
        try:
            await gw.complete_structured(system="s", user="u")
        except LLMError:
            raised = True
        assert raised, "non-object / non-JSON output must raise LLMError"


# ── Factory provider selection ────────────────────────────────────────────────

def test_factory_selects_ollama() -> None:
    gw = make_gateway(
        provider="ollama",
        models={ModelTier.FAST: "a", ModelTier.REASONING: "b"},
        auto_detect=False,
    )
    assert isinstance(gw, OllamaGateway)


def test_factory_selects_openai_and_rejects_unknown() -> None:
    if _HAS_OPENAI:
        gw = make_gateway(provider="openai", api_key="dummy-key-not-used")
        assert isinstance(gw, OpenAIGateway)
    raised = False
    try:
        make_gateway(provider="bogus")
    except ValueError:
        raised = True
    assert raised


# ── Guarded live test (only if an Ollama server is reachable) ─────────────────

def _server_up() -> bool:
    from app.config import settings
    try:
        r = httpx.get(f"{settings.OLLAMA_HOST.rstrip('/')}/api/tags", timeout=2.0)
        r.raise_for_status()
        return bool(r.json().get("models"))
    except Exception:
        return False


async def test_ollama_live_optional() -> None:
    if not _server_up():
        print("    (skipped: no reachable Ollama server with models)")
        return
    gw = OllamaGateway()  # real auto-detection from the running daemon
    out = await gw.complete_structured(
        system="You return only JSON.",
        user='Return exactly {"ok": true}.',
        model=ModelTier.FAST,
    )
    await gw.aclose()
    assert isinstance(out, dict)
    assert out.get("ok") is True


ALL_TESTS = [
    test_parse_size_units,
    test_parse_cli_table_skips_header_and_reads_sizes,
    test_resolve_models_auto_picks_smallest_fast_largest_reasoning,
    test_resolve_models_explicit_honored_else_fallback,
    test_resolve_models_no_models_raises,
    test_resolve_models_no_auto_detect_requires_explicit,
    test_extract_json_handles_wrappers,
    test_gateway_is_an_llmgateway_and_maps_tiers,
    test_payload_can_disable_thinking_from_environment,
    test_complete_structured_enforces_object,
    test_factory_selects_ollama,
    test_factory_selects_openai_and_rejects_unknown,
    test_ollama_live_optional,
]


def main() -> None:
    passed = failed = 0
    for test in ALL_TESTS:
        try:
            result = test()
            if inspect.iscoroutine(result):
                asyncio.run(result)
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

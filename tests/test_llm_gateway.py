"""
tests/test_llm_gateway.py
-------------------------
Tests for the LLM gateway layer — File 3.

Two groups:

  A) FakeGateway + interface (run fully offline, no dependencies beyond stdlib).
     These prove the test double future path-tests will rely on actually behaves:
     canned structured/text/stream responses, sequenced responses for multi-call
     flows, call recording, and conformance to the LLMGateway interface.

  B) OpenAIGateway (guarded). Constructing it needs the `openai` package
     importable; the real network call needs a key. So:
       - construction/validation and tier-mapping tests run whenever `openai`
         is installed (no network needed — the client makes no call until used);
       - one optional live test runs only if OPENAI_API_KEY is set, letting you
         verify the real call end-to-end.
     If `openai` is not installed, group B is skipped automatically.

Run:
    pip install -r requirements.txt
    python -m tests.test_llm_gateway
"""

from __future__ import annotations

import asyncio
import os

from app.llm.base import LLMError, LLMGateway, LLMMessage, ModelTier
from tests.fakes import FakeGateway

# Guarded import: the OpenAI implementation needs the `openai` package.
try:
    from app.llm.openai_gateway import OpenAIGateway
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


# ─────────────────────────────────────────────────────────────────────────────
# A) FakeGateway + interface — fully offline
# ─────────────────────────────────────────────────────────────────────────────

async def test_fake_structured_returns_canned_dict():
    gw = FakeGateway(structured={"diagnosis": "angina", "confidence": 0.7})
    out = await gw.complete_structured(system="s", user="u")
    assert out == {"diagnosis": "angina", "confidence": 0.7}
    # Returns a copy, not the internal object (mutating the result is safe).
    out["diagnosis"] = "changed"
    out2 = await gw.complete_structured(system="s", user="u")
    assert out2["diagnosis"] == "angina"


async def test_fake_structured_sequence_returns_in_order():
    gw = FakeGateway(structured=[{"q": 1}, {"q": 2}, {"q": 3}])
    a = await gw.complete_structured(system="s", user="u")
    b = await gw.complete_structured(system="s", user="u")
    c = await gw.complete_structured(system="s", user="u")
    d = await gw.complete_structured(system="s", user="u")  # exhausted → last
    assert [a["q"], b["q"], c["q"]] == [1, 2, 3]
    assert d["q"] == 3


async def test_fake_text_returns_fixed_string():
    gw = FakeGateway(text="hello there")
    assert await gw.complete_text(system="s", user="u") == "hello there"


async def test_fake_stream_yields_chunks_in_order():
    gw = FakeGateway(chunks=["Hel", "lo, ", "world"])
    collected = [chunk async for chunk in gw.stream_text(system="s", user="u")]
    assert collected == ["Hel", "lo, ", "world"]
    assert "".join(collected) == "Hello, world"


async def test_fake_records_calls_and_model_tier():
    gw = FakeGateway(structured={"ok": True}, text="t", chunks=["x"])
    await gw.complete_structured(system="s", user="u1", model=ModelTier.REASONING)
    await gw.complete_text(system="s", user="u2", model=ModelTier.FAST)
    async for _ in gw.stream_text(system="s", user="u3"):
        pass
    kinds = [c["kind"] for c in gw.calls]
    assert kinds == ["structured", "text", "stream"]
    assert gw.calls[0]["model"] is ModelTier.REASONING
    assert gw.calls[1]["model"] is ModelTier.FAST
    assert gw.calls[0]["user"] == "u1"


def test_fake_is_an_llmgateway():
    # Future path-tests depend on this substitutability.
    assert isinstance(FakeGateway(), LLMGateway)


def test_model_tier_values_are_stable_strings():
    # These strings may end up in logs/config; pin them.
    assert ModelTier.FAST.value == "fast"
    assert ModelTier.REASONING.value == "reasoning"


def test_llmmessage_is_immutable():
    m = LLMMessage(role="user", content="hi")
    assert m.role == "user" and m.content == "hi"
    raised = False
    try:
        m.content = "changed"  # frozen dataclass → should fail
    except Exception:
        raised = True
    assert raised


def test_llmerror_is_an_exception():
    assert issubclass(LLMError, Exception)


# ─────────────────────────────────────────────────────────────────────────────
# B) OpenAIGateway — guarded (needs `openai` installed; live test needs a key)
# ─────────────────────────────────────────────────────────────────────────────

def test_openai_construction_requires_key():
    if not _HAS_OPENAI:
        print("    (skipped: openai not installed)")
        return
    raised = False
    try:
        OpenAIGateway(api_key="")          # empty key must be rejected
    except ValueError:
        raised = True
    assert raised, "OpenAIGateway should reject an empty API key"


def test_openai_tier_maps_to_model_names():
    if not _HAS_OPENAI:
        print("    (skipped: openai not installed)")
        return
    # Construction makes no network call, so a dummy key is fine here.
    gw = OpenAIGateway(api_key="dummy-key-not-used")
    assert gw.model_name_for(ModelTier.FAST) == "gpt-4o-mini"
    assert gw.model_name_for(ModelTier.REASONING) == "gpt-4o"


async def test_openai_live_optional():
    if not _HAS_OPENAI:
        print("    (skipped: openai not installed)")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("    (skipped: set OPENAI_API_KEY to run the live call)")
        return
    gw = OpenAIGateway()
    out = await gw.complete_structured(
        system="You return only JSON.",
        user='Return exactly {"ok": true}.',
        model=ModelTier.FAST,
    )
    assert isinstance(out, dict)
    assert out.get("ok") is True


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (handles sync + async tests; skips print their reason)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import inspect
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            if inspect.iscoroutinefunction(t):
                asyncio.run(t())
            else:
                t()
        except Exception:
            failed += 1
            print(f"[FAIL]  {t.__name__}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[PASS]  {t.__name__}")

    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)

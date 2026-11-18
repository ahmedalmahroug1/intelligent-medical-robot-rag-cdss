"""
tests/test_concierge.py
-----------------------
Offline tests for the Stage-A concierge (conversation owner). The CDSS reasoning engine is a
StubFacade (canned differential + suggested question + sufficiency), so the loop is tested with
no network. What these prove:
  1. SHARED sufficiency: even when the engine says `sufficient`, the concierge KEEPS interviewing
     while a critical field (duration/severity) is missing — and ENDS (bridges) when both agree.
  2. The running ClinicalPicture merges across turns.
  3. Meta / non-clinical turns are handled by the concierge and NEVER reach the engine.
  4. LLM-driven identity: a non-name ("speak in english") is NOT taken as the name — the
     concierge asks again; a real name is registered.
  5. Helpers behave.

    PYTHONPATH=. python tests/test_concierge.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile

sys.path.insert(0, ".")

from app.concierge import prompts
from app.concierge.agent import ConciergeAgent, _looks_like_name
from app.concierge.identity import (
    IdentityRegistry, detect_language, explicit_language_switch, parse_name,
)
from app.concierge.schemas import AssessResult, FinalizeResult
from app.core.enums import RiskTier
from app.core.state import WorkingDiagnosis
from tests.fakes import FakeGateway


class StubFacade:
    """Stands in for the CDSS reasoning engine: canned assess()/finalize(), records call counts."""

    def __init__(self, *, sufficient: bool = True, confidence: float = 0.9,
                 question: str = "Does it radiate?"):
        self.confidence_floor = 0.6
        self._sufficient = sufficient
        self._confidence = confidence
        self._question = question
        self.calls = 0
        self.finalize_calls = 0

    async def assess(self, picture) -> AssessResult:
        self.calls += 1
        return AssessResult(hypotheses=[], suggested_question=self._question,
                            sufficient=self._sufficient, confidence=self._confidence)

    async def finalize(self, picture, *, differential=None) -> FinalizeResult:
        self.finalize_calls += 1
        return FinalizeResult(working_diagnosis=WorkingDiagnosis(
            condition="Migraine", confidence=0.8, risk_tier=RiskTier.LOW, reasoning="stub"))


def _reg() -> IdentityRegistry:
    return IdentityRegistry(path=tempfile.mktemp(suffix=".json"))


async def _run() -> None:
    # 1+2. SHARED sufficiency + picture merge: engine says "sufficient", but the concierge keeps
    #      going while duration/severity are missing, and ends once they are provided.
    ident = {"name": "Patient", "has_name": True, "language": "en"}
    ext1 = {"is_clinical": True, "chief_complaint": "headache", "symptoms": ["headache"],
            "language": "en"}
    ext2 = {"is_clinical": True, "duration": "2 days", "severity": "7/10", "language": "en"}
    gw = FakeGateway(structured=[ident, ext1, ext2], text="Tell me more about the headache.")
    facade = StubFacade(sufficient=True, confidence=0.9)
    agent = ConciergeAgent(facade=facade, gateway=gw, identity=_reg())

    await agent.respond("my name is Patient")
    r1 = await agent.respond("I have a headache")
    assert facade.calls == 1
    assert r1 == "Tell me more about the headache.", r1     # asked a question, did NOT finish
    assert agent._complete is False
    assert agent._picture.chief_complaint == "headache"

    r2 = await agent.respond("it has been 2 days, severity 7")
    assert facade.calls == 2
    assert agent._picture.duration == "2 days" and agent._picture.severity == "7/10"
    assert agent._complete is True
    assert facade.finalize_calls == 1, "completion must call finalize (Stage B)"  # both agree → finalize

    # 3. meta turn handled by the concierge, NEVER reaches the engine
    gw = FakeGateway(structured=[{"name": "Sara", "has_name": True, "language": "en"},
                                 {"is_clinical": False}], text="I take your information; the system assesses it.")
    facade = StubFacade()
    agent = ConciergeAgent(facade=facade, gateway=gw, identity=_reg())
    await agent.respond("Sara")
    r = await agent.respond("you are the doctor")
    assert facade.calls == 0, "a non-clinical message must not reach the engine"
    assert "diagnos" not in r.lower()
    text_calls = [c for c in gw.calls if c["kind"] == "text"]
    assert text_calls and "Sara" in text_calls[-1]["user"], \
        "the concierge must pass its memory (the name) into its conversational call"

    # 4. LLM-driven identity: a non-name is NOT taken as the name
    gw = FakeGateway(structured=[{"has_name": False, "language": "en"},
                                 {"name": "Patient", "has_name": True, "language": "en"}], text="x")
    agent = ConciergeAgent(facade=StubFacade(), gateway=gw, identity=_reg())
    r = await agent.respond("speak in english")
    assert agent._identified is False, "a non-name must not identify the patient"
    assert "name" in r.lower(), r
    r = await agent.respond("Patient")
    assert agent._identified is True and agent._name == "Patient"

    # 5. helpers
    assert _looks_like_name("Patient") and not _looks_like_name("speak in english")
    assert "diagnos" not in prompts.ask_name("en").lower()
    assert detect_language("مرحبا") == "ar" and detect_language("hello") == "en"
    assert parse_name("my name is Patient Ali").lower().startswith("Patient")

    # 6. latency: a clinical turn that adds NOTHING new must NOT re-consult the engine —
    #    the last assessment is reused (assess is the expensive reasoning call of the turn).
    ident = {"name": "Omar", "has_name": True, "language": "en"}
    ext_new = {"is_clinical": True, "chief_complaint": "headache", "symptoms": ["headache"],
               "language": "en"}
    ext_repeat = {"is_clinical": True, "symptoms": ["headache"], "language": "en"}
    gw = FakeGateway(structured=[ident, ext_new, ext_repeat], text="Since when?")
    facade = StubFacade(sufficient=False)
    agent = ConciergeAgent(facade=facade, gateway=gw, identity=_reg())
    await agent.respond("my name is Omar")
    await agent.respond("I have a headache")
    assert facade.calls == 1
    r = await agent.respond("like I said, a headache")
    assert facade.calls == 1, "no new clinical info must reuse the last assessment"
    assert r == "Since when?", r                       # still answers normally

    # 7. streaming: respond_stream yields the reply chunk by chunk, shares the same flow
    #    (engine consulted once), and records the FULL reply to history exactly once.
    ident = {"name": "Lina", "has_name": True, "language": "en"}
    ext = {"is_clinical": True, "chief_complaint": "cough", "symptoms": ["cough"],
           "language": "en"}
    gw = FakeGateway(structured=[ident, ext], chunks=["How long ", "has it lasted?"])
    facade = StubFacade(sufficient=False)
    agent = ConciergeAgent(facade=facade, gateway=gw, identity=_reg())
    await agent.respond("my name is Lina")
    parts = [c async for c in agent.respond_stream("I have a cough")]
    assert parts == ["How long ", "has it lasted?"], parts
    assert facade.calls == 1
    assert agent._history[-1] == "assistant: How long has it lasted?", agent._history[-1]
    assert agent._history.count("assistant: How long has it lasted?") == 1

    # 7b. a stream that yields nothing falls back to the deterministic question text.
    gw = FakeGateway(structured=[dict(ident), dict(ext)], chunks=[])
    agent = ConciergeAgent(facade=StubFacade(sufficient=False), gateway=gw, identity=_reg())
    await agent.respond("my name is Lina")
    parts = [c async for c in agent.respond_stream("I have a cough")]
    joined = "".join(parts)
    assert joined.strip(), "an empty stream must fall back, never reply with nothing"
    assert agent._history[-1] == f"assistant: {joined}"

    # 8. LANGUAGE LOCK: chosen once (first message's script), then held even when the patient
    #    mixes scripts; only an EXPLICIT request switches it.
    ident = {"name": "Nora", "has_name": True, "language": "en"}
    ar_turn = {"is_clinical": True, "chief_complaint": "cough", "symptoms": ["cough"],
               "language": "ar"}
    en_turn = {"is_clinical": True, "symptoms": ["fever"], "language": "en"}
    meta = {"is_clinical": False, "language": "en"}
    gw = FakeGateway(structured=[ident, ar_turn, en_turn, meta], text="q")
    agent = ConciergeAgent(facade=StubFacade(sufficient=False), gateway=gw, identity=_reg())
    await agent.respond("my name is Nora")
    await agent.respond("عندي كحة")                       # first turn: Arabic script → locked
    assert agent._language == "ar" and agent._language_locked
    await agent.respond("and now I have a fever too")     # English script must NOT flip it
    assert agent._language == "ar", "language must stay locked despite the message script"
    await agent.respond("switch to english please")       # explicit command → switches
    assert agent._language == "en"

    # 9. a returning patient's SAVED language wins, locked from the first word.
    gw = FakeGateway(structured=[{"is_clinical": True, "symptoms": ["headache"],
                                  "language": "ar"}], text="q")
    agent = ConciergeAgent(facade=StubFacade(sufficient=False), gateway=gw, identity=_reg())
    agent.preload(name="Omar", patient_id="p1", language="en")
    await agent.respond("راسي يوجعني")                    # Arabic message; saved language wins
    assert agent._language == "en" and agent._language_locked

    # 10. the deterministic switch detector: choices and commands, never ordinary text.
    assert explicit_language_switch("عربي") == "ar"
    assert explicit_language_switch("العربية") == "ar"
    assert explicit_language_switch("English") == "en"
    assert explicit_language_switch("بالانجليزية من فضلك") == "en"
    assert explicit_language_switch("switch to arabic") == "ar"
    assert explicit_language_switch("I have a headache") is None
    assert explicit_language_switch("عندي صداع") is None

    print("test_concierge: ALL PASS")


def main() -> None:
    asyncio.run(_run())


def test_safety_preempts_extraction_and_reasoning() -> None:
    async def run() -> None:
        async def safety_check(message, picture):
            assert message == "I have crushing chest pain"
            return "Seek emergency care now."

        gateway = FakeGateway(structured={"is_clinical": True}, text="unused")
        facade = StubFacade()
        agent = ConciergeAgent(
            facade=facade,
            gateway=gateway,
            identity=_reg(),
            safety_check=safety_check,
        )
        agent.preload(name="Patient", patient_id="p1", language="en")
        reply = await agent.respond("I have crushing chest pain")
        assert reply == "Seek emergency care now."
        assert facade.calls == 0
        assert gateway.calls == []
        assert agent._history[-2:] == [
            "patient: I have crushing chest pain",
            "assistant: Seek emergency care now.",
        ]

    asyncio.run(run())


if __name__ == "__main__":
    main()


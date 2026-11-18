"""Offline tests for the four bounded advanced-RAG paths."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.core.state import EvidenceItem
from app.llm.base import LLMGateway, LLMMessage, ModelTier
from app.paths.diagnosis.advanced_rag import (
    ParallelAdvancedRAGRetriever,
    SelfRAGRetriever,
    build_advanced_rag_paths,
    reciprocal_rank_fuse,
)
from app.paths.diagnosis.retrieval import EvidenceQuery, Retriever


def _item(identifier: str) -> EvidenceItem:
    return EvidenceItem(
        source="PubMed",
        title=f"Study {identifier}",
        identifier=identifier,
        summary=f"Abstract for {identifier}",
    )


class RecordingRetriever(Retriever):
    source_name = "PubMed"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.active = 0
        self.max_active = 0

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        text = query.as_text()
        self.queries.append(text)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        if "variant" in text:
            return [_item("B"), _item("C")]
        if "repair" in text:
            return [_item("C")]
        if "refine" in text:
            return [_item("D")]
        if "follow" in text:
            return [_item("E")]
        return [_item("A"), _item("B")]


class ControllerGateway(LLMGateway):
    async def complete_structured(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> dict[str, Any]:
        if "query-expansion" in system:
            return {"queries": ["migraine variant"]}
        if "metacognitive" in system:
            return {
                "adequate": False,
                "relevant_identifiers": ["A"],
                "repair_queries": ["migraine repair"],
            }
        if "retrieval-on-demand" in system:
            return {"retrieve": True, "query": "migraine diagnosis"}
        if "self-critique" in system:
            return {
                "adequate": False,
                "relevant_identifiers": ["B"],
                "refinement_query": "migraine refine",
            }
        if "bounded follow-up" in system:
            return {
                "continue_retrieval": True,
                "follow_up_query": "migraine follow",
                "unresolved_information": "red flags",
            }
        raise AssertionError(system)

    async def complete_text(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> str:
        return ""

    async def stream_text(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None,
        model: ModelTier | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


class SkipGateway(ControllerGateway):
    async def complete_structured(self, **kwargs) -> dict[str, Any]:
        return {"retrieve": False, "query": "", "reason": "not needed"}


def test_builder_exposes_all_four_named_paths() -> None:
    paths = build_advanced_rag_paths(
        base=RecordingRetriever(), gateway=ControllerGateway(),
    )
    assert [path.source_name for path in paths] == [
        "RAG-Fusion", "MetaRAG", "Self-RAG", "i-MedRAG",
    ]


async def test_four_paths_execute_concurrently_and_return_bounded_evidence() -> None:
    base = RecordingRetriever()
    paths = build_advanced_rag_paths(base=base, gateway=ControllerGateway())
    retriever = ParallelAdvancedRAGRetriever(paths)

    items = await retriever.search(
        EvidenceQuery(query_string="migraine diagnosis", max_results=5)
    )

    assert base.max_active > 1, "peer RAG paths must overlap rather than run serially"
    assert len(items) <= 5
    assert {item.identifier for item in items}.issuperset({"A", "B"})
    assert any("RAG" in item.source for item in items)
    assert any("variant" in query for query in base.queries)
    assert any("repair" in query for query in base.queries)
    assert any("refine" in query for query in base.queries)
    assert any("follow" in query for query in base.queries)


async def test_self_rag_can_skip_retrieval_on_demand() -> None:
    base = RecordingRetriever()
    retriever = SelfRAGRetriever(base=base, gateway=SkipGateway())
    items = await retriever.search(EvidenceQuery(query_string="hello"))
    assert items == []
    assert base.queries == []


def test_reciprocal_rank_fusion_rewards_cross_query_hits() -> None:
    fused = reciprocal_rank_fuse(
        [[_item("A"), _item("B")], [_item("B"), _item("C")]], limit=3,
    )
    assert fused[0].identifier == "B"


if __name__ == "__main__":
    import inspect
    import traceback

    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            result = test()
            if inspect.isawaitable(result):
                asyncio.run(result)
        except Exception:
            failed += 1
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[PASS] {test.__name__}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


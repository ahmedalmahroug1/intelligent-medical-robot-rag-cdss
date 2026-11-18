"""Bounded, parallel advanced-RAG strategies for the diagnosis milestone.

The four strategies apply alternative retrieval and evidence-selection methods
while preserving the project's existing ``Retriever`` contract:

* RAG-Fusion expands one request into several focused queries and applies
  reciprocal-rank fusion.
* MetaRAG monitors the first retrieval, selects relevant hits, and plans one
  bounded repair search when coverage is inadequate.
* Self-RAG decides whether retrieval is useful, then critiques the returned
  evidence and optionally asks for one refinement search.
* i-MedRAG performs a bounded sequence of medically focused follow-up searches.

The production factory registers the four objects as peer retrievers.  The DAG
executor therefore starts them concurrently after the literature query plan is
ready, while test recommendation and medication-safety work continue on their
own branches.  All controller failures degrade to ordinary retrieval; no RAG
controller is allowed to sink the clinical milestone.

These classes retrieve evidence only. The independent curator converts useful
evidence into a structured, immediately retrievable local record labelled
``unapproved``. A clinician can later approve, reject, or delete that addition.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.core.state import EvidenceItem
from app.llm.base import LLMGateway, ModelTier
from app.paths.diagnosis.retrieval import EvidenceQuery, Retriever

logger = logging.getLogger(__name__)


class QueryVariants(BaseModel):
    """Focused variants used by RAG-Fusion (the original query is implicit)."""

    queries: list[str] = Field(default_factory=list, max_length=4)


class MetaRetrievalReview(BaseModel):
    """Metacognitive relevance/coverage decision over an initial result set."""

    adequate: bool = True
    relevant_identifiers: list[str] = Field(default_factory=list, max_length=12)
    repair_queries: list[str] = Field(default_factory=list, max_length=2)
    note: str = ""


class SelfRetrievalDecision(BaseModel):
    """Self-RAG retrieval-on-demand decision."""

    retrieve: bool = True
    query: str = ""
    reason: str = ""


class SelfRetrievalCritique(BaseModel):
    """Self-RAG critique after the first retrieval."""

    adequate: bool = True
    relevant_identifiers: list[str] = Field(default_factory=list, max_length=12)
    refinement_query: str = ""
    note: str = ""


class MedicalFollowUp(BaseModel):
    """One bounded i-MedRAG follow-up decision."""

    continue_retrieval: bool = False
    follow_up_query: str = ""
    unresolved_information: str = ""


_FUSION_PROMPT = """You are the query-expansion controller of a medical RAG system.
Given one PubMed query, return focused alternative queries that cover different
diagnostic facets without adding a diagnosis or unsupported patient fact. Keep
each query concise (roughly 2-5 clinical concepts), searchable, and different
from the original. Return only QueryVariants JSON."""

_META_PROMPT = """You are the metacognitive controller of a medical RAG system.
Review the supplied PubMed hits against the search objective. Mark identifiers
that are genuinely relevant to diagnosis or discrimination. If coverage is
inadequate, propose at most one concise repair query targeting the missing
evidence. Do not answer the clinical question and do not invent a citation.
Return only MetaRetrievalReview JSON."""

_SELF_DECISION_PROMPT = """You are the retrieval-on-demand controller of Self-RAG.
Decide whether external evidence is useful for this medical literature request.
Retrieve when the request concerns diagnosis, discrimination, red flags, tests,
or evidence that can change confidence. Skip only when retrieval plainly adds no
value. You may rewrite the query concisely. Do not answer the medical question.
Return only SelfRetrievalDecision JSON."""

_SELF_CRITIQUE_PROMPT = """You are the self-critique stage of Self-RAG. Check
whether the supplied hits adequately address the search objective and identify
the relevant identifiers. If they do not, provide at most one concise refinement
query. Do not create facts or citations. Return only SelfRetrievalCritique JSON."""

_IMED_PROMPT = """You are the bounded follow-up controller of i-MedRAG. Given a
medical search objective and evidence already retrieved, decide whether one more
search is needed to resolve an important diagnostic evidence gap. If so, return
one concise follow-up PubMed query. Do not answer the clinical question, invent
facts, or repeat an earlier query. Return only MedicalFollowUp JSON."""


def _clean_queries(values: Sequence[str], *, exclude: Sequence[str] = ()) -> list[str]:
    excluded = {" ".join(value.casefold().split()) for value in exclude if value.strip()}
    seen: set[str] = set(excluded)
    cleaned: list[str] = []
    for value in values:
        query = " ".join((value or "").split()).strip()
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
    return cleaned


def _as_query(text: str, template: EvidenceQuery) -> EvidenceQuery:
    return EvidenceQuery(query_string=text, max_results=template.max_results)


def _tag(item: EvidenceItem, path: str) -> EvidenceItem:
    source = item.source.strip() if item.source else "external evidence"
    label = f"{source} via {path}"
    return item.model_copy(update={"source": label})


def _evidence_digest(items: Sequence[EvidenceItem], *, limit: int = 8) -> str:
    if not items:
        return "(no hits)"
    lines: list[str] = []
    for item in items[:limit]:
        summary = " ".join((item.summary or "").split())[:360]
        lines.append(
            f"- [{item.identifier}] {item.title}; year={item.published_year or 'n/a'}; "
            f"type={item.study_type or 'n/a'}; abstract={summary or '(none)'}"
        )
    return "\n".join(lines)


def _selected_or_all(
    items: Sequence[EvidenceItem], identifiers: Sequence[str],
) -> list[EvidenceItem]:
    wanted = {value.strip().casefold() for value in identifiers if value.strip()}
    if not wanted:
        return list(items)
    selected = [item for item in items if item.identifier.strip().casefold() in wanted]
    # A malformed controller echo must never erase otherwise usable retrieval.
    return selected or list(items)


def reciprocal_rank_fuse(
    ranked_groups: Sequence[Sequence[EvidenceItem]],
    *,
    limit: int,
    rank_constant: int = 60,
) -> list[EvidenceItem]:
    """Deterministically fuse ranked result lists and merge path provenance."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    items: dict[str, EvidenceItem] = {}
    sources: dict[str, list[str]] = {}
    order = 0
    for group in ranked_groups:
        seen_in_group: set[str] = set()
        for rank, item in enumerate(group, start=1):
            key = item.identifier.strip().casefold() or item.title.strip().casefold()
            if not key or key in seen_in_group:
                continue
            seen_in_group.add(key)
            if key not in first_seen:
                first_seen[key] = order
                order += 1
                items[key] = item
                sources[key] = []
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
            if item.source and item.source not in sources[key]:
                sources[key].append(item.source)

    ranked_keys = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    fused: list[EvidenceItem] = []
    for key in ranked_keys[: max(0, limit)]:
        item = items[key]
        merged_source = " | ".join(sources[key]) or item.source
        fused.append(item.model_copy(update={"source": merged_source}))
    return fused


class _AdvancedRetriever(Retriever):
    def __init__(
        self,
        *,
        base: Retriever,
        gateway: LLMGateway,
        model: ModelTier = ModelTier.FAST,
    ) -> None:
        self._base = base
        self._gateway = gateway
        self._model = model

    async def _controller(self, schema, *, system: str, user: str):
        try:
            return await self._gateway.complete_typed(
                schema, system=system, user=user, model=self._model,
            )
        except Exception as exc:  # noqa: BLE001 - controller failure degrades safely
            logger.warning("%s controller unavailable: %s", self.source_name, exc)
            return None

    async def _search_many(
        self, queries: Sequence[EvidenceQuery],
    ) -> tuple[list[list[EvidenceItem]], list[BaseException]]:
        outcomes = await asyncio.gather(
            *(self._base.search(query) for query in queries), return_exceptions=True,
        )
        groups: list[list[EvidenceItem]] = []
        errors: list[BaseException] = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                errors.append(outcome)
            else:
                groups.append([_tag(item, self.source_name) for item in outcome])
        return groups, errors

    @staticmethod
    def _raise_if_total_failure(
        groups: Sequence[Sequence[EvidenceItem]], errors: Sequence[BaseException],
    ) -> None:
        if errors and not groups:
            raise errors[0]


class RAGFusionRetriever(_AdvancedRetriever):
    """Multi-query expansion followed by reciprocal-rank fusion."""

    source_name = "RAG-Fusion"

    def __init__(self, *, query_count: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self._query_count = max(2, min(5, int(query_count)))

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        original = query.as_text()
        variants = await self._controller(
            QueryVariants,
            system=_FUSION_PROMPT,
            user=f"Original search objective:\n{original}",
        )
        alternatives = _clean_queries(
            variants.queries if variants is not None else [], exclude=[original],
        )[: self._query_count - 1]
        queries = [query, *(_as_query(value, query) for value in alternatives)]
        groups, errors = await self._search_many(queries)
        self._raise_if_total_failure(groups, errors)
        return reciprocal_rank_fuse(groups, limit=query.max_results)


class MetaRAGRetriever(_AdvancedRetriever):
    """Retrieve, monitor relevance/coverage, then run one repair plan."""

    source_name = "MetaRAG"

    def __init__(self, *, max_repair_queries: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_repair_queries = max(0, min(2, int(max_repair_queries)))

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        initial_groups, errors = await self._search_many([query])
        self._raise_if_total_failure(initial_groups, errors)
        initial = initial_groups[0] if initial_groups else []
        review = await self._controller(
            MetaRetrievalReview,
            system=_META_PROMPT,
            user=(
                f"Search objective:\n{query.as_text()}\n\nRetrieved evidence:\n"
                f"{_evidence_digest(initial)}"
            ),
        )
        if review is None:
            return initial[: query.max_results]

        selected = _selected_or_all(initial, review.relevant_identifiers)
        repair_texts = [] if review.adequate else _clean_queries(
            review.repair_queries, exclude=[query.as_text()],
        )[: self._max_repair_queries]
        repair_groups: list[list[EvidenceItem]] = []
        if repair_texts:
            repair_groups, _ = await self._search_many(
                [_as_query(value, query) for value in repair_texts]
            )
        return reciprocal_rank_fuse(
            [selected, *repair_groups], limit=query.max_results,
        )


class SelfRAGRetriever(_AdvancedRetriever):
    """Retrieval on demand plus one bounded self-critique/refinement."""

    source_name = "Self-RAG"

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        decision = await self._controller(
            SelfRetrievalDecision,
            system=_SELF_DECISION_PROMPT,
            user=f"Medical search request:\n{query.as_text()}",
        )
        if decision is not None and not decision.retrieve:
            return []
        target_text = (
            " ".join(decision.query.split())
            if decision is not None and decision.query.strip()
            else query.as_text()
        )
        target = _as_query(target_text, query)
        initial_groups, errors = await self._search_many([target])
        self._raise_if_total_failure(initial_groups, errors)
        initial = initial_groups[0] if initial_groups else []

        critique = await self._controller(
            SelfRetrievalCritique,
            system=_SELF_CRITIQUE_PROMPT,
            user=(
                f"Search objective:\n{query.as_text()}\n\nRetrieved evidence:\n"
                f"{_evidence_digest(initial)}"
            ),
        )
        if critique is None:
            return initial[: query.max_results]
        selected = _selected_or_all(initial, critique.relevant_identifiers)
        refinement = _clean_queries(
            [critique.refinement_query] if not critique.adequate else [],
            exclude=[query.as_text(), target_text],
        )
        refinement_groups: list[list[EvidenceItem]] = []
        if refinement:
            refinement_groups, _ = await self._search_many(
                [_as_query(refinement[0], query)]
            )
        return reciprocal_rank_fuse(
            [selected, *refinement_groups], limit=query.max_results,
        )


class IMedRAGRetriever(_AdvancedRetriever):
    """Bounded iterative medical follow-up retrieval."""

    source_name = "i-MedRAG"

    def __init__(self, *, max_iterations: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_iterations = max(1, min(3, int(max_iterations)))

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        current = query
        used = [query.as_text()]
        groups: list[list[EvidenceItem]] = []
        errors: list[BaseException] = []
        for iteration in range(self._max_iterations):
            found, failed = await self._search_many([current])
            groups.extend(found)
            errors.extend(failed)
            if iteration >= self._max_iterations - 1:
                break
            accumulated = reciprocal_rank_fuse(groups, limit=query.max_results)
            follow_up = await self._controller(
                MedicalFollowUp,
                system=_IMED_PROMPT,
                user=(
                    f"Original objective:\n{query.as_text()}\n\nEarlier queries:\n"
                    + "\n".join(f"- {value}" for value in used)
                    + "\n\nEvidence so far:\n"
                    + _evidence_digest(accumulated)
                ),
            )
            if follow_up is None or not follow_up.continue_retrieval:
                break
            next_queries = _clean_queries([follow_up.follow_up_query], exclude=used)
            if not next_queries:
                break
            used.append(next_queries[0])
            current = _as_query(next_queries[0], query)
        self._raise_if_total_failure(groups, errors)
        return reciprocal_rank_fuse(groups, limit=query.max_results)


class ParallelAdvancedRAGRetriever(Retriever):
    """Aggregate all four peers for background curator research."""

    source_name = "Advanced-RAG"

    def __init__(self, paths: Sequence[Retriever]) -> None:
        self._paths = list(paths)

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        outcomes = await asyncio.gather(
            *(path.search(query) for path in self._paths), return_exceptions=True,
        )
        groups: list[list[EvidenceItem]] = []
        errors: list[BaseException] = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                errors.append(outcome)
            else:
                groups.append(outcome)
        if errors and not groups:
            raise errors[0]
        return reciprocal_rank_fuse(groups, limit=query.max_results)


def build_advanced_rag_paths(
    *,
    base: Retriever,
    gateway: LLMGateway,
    fusion_query_count: int = 3,
    meta_repair_queries: int = 1,
    imed_iterations: int = 2,
) -> list[Retriever]:
    """Build the four peer paths over one shared, rate-limited evidence source."""
    common = {"base": base, "gateway": gateway}
    return [
        RAGFusionRetriever(query_count=fusion_query_count, **common),
        MetaRAGRetriever(max_repair_queries=meta_repair_queries, **common),
        SelfRAGRetriever(**common),
        IMedRAGRetriever(max_iterations=imed_iterations, **common),
    ]

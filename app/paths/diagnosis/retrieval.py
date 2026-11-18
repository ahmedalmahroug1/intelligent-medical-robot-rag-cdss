"""
app/paths/diagnosis/retrieval.py
--------------------------------
The evidence-retrieval seam for the milestone's RAG step.

`Retriever` is the interface every evidence source implements (PubMed, clinical
guidelines, and — later — the image-analysis path's findings). The milestone runs
the available retrievers IN PARALLEL through the executor, then a verification
step ranks whatever came back. Putting retrieval behind this interface is what
lets the milestone be tested fully offline (with the fakes below) and lets the
real, network-bound PubMed client drop in later as just another implementation —
the milestone graph does not change.

Each retriever returns domain `EvidenceItem`s with only the descriptive fields
filled; the quality_score / quality_notes are left empty and set by the
verification step, never by the retriever itself.

Not built here (deliberately): the concrete `PubMedRetriever` (httpx against the
NCBI E-utilities API). It is a straightforward implementation of this interface
and is the next concrete piece; importantly, it must be properly async so the
executor's per-task timeout can actually cancel a slow request (see File 7's
note on cancellation-safety).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.core.state import EvidenceItem


@dataclass(frozen=True)
class EvidenceQuery:
    """What to search for.

    Two ways to express the search, in precedence order:
      - `query_string`: a pre-built boolean query produced by the query-planner
        agent (e.g. a targeted PubMed expression). When set, it is used verbatim.
      - `terms`: a few condition names drawn from the differential, OR-ed together.
        The fallback when no planner query is available.
    """
    terms: list[str] = field(default_factory=list)
    query_string: str = ""
    max_results: int = 5

    def as_text(self) -> str:
        if self.query_string.strip():
            return self.query_string.strip()
        return " OR ".join(t for t in self.terms if t) or "general"


class Retriever(ABC):
    """One evidence source. Implementations must be async and cancellation-safe."""

    #: Short label used to name this source's executor node and to tag its items.
    source_name: str

    @abstractmethod
    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        """Return matching evidence items (quality fields left unset). Returning
        an empty list is fine and means "no evidence found"; raising signals a
        retrieval failure, which the executor records without killing the run."""
        ...


# ── Offline fakes (for tests and for running the spine without network) ───────

class StaticRetriever(Retriever):
    """Returns a fixed list of items, ignoring the query. For tests/demos."""

    def __init__(self, source_name: str, items: list[EvidenceItem]) -> None:
        self.source_name = source_name
        self._items = list(items)

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        return list(self._items)


class NullRetriever(Retriever):
    """Always returns nothing — a stand-in for a source that is configured but
    has no data yet (e.g. the guideline corpus before it is populated)."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        return []

"""
app/paths/diagnosis/pubmed.py
-----------------------------
The real evidence source: a PubMed retriever over the NCBI E-utilities API,
implementing the `Retriever` interface. This is the concrete RAG source the
milestone runs in parallel; the verification node ranks what it returns.

It is isolated in its own module (the only one importing `httpx`) for the same
reason the OpenAI gateway is the only module importing `openai`: the dependency
stays at the edge, and the rest of the diagnosis package imports cleanly without
it. It is deliberately NOT re-exported from the package __init__ — callers that
want live retrieval import it explicitly (the API wiring and the eval runner do).

Flow (two requests per search):
    1. esearch — find PMIDs matching the condition terms.
    2. efetch  — fetch those records as XML and parse out title, abstract, year,
                 and publication type for each.

Cancellation-safety: it uses httpx.AsyncClient and awaits network I/O, so the
executor's per-task timeout actually cancels an in-flight request and frees the
connection — the requirement flagged when the executor was built.

Failure handling: on no matches it returns [] (a normal "no evidence" result);
on an HTTP, network, or parse failure it raises, which the executor records as a
failed node without stopping the milestone (ranking proceeds on other sources).

NCBI etiquette: requests carry `tool` and `email` identifiers and an optional
`api_key` (which raises the rate limit from 3 to 10 req/sec). Usage here is light
(two requests per consultation), so no explicit throttling is added.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from xml.etree import ElementTree as ET

import httpx

from app.config import settings
from app.core.state import EvidenceItem
from app.paths.diagnosis.retrieval import EvidenceQuery, Retriever

logger = logging.getLogger(__name__)


class PubMedError(RuntimeError):
    """A PubMed retrieval or parse failure (distinct from 'no results found')."""


# Publication types in rough descending order of evidence strength; used to pick a
# single representative study_type for an article from its (possibly several) types.
_STUDY_PRIORITY = (
    "Meta-Analysis",
    "Systematic Review",
    "Practice Guideline",
    "Guideline",
    "Randomized Controlled Trial",
    "Review",
    "Clinical Trial",
    "Comparative Study",
    "Observational Study",
)


class PubMedRetriever(Retriever):
    """Retrieves evidence from PubMed via the NCBI E-utilities API."""

    source_name = "PubMed"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        email: str | None = None,
        tool: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_results: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else (settings.NCBI_API_KEY or None)
        self._email = email if email is not None else (settings.NCBI_EMAIL or None)
        self._tool = tool or settings.NCBI_TOOL
        self._base = (base_url or settings.NCBI_EUTILS_BASE).rstrip("/")
        self._timeout = settings.PUBMED_TIMEOUT_S if timeout is None else timeout
        self._max_results = settings.PUBMED_MAX_RESULTS if max_results is None else max_results
        # NCBI permits roughly 3 requests/sec without an API key and 10/sec with
        # one. A single search performs esearch + efetch, and the curator can run
        # several planned searches, so throttle starts across the shared instance.
        self._minimum_interval = 0.11 if self._api_key else 0.36
        self._request_lock = asyncio.Lock()
        self._last_request_at = 0.0
        # An injected client is reused across calls (efficient for the eval, and
        # lets tests pass an httpx.MockTransport). Otherwise one is opened per
        # search and closed via `async with`.
        self._client = client

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        term = self._build_term(query)
        if not term:
            return []
        if self._client is not None:
            return await self._run(self._client, term, query.max_results)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._run(client, term, query.max_results)

    # ── Pipeline ───────────────────────────────────────────────────────────────

    async def _run(self, client: httpx.AsyncClient, term: str, max_results: int) -> list[EvidenceItem]:
        limit = min(max_results or self._max_results, self._max_results)
        pmids = await self._esearch(client, term, limit)
        if not pmids:
            return []
        xml = await self._efetch(client, pmids)
        return self._parse_articles(xml)

    async def _esearch(self, client: httpx.AsyncClient, term: str, limit: int) -> list[str]:
        params = self._params({
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": str(limit),
            "sort": "relevance",
        })
        try:
            resp = await self._get(client, f"{self._base}/esearch.fcgi", params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise PubMedError(f"PubMed esearch failed: {exc}") from exc
        return list(data.get("esearchresult", {}).get("idlist", []))

    async def _efetch(self, client: httpx.AsyncClient, pmids: list[str]) -> str:
        params = self._params({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        })
        try:
            resp = await self._get(client, f"{self._base}/efetch.fcgi", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise PubMedError(f"PubMed efetch failed: {exc}") from exc
        return resp.text

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_articles(self, xml: str) -> list[EvidenceItem]:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as exc:
            raise PubMedError(f"PubMed returned unparseable XML: {exc}") from exc

        items: list[EvidenceItem] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = _first_text(article, ".//MedlineCitation/PMID")
            if not pmid:
                continue
            title = _joined_text(article.find(".//Article/ArticleTitle"))
            abstract = " ".join(
                _joined_text(node)
                for node in article.findall(".//Abstract/AbstractText")
            ).strip()
            year = self._extract_year(article)
            study_type = self._pick_study_type(
                [t.text for t in article.findall(".//PublicationType") if t.text]
            )
            items.append(EvidenceItem(
                source=self.source_name,
                title=title or "(untitled)",
                identifier=f"PMID:{pmid}",
                summary=abstract,
                published_year=year,
                study_type=study_type,
            ))
        return items

    def _extract_year(self, article: ET.Element) -> int | None:
        year = _first_text(article, ".//Article/Journal/JournalIssue/PubDate/Year")
        if year and year.isdigit():
            return int(year)
        medline = _first_text(article, ".//Article/Journal/JournalIssue/PubDate/MedlineDate")
        if medline:
            match = re.search(r"\b(19|20)\d{2}\b", medline)
            if match:
                return int(match.group())
        return None

    def _pick_study_type(self, types: list[str]) -> str | None:
        lowered = {t.lower(): t for t in types if t}
        for preferred in _STUDY_PRIORITY:
            if preferred.lower() in lowered:
                return lowered[preferred.lower()]
        for t in types:
            if t and t.lower() != "journal article":
                return t
        return types[0] if types else None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_term(self, query: EvidenceQuery) -> str:
        # A planned query (from the query-planner agent) is a complete boolean
        # expression — use it verbatim. Otherwise fall back to OR-ing the
        # condition names so any matching article is retrieved.
        if query.query_string and query.query_string.strip():
            return query.query_string.strip()
        terms = [t.strip() for t in query.terms if t and t.strip()]
        if not terms:
            return ""
        return " OR ".join(f"({t})" for t in terms)

    def _params(self, base: dict[str, str]) -> dict[str, str]:
        params = dict(base)
        params["tool"] = self._tool
        if self._email:
            params["email"] = self._email
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    async def _get(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, str],
    ) -> httpx.Response:
        async with self._request_lock:
            for attempt in range(2):
                remaining = self._minimum_interval - (time.monotonic() - self._last_request_at)
                if remaining > 0:
                    await asyncio.sleep(remaining)
                response = await client.get(url, params=params)
                self._last_request_at = time.monotonic()
                if response.status_code != 429 or attempt == 1:
                    return response
                retry_after = response.headers.get("retry-after")
                try:
                    delay = float(retry_after) if retry_after else 1.0
                except ValueError:
                    delay = 1.0
                await asyncio.sleep(max(delay, self._minimum_interval))
        raise RuntimeError("unreachable PubMed request state")


def _first_text(element: ET.Element, path: str) -> str | None:
    found = element.find(path)
    if found is None:
        return None
    text = "".join(found.itertext()).strip()
    return text or None


def _joined_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()

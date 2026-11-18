"""
tests/test_pubmed.py
--------------------
Tests for the PubMedRetriever. Run from the project root:

    python -m tests.test_pubmed

Needs httpx (your env has it). No network — requests are served by an
httpx.MockTransport returning canned NCBI esearch/efetch responses, so this
verifies the query/parse logic deterministically.
"""

from __future__ import annotations

import asyncio
import inspect
import traceback

import httpx

from app.paths.diagnosis.pubmed import PubMedError, PubMedRetriever
from app.paths.diagnosis.retrieval import EvidenceQuery


_ESEARCH = {"esearchresult": {"idlist": ["111", "222"]}}

_EFETCH = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">111</PMID>
      <Article>
        <Journal><JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Migraine treatment review</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background text.</AbstractText>
          <AbstractText Label="RESULTS">Results text.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
          <PublicationType>Review</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article>
        <Journal><JournalIssue><PubDate><MedlineDate>2020 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
        <ArticleTitle>RCT of <i>sumatriptan</i> efficacy</ArticleTitle>
        <Abstract><AbstractText>Single abstract.</AbstractText></Abstract>
        <PublicationTypeList>
          <PublicationType>Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok_handler(request: httpx.Request) -> httpx.Response:
    if "esearch" in request.url.path:
        return httpx.Response(200, json=_ESEARCH)
    if "efetch" in request.url.path:
        return httpx.Response(200, text=_EFETCH)
    return httpx.Response(404)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_parses_articles() -> None:
    client = _client(_ok_handler)
    try:
        retriever = PubMedRetriever(client=client)
        items = await retriever.search(EvidenceQuery(terms=["migraine"]))

        assert len(items) == 2
        first, second = items
        assert first.identifier == "PMID:111"
        assert first.source == "PubMed"
        assert first.title == "Migraine treatment review"
        assert "Background text." in first.summary and "Results text." in first.summary
        assert first.published_year == 2019
        assert first.study_type == "Review"  # chosen over 'Journal Article'

        assert second.identifier == "PMID:222"
        assert "sumatriptan" in second.title  # itertext captures nested <i> text
        assert second.published_year == 2020  # parsed from MedlineDate
        assert second.study_type == "Randomized Controlled Trial"
    finally:
        await client.aclose()


async def test_empty_search_returns_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "esearch" in request.url.path:
            return httpx.Response(200, json={"esearchresult": {"idlist": []}})
        return httpx.Response(200, text="<PubmedArticleSet/>")

    client = _client(handler)
    try:
        items = await PubMedRetriever(client=client).search(EvidenceQuery(terms=["zzzznope"]))
        assert items == []
    finally:
        await client.aclose()


async def test_http_error_raises_pubmed_error() -> None:
    client = _client(lambda request: httpx.Response(500, text="server error"))
    try:
        try:
            await PubMedRetriever(client=client).search(EvidenceQuery(terms=["x"]))
            raise AssertionError("expected PubMedError on HTTP 500")
        except PubMedError:
            pass
    finally:
        await client.aclose()


async def test_no_terms_makes_no_request() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})

    client = _client(handler)
    try:
        items = await PubMedRetriever(client=client).search(EvidenceQuery(terms=[]))
        assert items == []
        assert calls["n"] == 0, "no terms should short-circuit before any HTTP call"
    finally:
        await client.aclose()


async def test_query_term_ORs_conditions() -> None:
    seen = {"term": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if "esearch" in request.url.path:
            seen["term"] = request.url.params.get("term")
            return httpx.Response(200, json={"esearchresult": {"idlist": []}})
        return httpx.Response(200, text="<PubmedArticleSet/>")

    client = _client(handler)
    try:
        await PubMedRetriever(client=client).search(
            EvidenceQuery(terms=["migraine", "tension headache"])
        )
        assert seen["term"] == "(migraine) OR (tension headache)", seen["term"]
    finally:
        await client.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_parses_articles,
    test_empty_search_returns_empty,
    test_http_error_raises_pubmed_error,
    test_no_terms_makes_no_request,
    test_query_term_ORs_conditions,
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

"""Regulatory medication-safety evidence retrieved from openFDA labels."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable

import httpx

from app.config import settings
from app.knowledge.models import KnowledgeSource


def _slug(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", value.casefold()))[:50] or "drug"


class OpenFDALabelRetriever:
    """Fetch one current US prescribing-label record per generic medicine.

    The label is safety evidence, not a local formulary and not permission to
    prescribe.  Low-volume requests work without an API key; a key is optional.
    """

    source_name = "openFDA drug labels"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url or settings.OPENFDA_LABEL_BASE
        self._api_key = api_key if api_key is not None else (settings.OPENFDA_API_KEY or None)
        self._timeout = settings.OPENFDA_TIMEOUT_S if timeout is None else timeout
        self._client = client

    async def search_many(self, terms: Iterable[str]) -> dict[str, KnowledgeSource | None]:
        unique = list(dict.fromkeys(term.strip() for term in terms if term.strip()))[:5]
        semaphore = asyncio.Semaphore(3)

        async def one(term: str) -> tuple[str, KnowledgeSource | None]:
            async with semaphore:
                return term, await self.search(term)

        return dict(await asyncio.gather(*(one(term) for term in unique)))

    async def search(self, generic_name: str) -> KnowledgeSource | None:
        if self._client is not None:
            return await self._run(self._client, generic_name)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._run(client, generic_name)

    async def _run(
        self, client: httpx.AsyncClient, generic_name: str,
    ) -> KnowledgeSource | None:
        escaped = generic_name.replace('"', "")
        # Exact matching often misses salt forms (for example SUMATRIPTAN
        # SUCCINATE). Fall back to a field search, still constrained to the
        # generic-name field, and prefer the most recent effective label.
        searches = (
            f'openfda.generic_name.exact:"{escaped}"',
            f'openfda.generic_name:"{escaped}"',
        )
        for search in searches:
            params: dict[str, str] = {
                "search": search,
                "limit": "1",
                "sort": "effective_time:desc",
            }
            if self._api_key:
                params["api_key"] = self._api_key
            response = await client.get(self._base_url, params=params)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            results = response.json().get("results", [])
            if results:
                return self._to_source(generic_name, results[0])
        return None

    @staticmethod
    def _to_source(generic_name: str, label: dict) -> KnowledgeSource:
        openfda = label.get("openfda", {})
        set_ids = openfda.get("spl_set_id", [])
        set_id = set_ids[0] if set_ids else ""
        application = (openfda.get("application_number") or ["unknown"])[0]
        brand = (openfda.get("brand_name") or [generic_name])[0]
        generic = (openfda.get("generic_name") or [generic_name])[0]

        sections: list[str] = []
        mapping = (
            ("CONTRAINDICATIONS", "contraindications"),
            ("BOXED WARNING", "boxed_warning"),
            ("WARNINGS AND CAUTIONS", "warnings_and_cautions"),
            ("WARNINGS", "warnings"),
            ("DRUG INTERACTIONS", "drug_interactions"),
            ("PREGNANCY", "pregnancy"),
            ("INDICATIONS", "indications_and_usage"),
            ("PEDIATRIC USE", "pediatric_use"),
            ("GERIATRIC USE", "geriatric_use"),
            ("ADVERSE REACTIONS", "adverse_reactions"),
        )
        for heading, key in mapping:
            values = label.get(key) or []
            if values:
                text = " ".join(str(value) for value in values)
                sections.append(f"{heading}: {text[:850]}")
        summary = "\n".join(sections)[:3500]
        if not summary:
            summary = "The returned label did not expose the requested safety sections."

        url = (
            f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
            if set_id else "https://open.fda.gov/apis/drug/label/"
        )
        suffix = re.sub(r"[^a-zA-Z0-9]", "", set_id)[:10] or _slug(application)
        return KnowledgeSource(
            source_id=f"FDA_{_slug(generic_name).upper()}_{suffix.upper()}",
            source_type="Regulatory drug label via openFDA",
            title=f"{brand} ({generic}) prescribing label",
            identifier=f"SPL:{set_id}" if set_id else f"FDA:{application}",
            url=url,
            study_type="Regulatory prescribing information",
            summary=summary,
            evidence_scope=["treatment", "medication_safety"],
        )


class StaticDrugLabelRetriever:
    """Deterministic test double for the staged curator."""

    source_name = "Static drug labels"

    def __init__(self, labels: dict[str, KnowledgeSource] | None = None) -> None:
        self.labels = {key.casefold(): value for key, value in (labels or {}).items()}

    async def search_many(self, terms: Iterable[str]) -> dict[str, KnowledgeSource | None]:
        return {term: self.labels.get(term.casefold()) for term in terms}

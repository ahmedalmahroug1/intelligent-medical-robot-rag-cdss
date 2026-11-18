"""Offline checks for official-guideline and regulatory-label evidence tools."""

from __future__ import annotations

import asyncio

import httpx

from app.knowledge.drug_labels import OpenFDALabelRetriever
from app.knowledge.guidelines import OfficialGuidelineCatalog


async def _guidelines() -> None:
    catalog = OfficialGuidelineCatalog()
    migraine = await catalog.search("Migraine")
    knee_oa = await catalog.search("Knee osteoarthritis")
    unknown = await catalog.search("Imaginary example disorder")
    assert {source.source_id for source in migraine} >= {
        "ICHD3_MIGRAINE_WITHOUT_AURA", "NICE_CG150_HEADACHE",
    }
    assert {source.source_id for source in knee_oa} >= {
        "NICE_NG226_OA_DIAGNOSIS", "NICE_NG226_OA_MANAGEMENT",
    }
    assert all(source.url.startswith("https://") for source in [*migraine, *knee_oa])
    assert unknown == []


def test_official_guideline_catalog_is_condition_scoped() -> None:
    asyncio.run(_guidelines())


async def _openfda() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"] == 'openfda.generic_name.exact:"sumatriptan"'
        return httpx.Response(200, json={
            "results": [{
                "openfda": {
                    "generic_name": ["SUMATRIPTAN"],
                    "brand_name": ["Example brand"],
                    "spl_set_id": ["abc-123"],
                    "application_number": ["NDA000001"],
                },
                "indications_and_usage": ["Indicated for an approved use."],
                "contraindications": ["Contraindicated in example high-risk history."],
                "warnings_and_cautions": ["Review interacting medicines."],
                "pregnancy": ["Use requires an individual risk assessment."],
            }],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        retriever = OpenFDALabelRetriever(client=client)
        source = await retriever.search("sumatriptan")
    assert source is not None
    assert source.source_type == "Regulatory drug label via openFDA"
    assert source.url.endswith("setid=abc-123")
    assert "CONTRAINDICATIONS" in source.summary
    assert "medication_safety" in source.evidence_scope


def test_openfda_label_becomes_safety_evidence() -> None:
    asyncio.run(_openfda())

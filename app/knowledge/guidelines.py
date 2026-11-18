"""Curated official-guideline retrieval for the dynamic knowledge agent.

An LLM must not invent a guideline URL.  This retriever therefore searches a
small, version-controlled catalogue of sources whose URLs and summaries were
reviewed by a human.  The catalogue is deliberately extensible; lack of a match
is recorded as a research gap and makes the generated draft non-approvable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import settings
from app.knowledge.models import KnowledgeSource


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


class OfficialGuidelineCatalog:
    source_name = "Official guideline catalogue"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or settings.KNOWLEDGE_GUIDELINE_CATALOG)

    async def search(self, condition: str, topics: list[str] | None = None) -> list[KnowledgeSource]:
        if not self.path.exists():
            return []
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

        condition_tokens = _tokens(condition)
        topic_tokens = _tokens(" ".join(topics or []))
        ranked: list[tuple[int, KnowledgeSource]] = []
        for row in rows:
            aliases = " ".join(row.get("conditions", []))
            alias_tokens = _tokens(aliases)
            overlap = len(condition_tokens & alias_tokens)
            topic_overlap = len(topic_tokens & alias_tokens)
            exact = any(condition.casefold().strip() == alias.casefold().strip()
                        for alias in row.get("conditions", []))
            if not exact and overlap == 0:
                continue
            source = KnowledgeSource.model_validate(row["source"])
            ranked.append((overlap + topic_overlap + (100 if exact else 0), source))
        ranked.sort(key=lambda item: (-item[0], item[1].source_id))
        return [source for _, source in ranked]

    async def medication_terms(self, condition: str) -> list[str]:
        """Return human-reviewed generic label-search leads for matched guidance."""
        if not self.path.exists():
            return []
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        condition_tokens = _tokens(condition)
        terms: list[str] = []
        for row in rows:
            aliases = row.get("conditions", [])
            alias_tokens = _tokens(" ".join(aliases))
            exact = any(condition.casefold().strip() == alias.casefold().strip()
                        for alias in aliases)
            if not exact and not (condition_tokens & alias_tokens):
                continue
            terms.extend(row.get("medication_search_terms", []))
        return list(dict.fromkeys(term.strip() for term in terms if term.strip()))

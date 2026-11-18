"""
app/paths/lab_review/schemas.py
-------------------------------
The LAB_REVIEW path is deterministic. The ONLY structured LLM use is an optional
FALLBACK parser, used when `lab_reference.parse_labs` cannot extract any values from
messy free text. The model only proposes name->value pairs; those are then re-validated
by `parse_labs`, so the LLM never performs the clinical interpretation.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class LabParse(BaseModel):
    """Fallback name->value extraction. All defaults, so an empty/failed parse is safe."""
    values: dict[str, float] = Field(default_factory=dict)

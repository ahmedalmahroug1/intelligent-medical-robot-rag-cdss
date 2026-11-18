"""
app/paths/diagnosis/reasoning.py
--------------------------------
The SINGLE diagnosis-reasoning call, reused by both:
  * the DiagnosisPath interview loop (`path.py`), over an interview transcript, and
  * the concierge's `ReasoningFacade`, over a structured clinical picture.

Both pass the SAME system prompt (the DiagnosisPath's `_SYSTEM_PROMPT`) and parse into the SAME
`IntakeResult`, so there is exactly ONE differential/next-question/sufficiency prompt in the
system — no duplicate, no divergence. Only the `context` (transcript vs picture) differs by caller.
"""
from __future__ import annotations

from app.llm.base import LLMGateway, ModelTier
from app.paths.diagnosis.schemas import IntakeResult


async def reason_over_context(
    gateway: LLMGateway, *, system_prompt: str, context: str,
    model: ModelTier | None = None,
) -> IntakeResult:
    """Run the diagnosis reasoning (differential + single next question + sufficiency) over an
    arbitrary case context and return the parsed `IntakeResult`."""
    return await gateway.complete_typed(
        IntakeResult, system=system_prompt, user=context, model=model)

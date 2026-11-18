"""Production diagnosis adapters that read review-labelled local KB records."""

from __future__ import annotations

from app.concierge.reasoning import ReasoningFacade
from app.concierge.schemas import ClinicalPicture
from app.core.state import ConsultationState
from app.knowledge.retrieval import ClinicalKnowledgeBase
from app.paths.diagnosis import DiagnosisPath, Milestone


class KnowledgeGuidedDiagnosisPath(DiagnosisPath):
    def __init__(self, *args, knowledge: ClinicalKnowledgeBase, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._clinical_knowledge = knowledge

    def _build_user_prompt(self, turn: str, state: ConsultationState) -> str:
        base = super()._build_user_prompt(turn, state)
        guidance = self._clinical_knowledge.guidance(
            self._clinical_knowledge.query_from_state(state, turn)
        )
        return base + "\n\nDYNAMIC LOCAL KNOWLEDGE-BASE READING:\n" + guidance


class KnowledgeGuidedMilestone(Milestone):
    def __init__(self, *args, knowledge: ClinicalKnowledgeBase, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._clinical_knowledge = knowledge

    def _differential_user(self, state: ConsultationState) -> str:
        base = super()._differential_user(state)
        guidance = self._clinical_knowledge.guidance(
            self._clinical_knowledge.query_from_state(state)
        )
        return base + "\n\nDYNAMIC LOCAL KNOWLEDGE-BASE READING:\n" + guidance


class KnowledgeGuidedReasoningFacade(ReasoningFacade):
    """Concierge facade that reads the same local KB at interview and milestone stages."""

    def __init__(
        self,
        *args,
        knowledge: ClinicalKnowledgeBase,
        milestone: KnowledgeGuidedMilestone,
        **kwargs,
    ) -> None:
        super().__init__(*args, milestone=milestone, **kwargs)
        self._clinical_knowledge = knowledge

    def _assess_context(self, picture: ClinicalPicture) -> str:
        base = super()._assess_context(picture)
        query = "\n".join(
            str(value)
            for value in picture.model_dump(exclude={"language"}).values()
            if value not in (None, "", [], {})
        )
        guidance = self._clinical_knowledge.guidance(query)
        return base + "\n\nDYNAMIC LOCAL KNOWLEDGE-BASE READING:\n" + guidance

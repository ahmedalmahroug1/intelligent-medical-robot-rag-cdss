"""Offline integration checks for knowledge-guided Concierge wiring."""
from app.concierge.schemas import ClinicalPicture
from app.knowledge.guided import KnowledgeGuidedMilestone, KnowledgeGuidedReasoningFacade
from tests.fakes import FakeGateway


class StubKnowledge:
    def guidance(self, query: str) -> str:
        assert "headache" in query
        return "UNAPPROVED RECORD: migraine discriminators"


def test_concierge_facade_reads_review_labelled_knowledge_during_interview() -> None:
    gateway = FakeGateway()
    knowledge = StubKnowledge()
    milestone = KnowledgeGuidedMilestone(
        gateway,
        knowledge=knowledge,
        retrievers=[],
    )
    facade = KnowledgeGuidedReasoningFacade(
        gateway,
        knowledge=knowledge,
        milestone=milestone,
    )
    context = facade._assess_context(ClinicalPicture(
        chief_complaint="headache",
        symptoms=["photophobia"],
        duration="two days",
    ))
    assert "DYNAMIC LOCAL KNOWLEDGE-BASE READING" in context
    assert "UNAPPROVED RECORD: migraine discriminators" in context

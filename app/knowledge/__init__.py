"""Dynamic clinical knowledge with explicit clinician-review status.

The operational knowledge base has three deliberately separate states:

* ``data/kb100/diseases`` contains the hand-built seed records;
* ``data/kb100/pending`` contains live machine-authored additions labelled unapproved;
* ``data/kb100/approved_dynamic`` contains additions signed by a clinician.

The retriever reads valid pending records immediately and exposes their review
status in its prompt. Rejection or deletion removes them from retrieval;
approval replaces the provisional copy with a clinician-signed one.
"""

from app.knowledge.agent import KnowledgeCuratorAgent
from app.knowledge.factory import (
    build_diagnosis_path,
    build_knowledge_agent,
    build_reasoning_facade,
)
from app.knowledge.guided import (
    KnowledgeGuidedDiagnosisPath,
    KnowledgeGuidedMilestone,
    KnowledgeGuidedReasoningFacade,
)
from app.knowledge.models import (
    DynamicDiseaseRecord,
    KnowledgeProposal,
    KnowledgeProposalStatus,
)
from app.knowledge.repository import KnowledgeProposalRepository
from app.knowledge.retrieval import ClinicalKnowledgeBase

__all__ = [
    "ClinicalKnowledgeBase",
    "DynamicDiseaseRecord",
    "KnowledgeCuratorAgent",
    "KnowledgeGuidedDiagnosisPath",
    "KnowledgeGuidedMilestone",
    "KnowledgeGuidedReasoningFacade",
    "KnowledgeProposal",
    "KnowledgeProposalRepository",
    "KnowledgeProposalStatus",
    "build_diagnosis_path",
    "build_knowledge_agent",
    "build_reasoning_facade",
]

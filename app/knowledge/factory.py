"""Composition helpers for the live, review-labelled knowledge loop."""
from __future__ import annotations

import logging

from app.concierge.reasoning import ReasoningFacade
from app.config import settings
from app.knowledge.agent import KnowledgeCuratorAgent
from app.knowledge.guided import (
    KnowledgeGuidedDiagnosisPath,
    KnowledgeGuidedMilestone,
    KnowledgeGuidedReasoningFacade,
)
from app.knowledge.repository import KnowledgeProposalRepository
from app.knowledge.retrieval import ClinicalKnowledgeBase
from app.llm.base import LLMGateway, ModelTier
from app.llm.factory import make_gateway
from app.paths.diagnosis import DiagnosisPath, Milestone
from app.paths.diagnosis.advanced_rag import (
    ParallelAdvancedRAGRetriever,
    build_advanced_rag_paths,
)
from app.paths.diagnosis.pubmed import PubMedRetriever

logger = logging.getLogger(__name__)


def _diagnosis_retrievers(gateway: LLMGateway):
    """Four peer RAG paths over one shared, globally rate-limited PubMed source."""
    pubmed = PubMedRetriever()
    if not settings.ENABLE_ADVANCED_RAG:
        return [pubmed]
    return build_advanced_rag_paths(
        base=pubmed,
        gateway=gateway,
        fusion_query_count=settings.RAG_FUSION_QUERY_COUNT,
        meta_repair_queries=settings.METARAG_MAX_REPAIR_QUERIES,
        imed_iterations=settings.IMEDRAG_MAX_ITERATIONS,
    )


def _milestone_options() -> dict:
    if not settings.ENABLE_ADVANCED_RAG:
        return {"node_timeout_s": settings.MILESTONE_NODE_TIMEOUT_S}
    return {
        "node_timeout_s": settings.MILESTONE_NODE_TIMEOUT_S,
        "retrieval_timeout_s": settings.ADVANCED_RAG_NODE_TIMEOUT_S,
        "max_evidence_items": settings.ADVANCED_RAG_MAX_EVIDENCE_ITEMS,
    }


def build_diagnosis_path(
    gateway: LLMGateway,
    *,
    knowledge: ClinicalKnowledgeBase | None = None,
) -> DiagnosisPath:
    """Build the production diagnosis path with PubMed RAG and live dynamic KB."""
    retrievers = _diagnosis_retrievers(gateway)
    if settings.ENABLE_DYNAMIC_KB_RAG:
        knowledge = knowledge or ClinicalKnowledgeBase()
        milestone = KnowledgeGuidedMilestone(
            gateway,
            knowledge=knowledge,
            retrievers=retrievers,
            **_milestone_options(),
        )
        return KnowledgeGuidedDiagnosisPath(
            gateway,
            knowledge=knowledge,
            milestone=milestone,
        )
    return DiagnosisPath(
        gateway,
        milestone=Milestone(
            gateway,
            retrievers=retrievers,
            **_milestone_options(),
        ),
    )


def build_reasoning_facade(
    gateway: LLMGateway,
    *,
    knowledge: ClinicalKnowledgeBase | None = None,
) -> ReasoningFacade:
    """Build the same retrieval-enabled CDSS facade for web and robot Concierges."""
    retrievers = _diagnosis_retrievers(gateway)
    if settings.ENABLE_DYNAMIC_KB_RAG:
        knowledge = knowledge or ClinicalKnowledgeBase()
        milestone = KnowledgeGuidedMilestone(
            gateway,
            knowledge=knowledge,
            retrievers=retrievers,
            **_milestone_options(),
        )
        return KnowledgeGuidedReasoningFacade(
            gateway,
            knowledge=knowledge,
            milestone=milestone,
        )
    return ReasoningFacade(
        gateway,
        milestone=Milestone(
            gateway,
            retrievers=retrievers,
            **_milestone_options(),
        ),
    )


def build_knowledge_agent() -> KnowledgeCuratorAgent | None:
    """Build the background curator, inheriting the live provider as a fallback.

    A separately configured stronger model is preferred. If that provider is not
    configured, the curator falls back to the same provider that already powers
    the running system instead of silently disabling knowledge development.
    """
    if not settings.ENABLE_KNOWLEDGE_AGENT:
        return None
    provider_name = settings.KNOWLEDGE_AGENT_PROVIDER
    model_name = settings.KNOWLEDGE_AGENT_MODEL
    try:
        gateway = make_gateway(
            provider=provider_name,
            models={
                ModelTier.FAST: settings.KNOWLEDGE_AGENT_FAST_MODEL,
                ModelTier.REASONING: model_name,
            },
            temperature=0,
            timeout=settings.KNOWLEDGE_AGENT_TIMEOUT_S,
            max_retries=settings.KNOWLEDGE_AGENT_MAX_RETRIES,
        )
    except Exception:  # noqa: BLE001 - try the already configured operational provider
        logger.warning(
            "Dedicated knowledge-curator provider unavailable; trying the live provider.",
            exc_info=True,
        )
        try:
            gateway = make_gateway(
                temperature=0,
                timeout=settings.KNOWLEDGE_AGENT_TIMEOUT_S,
                max_retries=settings.KNOWLEDGE_AGENT_MAX_RETRIES,
            )
            provider_name = settings.LLM_PROVIDER
            model_name = (
                settings.OLLAMA_MODEL_REASONING or "auto-detected local reasoning model"
                if provider_name in {"ollama", "local"}
                else settings.MODEL_REASONING
            )
        except Exception:  # noqa: BLE001 - diagnosis remains available without curation
            logger.warning(
                "Knowledge curator unavailable (no configured LLM provider).",
                exc_info=True,
            )
            return None
    curator_pubmed = PubMedRetriever(max_results=settings.KNOWLEDGE_AGENT_MAX_SOURCES)
    if settings.ENABLE_ADVANCED_RAG:
        curator_retriever = ParallelAdvancedRAGRetriever(build_advanced_rag_paths(
            base=curator_pubmed,
            gateway=gateway,
            fusion_query_count=settings.RAG_FUSION_QUERY_COUNT,
            meta_repair_queries=settings.METARAG_MAX_REPAIR_QUERIES,
            imed_iterations=settings.IMEDRAG_MAX_ITERATIONS,
        ))
    else:
        curator_retriever = curator_pubmed
    return KnowledgeCuratorAgent(
        gateway=gateway,
        repository=KnowledgeProposalRepository(),
        knowledge=ClinicalKnowledgeBase(),
        # Runs after the patient reply. A valid generated record becomes a live,
        # visibly unapproved KB addition; the doctor can approve, reject or
        # delete it later without blocking the consultation.
        retriever=curator_retriever,
        provider_name=provider_name,
        model_name=model_name,
    )

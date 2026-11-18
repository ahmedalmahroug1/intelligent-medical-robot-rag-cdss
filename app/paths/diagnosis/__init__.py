"""
app/paths/diagnosis
-------------------
The diagnosis path: the cheap interview loop (this package, File 6) and — dropped
into DiagnosisPath._run_milestone later — the evidence-pipeline milestone (File 7).

Public surface:
    DiagnosisPath   — the router-registered Path peer
    IntakeResult    — the loop's structured LLM-output contract
    Milestone       — the evidence-pipeline milestone
    Retriever       — the base class for evidence retrieval
    StaticRetriever — a static retriever for offline testing
"""

from app.paths.diagnosis.path import DiagnosisPath
from app.paths.diagnosis.schemas import IntakeResult
from app.paths.diagnosis.milestone import Milestone
from app.paths.diagnosis.retrieval import Retriever, StaticRetriever
from app.paths.diagnosis.advanced_rag import (
    IMedRAGRetriever,
    MetaRAGRetriever,
    ParallelAdvancedRAGRetriever,
    RAGFusionRetriever,
    SelfRAGRetriever,
    build_advanced_rag_paths,
)

__all__ = [
    "DiagnosisPath",
    "IntakeResult",
    "Milestone",
    "Retriever",
    "StaticRetriever",
    "RAGFusionRetriever",
    "MetaRAGRetriever",
    "SelfRAGRetriever",
    "IMedRAGRetriever",
    "ParallelAdvancedRAGRetriever",
    "build_advanced_rag_paths",
]

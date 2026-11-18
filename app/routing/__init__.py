"""
app/routing
-----------
The routing layer: the router (single dispatch authority), the path contract
(what a path is and returns), and turn classification.

Public surface:
    Path, PathResult, Transition, TransitionKind   — the path contract
    TurnClassifier, LLMClassifier                  — turn classification
    Router                                         — dispatch + transitions
"""

from app.routing.classifier import LLMClassifier, TurnClassifier
from app.routing.contract import Path, PathResult, Transition, TransitionKind
from app.routing.router import Router

__all__ = [
    "Path",
    "PathResult",
    "Transition",
    "TransitionKind",
    "TurnClassifier",
    "LLMClassifier",
    "Router",
]

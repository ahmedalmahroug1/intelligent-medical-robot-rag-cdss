"""
app/paths/treatment
-------------------
The treatment path: general management guidance for an already-known condition,
with a deterministic medication-safety overlay. A router-registered `Path` peer.

Public surface:
    TreatmentPath    — the router-registered Path peer
    TreatmentResult  — the path's structured LLM-output contract
"""

from app.paths.treatment.path import TreatmentPath
from app.paths.treatment.schemas import TreatmentResult

__all__ = ["TreatmentPath", "TreatmentResult"]

"""
app/orchestration
-----------------
Client-agnostic orchestration. The ConsultationService is the single place the
guard + router + persistence contracts are assembled; the FastAPI/SSE layer and
the robot are both thin callers of it.

Public surface:
    ConsultationService, TurnOutcome, ConsultationError
"""

from app.orchestration.service import (
    ConsultationError,
    ConsultationService,
    TurnOutcome,
)

__all__ = ["ConsultationService", "TurnOutcome", "ConsultationError"]

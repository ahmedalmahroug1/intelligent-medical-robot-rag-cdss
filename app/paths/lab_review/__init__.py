"""
app/paths/lab_review
--------------------
The lab-review path: closes the diagnostic loop with objective data. It recommends
labs for the current differential, parses entered results, interprets them against a
deterministic clinical reference, and nudges the working diagnosis's confidence/ranking.
A router-registered `Path` peer on the existing `Route.TEST_REVIEW`. Informational only.

Public surface:
    LabReviewPath  — the router-registered Path peer
"""

from app.paths.lab_review.path import LabReviewPath

__all__ = ["LabReviewPath"]

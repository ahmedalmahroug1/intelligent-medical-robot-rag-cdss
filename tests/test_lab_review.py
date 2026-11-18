"""
tests/test_lab_review.py
------------------------
Standalone tests for the LAB_REVIEW path. Run from the project root:

    python -m tests.test_lab_review   (or: PYTHONPATH=. python tests/test_lab_review.py)

All offline (a fake gateway that is never actually needed for the deterministic paths).

What these prove:
  1. lab_reference parses both "Hb = 8.2 / MCV: 68 / ferritin 4" and JSON, normalising
     names, and classifies values against the reference ranges.
  2. The path runs on the EXISTING Route.TEST_REVIEW (no new enum value).
  3. Entering supportive labs raises the matching diagnosis, demotes a contradicted one,
     re-ranks the differential, updates the working diagnosis IN PLACE (schema unchanged),
     and stores the values on state.visit.lab_results.
  4. A condition the labs reference does not cover (open-world) is left untouched.
  5. With no values, the path recommends the discriminating labs (writes recommended_tests).
  6. The reply is informational — never a prescription, always with a clinician disclaimer.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence

from app.core.enums import Route, Sex, RiskTier
from app.core.state import (
    ConsultationState, Hypothesis, PatientRecord, SessionState, VisitState, WorkingDiagnosis,
)
from app.llm.base import LLMGateway, LLMMessage, ModelTier
from app.routing.contract import TransitionKind
from app.paths.lab_review import LabReviewPath
from app.paths.lab_review.lab_reference import (
    classify_value, interpret, parse_labs, recommend_labs,
)


class FakeGateway(LLMGateway):
    """Only used by the optional fallback parser; the deterministic paths never call it."""
    async def complete_structured(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None, model: ModelTier | None = None,
    ) -> dict[str, Any]:
        return {}

    async def complete_text(self, *, system, user, history=None, model=None) -> str:
        return ""

    async def stream_text(self, *, system, user, history=None, model=None) -> AsyncIterator[str]:
        yield ""


def _state(differential=None, working=None) -> ConsultationState:
    patient = PatientRecord(patient_id="p1", age=40, sex=Sex.FEMALE)
    visit = VisitState(visit_id="v1", patient_id="p1")
    if differential:
        visit.differential = differential
    if working:
        visit.working_diagnosis = working
    session = SessionState(session_id="s1", patient_id="p1", visit_id="v1",
                           active_route=Route.TEST_REVIEW)
    return ConsultationState(patient=patient, visit=visit, session=session)


def _anemia_diff():
    return [
        Hypothesis(condition="Iron deficiency anemia", confidence=0.40),
        Hypothesis(condition="Thalassemia", confidence=0.45),
        Hypothesis(condition="Anemia of chronic disease", confidence=0.35),
    ]


async def _run():
    # 1. lab_reference parsing + classification
    assert parse_labs("Hb = 8.2\nMCV: 68\nferritin 4") == {"Hb": 8.2, "MCV": 68.0, "Ferritin": 4.0}
    assert parse_labs('{"Hb":8.2,"MCV":68,"Ferritin":4}') == {"Hb": 8.2, "MCV": 68.0, "Ferritin": 4.0}
    assert parse_labs({"hgb": 8.2}) == {"Hb": 8.2}
    assert classify_value("Hb", 8.2) == "low"
    assert classify_value("MCV", 68) == "low"
    assert classify_value("Hb", 14) == "normal"
    assert classify_value("TSH", 0.1) == "low"
    assert classify_value("T4", 3.0) == "high"

    # 2. reuses the existing route, no new enum
    assert LabReviewPath(FakeGateway()).route == Route.TEST_REVIEW

    # 3. supportive labs -> IDA up & leads, Anemia-of-chronic-disease (contradicted) down,
    #    working diagnosis updated in place, values stored.
    wd = WorkingDiagnosis(condition="Thalassemia", confidence=0.45,
                          risk_tier=RiskTier.LOW, reasoning="initial")
    st = _state(differential=_anemia_diff(), working=wd)
    path = LabReviewPath(FakeGateway())
    res = await path.handle(turn="Hb 8.2, MCV 68, Ferritin 4", state=st)

    assert st.visit.lab_results == {"Hb": 8.2, "MCV": 68.0, "Ferritin": 4.0}
    conf = {h.condition: float(h.confidence) for h in st.visit.differential}
    assert st.visit.differential[0].condition == "Iron deficiency anemia", \
        [h.condition for h in st.visit.differential]
    assert conf["Iron deficiency anemia"] > 0.40              # supported -> up
    assert conf["Anemia of chronic disease"] < 0.35          # contradicted -> down
    assert conf["Thalassemia"] <= 0.45                       # not boosted (ferritin contradicts)
    assert st.visit.working_diagnosis.condition == "Iron deficiency anemia"  # lead flipped, in place
    assert st.visit.working_diagnosis.risk_tier == RiskTier.LOW              # other fields untouched
    assert "information" in res.reply.lower() and "clinician" in res.reply.lower()  # informational disclaimer
    assert res.transition.kind == TransitionKind.STAY

    # 4. out-of-KB condition untouched (open-world safe)
    st3 = _state(differential=[
        Hypothesis(condition="Migraine", confidence=0.70),
        Hypothesis(condition="Iron deficiency anemia", confidence=0.30),
    ])
    await path.handle(turn="Hb 8.2, MCV 68, Ferritin 4", state=st3)
    mig = next(h for h in st3.visit.differential if h.condition == "Migraine")
    assert abs(float(mig.confidence) - 0.70) < 1e-6

    # 5. no values -> recommend the discriminating labs
    st2 = _state(differential=_anemia_diff(), working=wd)
    res2 = await path.handle(turn="I did the labs, what should I look at?", state=st2)
    names = [t.name for t in st2.visit.recommended_tests]
    assert "Ferritin" in names and "Hb" in names, names
    assert "clinician" in res2.reply.lower()
    assert res2.transition.kind == TransitionKind.STAY

    # 6. helpers
    assert dict(interpret({"Hb": 8.2}, ["Anemia"])[1])["Anemia"]["matched_kb"] is True
    recs = dict(recommend_labs(["Iron deficiency anemia", "Thalassemia"]))
    assert "Ferritin" in recs and "Hb" in recs

    print("test_lab_review: ALL PASS")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

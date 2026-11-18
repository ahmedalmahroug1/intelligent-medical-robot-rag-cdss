"""
tests/test_milestone.py
-----------------------
Standalone tests for File 7b (the diagnosis milestone). Run from the project root:

    python -m tests.test_milestone

Needs pydantic (your env has it). No network — retrieval goes through fakes.

What these tests PROVE:
    1. The full pipeline runs end to end through the executor and writes a working
       diagnosis, ranked evidence, and a PENDING_APPROVAL report to the
       blackboard; the patient gets a safe summary and phase becomes COMPLETE.
    2. The deterministic safety floor: a high-harm or low-confidence conclusion
       is given a concrete confirmation pathway even when the model omits one.
    3. Graceful partial failure: one evidence source failing does NOT stop the
       milestone — verification runs on what arrived and a diagnosis is produced.
    4. Degraded path: if the load-bearing differential node fails, the milestone
       does not guess — it returns a "see a clinician" reply and completes.

Because the milestone's nodes run in PARALLEL, their LLM calls do not arrive in a
fixed order. The MappingGateway below therefore dispatches canned responses by a
keyword in each node's system prompt (not by call order), which is robust to the
scheduling. The keywords match the opening phrases of the milestone prompts.
"""

from __future__ import annotations

import asyncio
import inspect
import traceback
from typing import Any, AsyncIterator, Sequence

from app.core.enums import DiagnosticPhase, ReportStatus, Route
from app.core.state import (
    ConsultationState,
    EvidenceItem,
    PatientRecord,
    SessionState,
    VisitState,
)
from app.llm.base import LLMError, LLMGateway, LLMMessage, ModelTier
from app.paths.diagnosis import DiagnosisPath, Milestone, Retriever, StaticRetriever
from app.paths.diagnosis.retrieval import EvidenceQuery


# ─────────────────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────────────────

class MappingGateway(LLMGateway):
    """Returns a canned structured dict chosen by matching a keyword found in the
    system prompt — order-independent, so it works under parallel scheduling."""

    def __init__(self, by_keyword: dict[str, dict[str, Any]]) -> None:
        self._by_keyword = by_keyword
        self.seen: list[str] = []
        # Last user prompt seen for each matched keyword, so a test can assert WHAT
        # a node was actually given (e.g. that the report node received the tests,
        # evidence, and med-safety it is asked to summarise).
        self.user_by_keyword: dict[str, str] = {}

    async def complete_structured(
        self, *, system: str, user: str,
        history: Sequence[LLMMessage] | None = None, model: ModelTier | None = None,
    ) -> dict[str, Any]:
        for keyword, response in self._by_keyword.items():
            if keyword in system:
                self.seen.append(keyword)
                self.user_by_keyword[keyword] = user
                # A canned response may be a function of the user prompt — needed for
                # the per-paper appraisal, where one prompt is sent per paper.
                return response(user) if callable(response) else response
        raise LLMError(f"MappingGateway: no canned response for system prompt "
                       f"starting: {system[:48]!r}")

    async def complete_text(self, *, system, user, history=None, model=None) -> str:
        return ""

    async def stream_text(self, *, system, user, history=None, model=None) -> AsyncIterator[str]:
        yield ""


class FailingRetriever(Retriever):
    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    async def search(self, query: EvidenceQuery) -> list[EvidenceItem]:
        raise RuntimeError(f"{self.source_name} is unreachable")


# Keyword -> canned response, keyed on each milestone prompt's opening phrase.
DIFFERENTIAL_KW = "refining a differential"
TEST_REC_KW = "recommend diagnostic tests"
QUERY_PLAN_KW = "plan literature-search queries"
APPRAISE_KW = "appraise one retrieved medical paper"
SYNTHESIS_KW = "synthesise the appraised evidence"
MED_SAFETY_KW = "check medication safety"
WORKING_DX_KW = "produce a working diagnosis"
REPORT_KW = "write a clinical report"


def _appraise_paper(user: str) -> dict[str, Any]:
    """Per-paper appraisal stand-in: the node sends one prompt per paper, so this
    keys on which paper is in the prompt. Weak papers (PMID:2) are dropped."""
    if "PMID:2" in user:
        return {"identifier": "PMID:2", "quality_score": 0.2, "keep": False,
                "supports": [], "refutes": [], "key_finding": "underpowered", "notes": "weak"}
    return {"identifier": "PMID:1", "quality_score": 0.8, "keep": True,
            "supports": ["Viral pharyngitis"], "refutes": [],
            "key_finding": "Example finding used by the test fixture", "notes": "fixture"}


def _full_responses(
    *, risk: str = "moderate", confidence: float = 0.72, pathway: str = "Order a CBC to confirm.",
) -> dict[str, Any]:
    return {
        DIFFERENTIAL_KW: {
            "hypotheses": [
                {"condition": "Viral pharyngitis", "confidence": 0.6, "supporting": ["sore throat"], "against": []},
                {"condition": "Strep throat", "confidence": 0.3, "supporting": [], "against": ["no fever"]},
            ],
            "summary": "Likely viral.",
        },
        TEST_REC_KW: {
            "tests": [
                {"name": "Rapid strep test", "rationale": "rule out strep", "discriminates": ["Strep throat"]}
            ]
        },
        QUERY_PLAN_KW: {
            "queries": [
                {"hypothesis": "Viral pharyngitis", "query": "viral pharyngitis diagnosis",
                 "rationale": "confirm the leading hypothesis"},
            ]
        },
        # Per-paper appraisal: one call per paper, so the response keys on the paper
        # in the user prompt. The weak item (PMID:2) is dropped (keep=False).
        APPRAISE_KW: _appraise_paper,
        SYNTHESIS_KW: {
            "by_hypothesis": [
                {"condition": "Viral pharyngitis", "supporting_refs": ["PMID:1"],
                 "refuting_refs": [], "key_finding": "Example finding used by the test fixture",
                 "strength": "moderate"},
            ],
            "contradictions": "",
            "overall": "moderate evidence favouring viral pharyngitis",
        },
        MED_SAFETY_KW: {"has_concern": False, "concerns": [], "summary": "no concerns"},
        WORKING_DX_KW: {
            "condition": "Viral pharyngitis",
            "confidence": confidence,
            "risk_tier": risk,
            "reasoning": "Self-limiting viral illness.",
            "confirmation_pathway": pathway,
        },
        REPORT_KW: {"content": "DRAFT REPORT: viral pharyngitis, pending approval."},
    }


def _state() -> ConsultationState:
    patient = PatientRecord(patient_id="p1", age=29)
    visit = VisitState(visit_id="v1", patient_id="p1")
    session = SessionState(
        session_id="s1", patient_id="p1", visit_id="v1",
        active_route=Route.DIAGNOSIS, phase=DiagnosticPhase.MILESTONE,
    )
    return ConsultationState(patient=patient, visit=visit, session=session)


def _evidence(identifier: str, title: str) -> EvidenceItem:
    return EvidenceItem(source="", title=title, identifier=identifier,
                        summary="…", published_year=2020, study_type="RCT")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_milestone_runs_end_to_end_and_drafts_report() -> None:
    gw = MappingGateway(_full_responses())
    retrievers = [
        StaticRetriever("PubMed", [_evidence("PMID:1", "Good study"), _evidence("PMID:2", "Weak study")]),
        StaticRetriever("Guidelines", []),
    ]
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=retrievers))
    st = _state()

    res = await path.handle(turn="(enter milestone)", state=st)

    assert st.visit.working_diagnosis is not None
    assert st.visit.working_diagnosis.condition == "Viral pharyngitis"
    assert st.visit.report is not None
    assert st.visit.report.status is ReportStatus.PENDING_APPROVAL
    assert st.session.phase is DiagnosticPhase.COMPLETE
    # ranked evidence: the weak item (keep=False) was dropped, the good one kept
    assert [e.identifier for e in st.visit.evidence] == ["PMID:1"]
    assert st.visit.evidence[0].quality_score == 0.8
    assert st.visit.evidence[0].source == "PubMed"  # source stamped during retrieval
    # patient-facing reply is safe and mentions clinician review
    assert "review" in res.reply.lower()
    assert "Viral pharyngitis" in res.reply


async def test_high_risk_forces_confirmation_pathway() -> None:
    # Model returns CRITICAL risk but no confirmation pathway -> node fills a default.
    responses = _full_responses(risk="critical", confidence=0.9, pathway="")
    gw = MappingGateway(responses)
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=[StaticRetriever("PubMed", [])]))
    st = _state()

    res = await path.handle(turn="x", state=st)

    wd = st.visit.working_diagnosis
    assert wd is not None
    assert wd.confirmation_pathway != "", "high risk must carry a confirmation pathway"
    assert "clinician" in wd.confirmation_pathway.lower()
    # and the patient is told to be seen promptly
    assert "promptly" in res.reply.lower()


async def test_low_confidence_forces_confirmation_pathway() -> None:
    responses = _full_responses(risk="low", confidence=0.2, pathway="")
    gw = MappingGateway(responses)
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=[StaticRetriever("PubMed", [])],
                                                 closure_confidence_floor=0.6))
    st = _state()

    await path.handle(turn="x", state=st)

    wd = st.visit.working_diagnosis
    assert wd is not None and wd.confirmation_pathway != "", "low confidence needs a pathway"


async def test_evidence_source_failure_does_not_stop_milestone() -> None:
    gw = MappingGateway(_full_responses())
    retrievers = [
        FailingRetriever("PubMed"),                        # this one blows up
        StaticRetriever("Guidelines", [_evidence("G:1", "Guideline")]),  # this one works
    ]
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=retrievers))
    st = _state()

    res = await path.handle(turn="x", state=st)

    # milestone still completed despite PubMed failing
    assert st.visit.working_diagnosis is not None
    assert st.visit.report is not None
    assert st.session.phase is DiagnosticPhase.COMPLETE
    assert "review" in res.reply.lower()


async def test_degraded_when_differential_fails() -> None:
    # No canned response for the differential node -> it raises LLMError -> FAILED,
    # everything downstream is skipped, milestone degrades safely.
    responses = _full_responses()
    del responses[DIFFERENTIAL_KW]
    gw = MappingGateway(responses)
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=[]))
    st = _state()

    res = await path.handle(turn="x", state=st)

    assert st.visit.working_diagnosis is None
    assert "clinician" in res.reply.lower()
    assert st.session.phase is DiagnosticPhase.COMPLETE


async def test_report_and_working_dx_are_fed_tests_evidence_and_med_safety() -> None:
    # Regression guard: the report node must be GIVEN the recommended tests, the
    # appraised evidence, and the med-safety finding it is asked to summarise (it
    # previously received only the working dx + differential, so it omitted or
    # fabricated those sections). The working-dx node must ground on the retrieved
    # evidence itself, not merely a count, or RAG cannot affect the diagnosis.
    gw = MappingGateway(_full_responses())
    retrievers = [
        StaticRetriever("PubMed", [_evidence("PMID:1", "Good study"), _evidence("PMID:2", "Weak study")]),
    ]
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=retrievers))
    st = _state()

    await path.handle(turn="x", state=st)

    report_prompt = gw.user_by_keyword[REPORT_KW]
    assert "Rapid strep test" in report_prompt, "report must receive the recommended tests"
    assert "PMID:1" in report_prompt, "report must receive the appraised evidence"
    assert "no concerns" in report_prompt, "report must receive the med-safety finding"
    # The dropped (keep=False) evidence must not be smuggled into the report.
    assert "PMID:2" not in report_prompt

    wd_prompt = gw.user_by_keyword[WORKING_DX_KW]
    assert "PMID:1" in wd_prompt, "working diagnosis must be grounded on retrieved evidence"


async def test_non_destructive_working_dx_keeps_differential_lead() -> None:
    # The measured RAG failure mode: the working-dx model demotes the already-correct
    # #1 hypothesis to a rival. With the NON-DESTRUCTIVE default, the milestone must
    # keep the differential's top hypothesis as the lead regardless of the model's pick.
    responses = _full_responses()
    responses[WORKING_DX_KW] = {
        "condition": "Strep throat",           # model tries to demote from Viral pharyngitis
        "confidence": 0.7, "risk_tier": "moderate",
        "reasoning": "x", "confirmation_pathway": "see clinician",
    }
    gw = MappingGateway(responses)
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=[]))
    st = _state()

    await path.handle(turn="x", state=st)

    wd = st.visit.working_diagnosis
    assert wd is not None and wd.condition == "Viral pharyngitis", wd.condition
    assert wd.differential[0].condition == "Viral pharyngitis"


async def test_evidence_can_override_lets_model_repick_lead() -> None:
    # The ablation path: with evidence_can_override=True the model's re-pick wins, so
    # the on/off behaviour can be measured against the same cases.
    responses = _full_responses()
    responses[WORKING_DX_KW] = {
        "condition": "Strep throat", "confidence": 0.7, "risk_tier": "moderate",
        "reasoning": "x", "confirmation_pathway": "see clinician",
    }
    gw = MappingGateway(responses)
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=[], evidence_can_override=True))
    st = _state()

    await path.handle(turn="x", state=st)

    assert st.visit.working_diagnosis.condition == "Strep throat"


async def test_kb_fusion_node_runs_in_dag_non_destructively() -> None:
    # The KB reciprocal-rank fusion node runs inside the milestone DAG: the extractor
    # is called, the milestone completes, and the lead stays the differential's top
    # (top-1 protected). Fusion surfaces KB conditions into the tail for coverage.
    from app.paths.diagnosis.kb_prior import KBReranker
    responses = _full_responses()
    # Present symptoms in the curated-KB vocabulary, so fusion produces a KB ranking.
    responses["clinical information extractor"] = {
        "present": ["sore_throat", "pain_swallowing", "cough", "nasal_congestion"]
    }
    gw = MappingGateway(responses)
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=[], kb_rerank=KBReranker(mode="fuse")))
    st = _state()

    res = await path.handle(turn="x", state=st)

    wd = st.visit.working_diagnosis
    assert wd is not None and wd.condition == "Viral pharyngitis"
    assert "clinical information extractor" in gw.seen, "fusion must call the KB extractor"
    assert st.session.phase is DiagnosticPhase.COMPLETE
    assert "review" in res.reply.lower()


async def test_ensemble_differential_runs_and_completes() -> None:
    # Self-consistency ensemble: sampling the differential K times on the ensemble
    # gateway and RRF-fusing them. K identical fakes fuse to the same ranking, so this
    # proves the ensemble path runs end-to-end and produces the differential.
    gw = MappingGateway(_full_responses())
    path = DiagnosisPath(
        gw, milestone=Milestone(gw, retrievers=[], ensemble_gateway=gw, ensemble_k=3)
    )
    st = _state()
    await path.handle(turn="x", state=st)
    wd = st.visit.working_diagnosis
    assert wd is not None and wd.condition == "Viral pharyngitis"


async def test_ensemble_falls_back_when_all_samples_fail() -> None:
    # Floor: if EVERY ensemble sample fails (the ensemble gateway has no canned
    # differential), the node falls back to a single call on the MAIN gateway, so the
    # milestone still completes — never worse than one plain differential.
    gw = MappingGateway(_full_responses())
    bad = MappingGateway({})  # any differential call raises LLMError
    path = DiagnosisPath(
        gw, milestone=Milestone(gw, retrievers=[], ensemble_gateway=bad, ensemble_k=3)
    )
    st = _state()
    await path.handle(turn="x", state=st)
    assert st.visit.working_diagnosis is not None
    assert st.visit.working_diagnosis.condition == "Viral pharyngitis"


async def test_refute_prunes_contradicted_nonlead() -> None:
    # The refutation pass drops ONLY a non-lead candidate the account contradicts.
    responses = _full_responses()
    responses["review a differential"] = {
        "remove": [{"condition": "Strep throat", "reason": "no fever, no exudate"}]
    }
    gw = MappingGateway(responses)
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=[], refute=True))
    st = _state()
    await path.handle(turn="x", state=st)
    wd = st.visit.working_diagnosis
    assert wd is not None and wd.condition == "Viral pharyngitis"
    conds = [h.condition for h in wd.differential]
    assert "Strep throat" not in conds, conds


async def test_refute_never_drops_the_lead() -> None:
    # Floor: even if the model tries to remove the LEAD, refutation must keep it — so
    # the refutation pass can never push top-1 below the raw differential's top-1.
    responses = _full_responses()
    responses["review a differential"] = {
        "remove": [{"condition": "Viral pharyngitis", "reason": "spurious"}]
    }
    gw = MappingGateway(responses)
    path = DiagnosisPath(gw, milestone=Milestone(gw, retrievers=[], refute=True))
    st = _state()
    await path.handle(turn="x", state=st)
    wd = st.visit.working_diagnosis
    assert wd is not None and wd.condition == "Viral pharyngitis"
    assert wd.differential[0].condition == "Viral pharyngitis"


async def test_reranker_timeout_falls_back_to_raw_differential() -> None:
    # Floor (regression: surfaced on a throttled provider) — a KB rerank call that
    # exceeds the node timeout must NOT sink the milestone. As an OPTIONAL dep of
    # working_dx, a timed-out kb_rerank falls back to the raw differential and the
    # milestone still completes with the differential's lead.
    class _SlowKB:
        mode = "fuse"

        async def fuse(self, gateway, text, hyps):
            await asyncio.sleep(1.0)   # exceeds the tiny node timeout below
            return hyps

    gw = MappingGateway(_full_responses())
    path = DiagnosisPath(gw, milestone=Milestone(
        gw, retrievers=[], kb_rerank=_SlowKB(), node_timeout_s=0.2))
    st = _state()
    await path.handle(turn="x", state=st)
    wd = st.visit.working_diagnosis
    assert wd is not None and wd.condition == "Viral pharyngitis", \
        "a reranker timeout must fall back to the raw differential, not sink the milestone"


async def test_path_without_milestone_still_returns_bridge() -> None:
    # File 6 behaviour preserved: no milestone wired -> honest bridge, phase stays.
    gw = MappingGateway({})
    path = DiagnosisPath(gw)  # no milestone
    st = _state()

    res = await path.handle(turn="x", state=st)

    assert "assessment" in res.reply.lower()
    assert st.visit.working_diagnosis is None
    assert gw.seen == []  # nothing called


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_milestone_runs_end_to_end_and_drafts_report,
    test_high_risk_forces_confirmation_pathway,
    test_low_confidence_forces_confirmation_pathway,
    test_evidence_source_failure_does_not_stop_milestone,
    test_degraded_when_differential_fails,
    test_report_and_working_dx_are_fed_tests_evidence_and_med_safety,
    test_non_destructive_working_dx_keeps_differential_lead,
    test_evidence_can_override_lets_model_repick_lead,
    test_kb_fusion_node_runs_in_dag_non_destructively,
    test_ensemble_differential_runs_and_completes,
    test_ensemble_falls_back_when_all_samples_fail,
    test_refute_prunes_contradicted_nonlead,
    test_refute_never_drops_the_lead,
    test_reranker_timeout_falls_back_to_raw_differential,
    test_path_without_milestone_still_returns_bridge,
]


def main() -> None:
    passed = failed = 0
    for test in ALL_TESTS:
        try:
            result = test()
            if inspect.iscoroutine(result):
                asyncio.run(result)
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

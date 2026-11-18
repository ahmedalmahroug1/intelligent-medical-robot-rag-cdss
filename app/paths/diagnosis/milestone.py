"""
app/paths/diagnosis/milestone.py
--------------------------------
The diagnosis MILESTONE: the run-once evidence pipeline that fires when the
interview loop converges. It turns the gathered differential into a calibrated,
risk-gated working diagnosis and a draft report awaiting clinician approval.

It is built as a DAG handed to the executor (File 7), so the independent work
runs in parallel:

        differential                         (root; consolidates the differential)
        ├── test_rec                         } depend only on the differential,
        ├── med_safety                       } so they run concurrently
        └── query_plan                       } (writes one targeted query per hypothesis)
              └── evidence:<sourceA/B>  (optional; use the planned queries)
        evidence_appraise  (required: differential; optional: every evidence node)
        evidence_synthesis (required: differential, evidence_appraise)
        working_diagnosis  (required: differential, evidence_synthesis, med_safety;
                                       optional: test_rec)
        report             (required: working_diagnosis)

The RAG sub-pipeline is agentic: a query-planner turns each hypothesis into a
focused query; retrieval runs those; an appraisal agent reads each paper on its
own (one call per paper, fanned out concurrently) and extracts the concrete
external finding it contributes; a synthesis agent combines those findings into a
by-hypothesis brief. That brief — not truncated abstracts — is what grounds the
working diagnosis, so the literature can actually move the conclusion.

Dependency choices: evidence retrieval and medication-safety depend on the
*differential*, not on test-recommendation, because their inputs are the candidate
conditions and the patient's record. The query-planner is an OPTIONAL dependency of
retrieval (retrieval falls back to an OR of condition names if planning fails), and
evidence sources are OPTIONAL dependencies of appraisal, so one source failing or
timing out never stops the run. `evidence_appraise`/`evidence_synthesis` always
succeed (returning empty when there is no evidence), so the milestone produces a
working diagnosis with RAG on OR off. Only `differential` is load-bearing — if it
fails, the milestone degrades to "see a clinician" rather than guessing.

Safety that does not depend on the model behaving:
    - The working-diagnosis node applies a DETERMINISTIC floor on top of the
      model: if harm-if-wrong is HIGH/CRITICAL, or confidence is below the
      closure floor, a concrete confirmation pathway (a test, or in-person care)
      is REQUIRED — if the model omits it, the node fills a safe default. The
      system never closes a consequential case by asserting a diagnosis alone.
    - The patient-facing summary is composed deterministically from the working
      diagnosis (not free-generated), so the safety-critical wording cannot be
      hallucinated, and it always states that a clinician reviews the result.
    - The report is created as PENDING_APPROVAL — it has no authority until a
      clinician signs it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime, timezone

from app.core.enums import DiagnosticPhase, ReportStatus, RiskTier
from app.core.state import (
    ConsultationState,
    EvidenceItem,
    Hypothesis,
    Report,
    WorkingDiagnosis,
)
from app.execution import Executor, ExecutionReport, Task, TaskContext
from app.llm.base import LLMError, LLMGateway, ModelTier
from app.paths.diagnosis.kb_prior import KBReranker, fuse_llm_rankings
from app.paths.diagnosis.retrieval import EvidenceQuery, Retriever
from app.paths.diagnosis.schemas import (
    DifferentialResult,
    EvidenceSynthesisResult,
    MedicationSafetyResult,
    PaperAppraisal,
    QueryPlanResult,
    RefutationResult,
    ReportResult,
    TestRecommendationResult,
    WorkingDiagnosisResult,
)
from app.routing.contract import PathResult, Transition

logger = logging.getLogger(__name__)

DEFAULT_CLOSURE_CONFIDENCE_FLOOR = 0.6   # below this, a confirmation pathway is mandatory
DEFAULT_NODE_TIMEOUT_S = 20.0
DEFAULT_MAX_QUERY_HYPOTHESES = 3         # how many hypotheses the planner writes queries for
DEFAULT_MIN_QUERY_CONFIDENCE = 0.15      # below this prior, a hypothesis is not worth searching
_APPRAISE_CONCURRENCY = 5                # cap on simultaneous per-paper appraisal calls
_EVIDENCE_PREFIX = "evidence:"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Each node's system prompt opens with a stable identifying phrase (used to keep
# the prompts self-documenting; tests also key on these).
_DIFFERENTIAL_PROMPT = """You are refining a differential diagnosis for a \
decision-support system. Given the full interview and the running differential, \
produce a consolidated, honest differential ranked by a calibrated confidence in \
[0,1], each with supporting findings and findings against. List up to 8-10 \
plausible conditions: the leading hypotheses first, then less-likely but \
clinically important alternatives as an honestly LOW-confidence tail — do not pad \
with implausible entries, and do not inflate confidence. CARRY FORWARD the \
interview's leading hypotheses: do NOT silently drop or demote a well-supported \
hypothesis the running differential ranked highly unless the conversation gives a \
concrete reason against it. The chief complaint is a starting point, not the answer \
— do not re-anchor the whole differential on it and discard a better-supported \
lead (e.g. do not collapse a drug-induced dystonia, established from a recent \
medication, into a local jaw disorder just because the complaint mentions the jaw). \
Weigh the WHOLE presentation: a single non-specific finding (e.g. sputum colour, \
one low-grade symptom) must NOT override a hypothesis the overall picture still \
supports; let confidence move gradually unless a finding is genuinely decisive. \
Name each condition as a SPECIFIC, standard diagnosis (e.g. "Viral pharyngitis", \
"Pulmonary embolism", "Acute dystonic reaction") — NEVER a vague symptom category \
or grouping such as "infection or fever", "viral illness", "an infection", or "some \
kind of inflammation": those cannot be scored or acted on. Return ONLY JSON:
{"hypotheses":[{"condition":"...","confidence":0.0,"supporting":["..."],"against":["..."]}],"summary":"one or two sentences"}"""

_TEST_REC_PROMPT = """You recommend diagnostic tests for a decision-support \
system. Given the differential, list the few investigations that would best \
discriminate the leading hypotheses, each with a one-line rationale and which \
hypotheses it separates. Return ONLY JSON:
{"tests":[{"name":"...","rationale":"...","discriminates":["condition","condition"]}]}"""

_QUERY_PLAN_PROMPT = """You plan literature-search queries for a decision-support \
system. Given the differential, write ONE focused PubMed query per leading \
hypothesis: it should target papers that CONFIRM or DISTINGUISH that hypothesis \
(e.g. its diagnosis or the features that separate it from the rivals), not generic \
overviews. Keep each query broad enough to actually return results — use 2-4 \
well-chosen terms (the condition plus one or two clinical concepts), not a long \
over-constrained boolean. Return ONLY JSON:
{"queries":[{"hypothesis":"...","query":"...","rationale":"..."}]}"""

_APPRAISE_PROMPT = """You appraise one retrieved medical paper for a \
decision-support system. Given THIS patient, the candidate conditions, and the \
paper's abstract, judge whether it helps DIAGNOSE this patient — whether it \
describes symptoms, signs, or test findings that DISTINGUISH the candidate \
conditions for a patient with this presentation.

Check population fit FIRST: if the paper's study population clearly does not match \
this patient — e.g. a paediatric study for an adult patient (or vice versa), or a \
different sex/context that changes the finding's applicability — then its finding \
does not apply here: set keep=false, or if you keep it, say so in notes and do NOT \
mark it as support/refute for any hypothesis.

Set keep=false when the paper does not help diagnosis — for example a treatment or \
drug-efficacy trial, a prevalence/epidemiology survey, or a paper merely ABOUT a \
condition with no diagnostic, distinguishing detail. A paper being topically about a \
condition is NOT by itself a reason to keep it or to mark it as support.

For a kept paper, extract the single most relevant CONCRETE diagnostic finding (a \
result, statistic, or distinguishing feature) — quote specifics from the abstract, \
do not just paraphrase the title. Mark supports/refutes ONLY for a hypothesis the \
finding genuinely discriminates for this patient. Give a quality_score in [0,1]. \
Return ONLY JSON:
{"identifier":"...","quality_score":0.0,"keep":true,"supports":["..."],"refutes":["..."],"key_finding":"...","notes":"..."}"""

_SYNTHESIS_PROMPT = """You synthesise the appraised evidence for a decision-support \
system. Given the differential and the per-paper appraisals, produce a brief \
organised BY hypothesis: for each, the supporting and refuting references, the key \
external finding that bears on it, and an overall strength (strong/moderate/weak/\
none).

Rules:
- Base this ONLY on the supplied findings; do not invent evidence.
- "strength" reflects how strongly a finding DISCRIMINATES that hypothesis for THIS \
patient — NOT how many papers mention the condition. Several papers that are merely \
about a condition are 'weak' or 'none', not 'strong'.
- A finding from a population that does not match this patient (e.g. a paediatric \
study applied to an adult) is at most 'weak' and must NOT be treated as strong \
support, however striking the number.
- Having no appraised paper for a hypothesis means its evidence is 'none' (silent) — \
that does NOT count against the hypothesis, and you must not infer it is less likely.
Note any contradictions between papers. Return ONLY JSON:
{"by_hypothesis":[{"condition":"...","supporting_refs":["..."],"refuting_refs":["..."],"key_finding":"...","strength":"..."}],"contradictions":"...","overall":"one sentence"}"""

_REFUTE_PROMPT = """You review a differential diagnosis for a decision-support \
system. Given the differential and the FULL interview, identify ONLY the candidate \
conditions that the patient's OWN statements DIRECTLY CONTRADICT — a stated fact \
that rules the condition out (e.g. the patient explicitly denies its cardinal \
feature, or a demographic/finding makes it impossible). Be CONSERVATIVE: if a \
condition is merely less likely, or just unsupported by what was said, do NOT \
remove it — remove ONLY on an explicit contradiction. Never remove the single \
most-likely candidate unless the account flatly rules it out. It is correct to \
return an empty list when nothing is clearly contradicted. Return ONLY JSON:
{"remove":[{"condition":"exact name from the differential","reason":"the patient statement that rules it out"}]}"""

_MED_SAFETY_PROMPT = """You check medication safety for a decision-support \
system. Given the patient's active medications and allergies and the candidate \
conditions (and their likely treatments), flag any interactions, \
contraindications, or allergy risks. Be specific and do not invent drugs the \
patient is not taking. Return ONLY JSON:
{"has_concern":false,"concerns":["..."],"summary":"one sentence"}"""

_WORKING_DX_PROMPT = """You produce a working diagnosis for a decision-support \
system. You are NOT giving a final verdict — a clinician reviews this.

Choose the single most likely working diagnosis FROM THE DIFFERENTIAL below. The \
differential confidences are the patient-specific PRIOR: they already weigh THIS \
patient's history and symptoms, so they are your starting point.

How to use the external evidence — read carefully, this is where mistakes happen:
- The literature describes conditions in general. The mere EXISTENCE of papers \
about a condition (its treatment, prevalence, or management) is NOT evidence that \
THIS patient has it. A well-studied condition is not a more likely diagnosis.
- ABSENCE of retrieved papers for a hypothesis is NOT evidence against it: what got \
retrieved depends on how much is published and on query wording, not on this patient.
- Only let the evidence change the ranking when it supplies a PATIENT-SPECIFIC \
discriminating feature that clearly matches or contradicts this patient's \
presentation. Do NOT promote a lower-prior hypothesis over a higher-prior one just \
because more or higher-quality literature was retrieved for it.
- A finding drawn from a population that does not match this patient (e.g. a \
paediatric study for an adult) is NOT a patient-specific reason to move the \
ranking — discount it.
- When the evidence does not give a specific, patient-matched reason to differ, STAY \
with the differential's leading hypothesis.

Then give: a CALIBRATED confidence in [0,1] that honestly reflects uncertainty; a \
harm-if-wrong tier (low, moderate, high, or critical) based on what happens if this \
is wrong and untreated; concise reasoning; and a confirmation_pathway — the concrete \
next step (a specific test, or in-person clinical assessment) that would confirm it. \
If confidence is not high, or harm-if-wrong is high or critical, the \
confirmation_pathway MUST be a real next step and must not tell the patient to skip \
clinical care. Return ONLY JSON:
{"condition":"...","confidence":0.0,"risk_tier":"moderate","reasoning":"...","confirmation_pathway":"..."}"""

_REPORT_PROMPT = """You write a clinical report for a clinician to review in a \
decision-support system. Summarise, in clear clinical language: the presentation, \
the differential with confidences, the recommended tests, the appraised evidence, \
any medication-safety concerns, and the working diagnosis with its confirmation \
pathway. State clearly that it is a draft pending clinician approval. Return ONLY \
JSON:
{"content":"the full report text"}"""


class Milestone:
    """Runs the diagnosis evidence pipeline and returns the patient-facing reply."""

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        retrievers: Sequence[Retriever] = (),
        model: ModelTier = ModelTier.REASONING,
        executor: Executor | None = None,
        node_timeout_s: float = DEFAULT_NODE_TIMEOUT_S,
        closure_confidence_floor: float = DEFAULT_CLOSURE_CONFIDENCE_FLOOR,
        max_query_hypotheses: int = DEFAULT_MAX_QUERY_HYPOTHESES,
        min_query_confidence: float = DEFAULT_MIN_QUERY_CONFIDENCE,
        kb_rerank: KBReranker | None = None,
        evidence_can_override: bool = False,
        build_report: bool = True,
        lean: bool = False,
        ensemble_gateway: LLMGateway | None = None,
        ensemble_k: int = 1,
        ensemble_delay_s: float = 0.0,
        refute: bool = False,
        retrieval_timeout_s: float | None = None,
        max_evidence_items: int = 12,
    ) -> None:
        self._gateway = gateway
        self._retrievers = list(retrievers)
        self._model = model
        self._executor = executor or Executor(default_timeout=node_timeout_s)
        self._closure_floor = closure_confidence_floor
        self._max_query_hypotheses = max_query_hypotheses
        self._min_query_confidence = min_query_confidence
        # The clinician-facing report node is a large summary generation that does NOT
        # affect the working diagnosis or its ranking. During accuracy evaluation it is
        # pure cost (tokens/latency), so it can be skipped without changing any measured
        # top-k. Defaults True (real deployments draft the report a clinician signs).
        self._build_report = build_report
        # LEAN accuracy-eval graph: differential → (kb fusion) → working_diagnosis only.
        # Under the non-destructive rule the top-k ranking is fully determined by the
        # (fused) differential, so test-recommendation, medication-safety, evidence
        # synthesis and the report do NOT change the measured accuracy — skipping them
        # yields identical top-k at a fraction of the API tokens. Off in deployment.
        self._lean = lean
        # NON-DESTRUCTIVE by default: the working-diagnosis step reads the external
        # evidence for confidence/reasoning/pathway, but it may NOT change which
        # condition leads the ranking. The measured reason: literature RAG demoted
        # already-correct leads toward better-published rivals far more often than it
        # corrected a wrong lead. Set True to let evidence re-pick the lead (ablation).
        self._evidence_can_override = evidence_can_override
        # Optional naive-Bayes re-ranking layer between differential and working_dx.
        # When None (default), the milestone behaves exactly as before — so the node
        # can be ablated on/off by passing/omitting it.
        self._kb_rerank = kb_rerank
        # SELF-CONSISTENCY ENSEMBLE (accuracy-for-speed): when ensemble_k > 1 and a
        # higher-temperature `ensemble_gateway` is supplied, the differential node
        # samples the differential K times and reciprocal-rank-fuses them. Pure
        # variance reduction — the single most reliable lift for a WEAK model — and
        # it can only reorder the LLM's own hypotheses, never add outside knowledge.
        # k == 1 (default) or no ensemble gateway ⇒ the node behaves exactly as before.
        self._ensemble_gateway = ensemble_gateway
        self._ensemble_k = max(1, int(ensemble_k))
        # Pacing: run the K ensemble samples SEQUENTIALLY with this delay between them
        # (instead of a concurrent burst) so a throttled free-tier provider is not
        # tripped into rate-limit errors. 0.0 keeps them back-to-back (still sequential).
        self._ensemble_delay_s = max(0.0, float(ensemble_delay_s))
        # Conservative REFUTATION pass (between the differential and working_dx):
        # drop ONLY candidates the patient's own account explicitly contradicts.
        # Off by default; adds real signal without importing published-prior bias.
        self._refute = refute
        # Advanced retrieval can fan out into several controller/search calls.
        # Give only those network nodes a wider bound while the rest of the DAG
        # keeps the ordinary per-node timeout.  Appraisal is capped separately so
        # four successful paths cannot create an unbounded batch of LLM calls.
        self._retrieval_timeout_s = retrieval_timeout_s
        self._max_evidence_items = max(1, int(max_evidence_items))

    async def run(self, state: ConsultationState) -> PathResult:
        report = await self._executor.run(self._build_tasks(), state=state)
        return self._finalize(state, report)

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _build_tasks(self) -> list[Task]:
        if self._lean:
            return self._build_lean_tasks()
        tasks: list[Task] = [
            Task("differential", self._node_differential),
            Task("test_rec", self._node_test_rec, deps=("differential",)),
            Task("med_safety", self._node_med_safety, deps=("differential",)),
        ]
        have_retrievers = bool(self._retrievers)
        # The query-planner only earns its LLM call when there is something to search.
        if have_retrievers:
            tasks.append(Task("query_plan", self._node_query_plan, deps=("differential",)))
        evidence_names: list[str] = []
        for retriever in self._retrievers:
            name = f"{_EVIDENCE_PREFIX}{retriever.source_name}"
            evidence_names.append(name)
            tasks.append(
                Task(
                    name, self._make_retrieval_node(retriever),
                    deps=("differential",),
                    # The planner is best-effort: retrieval falls back to condition
                    # names if it failed, so it never blocks the source.
                    optional_deps=("query_plan",),
                    timeout=self._retrieval_timeout_s,
                )
            )
        # Appraisal + synthesis ALWAYS run (even with no retrievers) and always
        # succeed, returning empty when there is no evidence — so working_diagnosis'
        # dependency on synthesis is satisfied whether RAG is on or off.
        tasks.append(
            Task(
                "evidence_appraise", self._node_evidence_appraise,
                deps=("differential",), optional_deps=tuple(evidence_names),
            )
        )
        tasks.append(
            Task(
                "evidence_synthesis", self._node_evidence_synthesis,
                deps=("differential", "evidence_appraise"),
            )
        )
        # Optional KB re-ranking layer: sits BETWEEN differential and working_dx.
        # When enabled, working_dx requires it (and reads the re-ranked differential);
        # when disabled, working_dx depends only on the raw differential as before.
        wd_deps = ["differential", "evidence_synthesis", "med_safety"]
        # kb_rerank / refute feed working_dx but are BEST-EFFORT: if either fails OR
        # TIMES OUT (e.g. its extraction call is throttled past the node timeout), the
        # working diagnosis must still run on the raw differential rather than the whole
        # milestone collapsing. So they are OPTIONAL deps — working_dx reads whichever
        # ran (refute > kb_rerank > differential). This is the floor: a reranking layer
        # can never sink the milestone below the raw differential.
        wd_optional = ["test_rec"]
        if self._kb_rerank is not None:
            tasks.append(Task("kb_rerank", self._node_kb_rerank, deps=("differential",)))
            wd_optional.append("kb_rerank")
        if self._refute:
            # Only list kb_rerank as an optional dep when the node is actually in the
            # graph — the executor validates that every named dependency exists.
            refute_optional = ("kb_rerank",) if self._kb_rerank is not None else ()
            tasks.append(Task("refute", self._node_refute, deps=("differential",),
                              optional_deps=refute_optional))
            wd_optional.append("refute")
        tasks.append(
            Task(
                "working_diagnosis", self._node_working_dx,
                deps=tuple(wd_deps),
                optional_deps=tuple(wd_optional),
            )
        )
        if self._build_report:
            tasks.append(
                Task(
                    "report", self._node_report,
                    deps=("working_diagnosis",),
                    # The report SUMMARISES these, so it needs them as inputs. They are
                    # optional (not required): a missing test_rec/med_safety must not stop
                    # the report — it just omits that section rather than fabricating it.
                    optional_deps=("med_safety", "test_rec"),
                )
            )
        return tasks

    def _build_lean_tasks(self) -> list[Task]:
        """The minimal graph that fully determines top-k under non-destructive mode:
        differential → (optional KB fusion) → working_diagnosis. No test/med/synthesis/
        report calls. Used only for accuracy evaluation to conserve API quota."""
        tasks: list[Task] = [Task("differential", self._node_differential)]
        # kb_rerank / refute are OPTIONAL deps of working_dx: a timeout or failure of a
        # best-effort reranking node must fall back to the raw differential, never sink
        # the milestone (working_dx reads refute > kb_rerank > differential).
        wd_optional: list[str] = []
        if self._kb_rerank is not None:
            tasks.append(Task("kb_rerank", self._node_kb_rerank, deps=("differential",)))
            wd_optional.append("kb_rerank")
        if self._refute:
            # Only list kb_rerank as an optional dep when the node is actually in the
            # graph — the executor validates that every named dependency exists.
            refute_optional = ("kb_rerank",) if self._kb_rerank is not None else ()
            tasks.append(Task("refute", self._node_refute, deps=("differential",),
                              optional_deps=refute_optional))
            wd_optional.append("refute")
        tasks.append(Task("working_diagnosis", self._node_working_dx,
                          deps=("differential",), optional_deps=tuple(wd_optional)))
        return tasks

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _differential_user(self, state: ConsultationState) -> str:
        return (
            self._patient_block(state)
            + "Running differential so far:\n"
            + self._render_hypotheses(state.visit.differential)
            + "\n\nConversation:\n"
            + self._transcript(state)
        )

    async def _node_differential(self, ctx: TaskContext) -> DifferentialResult:
        state: ConsultationState = ctx.state
        if self._ensemble_k > 1 and self._ensemble_gateway is not None:
            result = await self._differential_ensemble(state)
        else:
            result = await self._gateway.complete_typed(
                DifferentialResult,
                system=_DIFFERENTIAL_PROMPT,
                user=self._differential_user(state),
                model=self._model,
            )
        if result.hypotheses:
            state.visit.differential = result.hypotheses
        return result

    async def _differential_ensemble(self, state: ConsultationState) -> DifferentialResult:
        """Sample the differential K times on the higher-temperature ensemble gateway
        and reciprocal-rank-fuse them (self-consistency). Any sample that fails is
        dropped; if ALL fail, fall back to a single greedy call on the main gateway so
        the milestone can never end up worse than one plain differential (the floor)."""
        user = self._differential_user(state)

        async def _one() -> DifferentialResult:
            return await self._ensemble_gateway.complete_typed(
                DifferentialResult, system=_DIFFERENTIAL_PROMPT, user=user, model=self._model,
            )

        # Sequential + paced (NOT asyncio.gather): a concurrent K-call burst trips
        # free-tier rate limits; spacing the samples keeps a throttled provider alive.
        outcomes: list = []
        for i in range(self._ensemble_k):
            try:
                outcomes.append(await _one())
            except Exception as exc:  # noqa: BLE001 - keep other samples + the fallback floor
                outcomes.append(exc)
            if self._ensemble_delay_s and i < self._ensemble_k - 1:
                await asyncio.sleep(self._ensemble_delay_s)
        samples = [
            r for r in outcomes if isinstance(r, DifferentialResult) and r.hypotheses
        ]
        if not samples:
            logger.warning(
                "Ensemble differential: all %d samples failed; single-call fallback.",
                self._ensemble_k,
            )
            return await self._gateway.complete_typed(
                DifferentialResult, system=_DIFFERENTIAL_PROMPT, user=user, model=self._model,
            )
        if len(samples) == 1:
            return samples[0]
        fused = fuse_llm_rankings([s.hypotheses for s in samples])
        return samples[0].model_copy(update={"hypotheses": fused})

    async def _node_kb_rerank(self, ctx: TaskContext) -> DifferentialResult:
        """Apply the curated-KB prior to the LLM differential. Two modes:
          - "rerank": light blend that only reorders the LLM's own hypotheses;
          - "fuse":   reciprocal-rank fusion with the KB Bayesian ranking, which can
                      surface a KB condition the LLM missed (coverage) while keeping
                      the LLM's top-1 protected.
        Never removes a hypothesis the LLM proposed. working_dx consumes this."""
        state: ConsultationState = ctx.state
        diff: DifferentialResult = ctx.results["differential"]
        assert self._kb_rerank is not None  # node only added when configured
        text = self._patient_block(state) + self._transcript(state)
        try:
            if getattr(self._kb_rerank, "mode", "rerank") == "fuse":
                adjusted = await self._kb_rerank.fuse(self._gateway, text, diff.hypotheses)
            else:
                adjusted = await self._kb_rerank.rerank(self._gateway, text, diff.hypotheses)
        except Exception:  # noqa: BLE001 - KB adjustment is best-effort; never lose the run
            logger.warning("KB %s failed; keeping LLM differential.",
                           getattr(self._kb_rerank, "mode", "rerank"), exc_info=True)
            adjusted = diff.hypotheses
        state.visit.differential = adjusted
        return diff.model_copy(update={"hypotheses": adjusted})

    @staticmethod
    def _same_condition(a: str, b: str) -> bool:
        """Lenient condition match (normalised; either string contains the other),
        so the refutation node can match the model's phrasing back to a hypothesis
        ('tension headache' ↔ 'tension-type headache') without an external table."""
        na = " ".join("".join(c if c.isalnum() else " " for c in (a or "").lower()).split())
        nb = " ".join("".join(c if c.isalnum() else " " for c in (b or "").lower()).split())
        if not na or not nb:
            return False
        return na == nb or na in nb or nb in na

    async def _node_refute(self, ctx: TaskContext) -> DifferentialResult:
        """Conservative refutation: drop ONLY candidates the patient's own account
        explicitly contradicts. Reads the current lead differential (KB-reranked when
        that node ran, else the raw differential), so it composes with fusion. The
        lead hypothesis is ALWAYS preserved and on any failure every hypothesis is
        kept — refutation can prune a distractor but never demote the lead or lose the
        run (the floor: top-1 stays ≥ the raw differential's top-1)."""
        state: ConsultationState = ctx.state
        diff: DifferentialResult = ctx.results.get("kb_rerank") or ctx.results["differential"]
        hyps = diff.hypotheses
        if len(hyps) < 2:
            return diff
        try:
            res = await self._gateway.complete_typed(
                RefutationResult,
                system=_REFUTE_PROMPT,
                user=(
                    "Differential:\n" + self._render_hypotheses(hyps)
                    + "\n\n" + self._patient_block(state)
                    + "Conversation:\n" + self._transcript(state)
                ),
                model=self._model,
            )
        except Exception:  # noqa: BLE001 - refutation is best-effort; never lose the run
            logger.warning("Refutation pass failed; keeping full differential.", exc_info=True)
            return diff
        remove = [r.condition for r in res.remove if r.condition]
        if not remove:
            return diff
        # Preserve the lead by construction; only NON-lead contradicted candidates drop.
        kept = [hyps[0]] + [
            h for h in hyps[1:]
            if not any(self._same_condition(h.condition, r) for r in remove)
        ]
        state.visit.differential = kept
        return diff.model_copy(update={"hypotheses": kept})

    async def _node_test_rec(self, ctx: TaskContext) -> TestRecommendationResult:
        diff: DifferentialResult = ctx.results["differential"]
        result = await self._gateway.complete_typed(
            TestRecommendationResult,
            system=_TEST_REC_PROMPT,
            user="Differential:\n" + self._render_hypotheses(diff.hypotheses),
            model=self._model,
        )
        ctx.state.visit.recommended_tests = result.tests
        return result

    async def _node_med_safety(self, ctx: TaskContext) -> MedicationSafetyResult:
        diff: DifferentialResult = ctx.results["differential"]
        state: ConsultationState = ctx.state
        meds = ", ".join(m.name for m in state.patient.active_medications) or "none recorded"
        allergies = ", ".join(a.substance for a in state.patient.allergies) or "none recorded"
        return await self._gateway.complete_typed(
            MedicationSafetyResult,
            system=_MED_SAFETY_PROMPT,
            user=(
                f"Active medications: {meds}\n"
                f"Allergies: {allergies}\n\n"
                "Candidate conditions:\n" + self._render_hypotheses(diff.hypotheses)
            ),
            model=self._model,
        )

    async def _node_query_plan(self, ctx: TaskContext) -> QueryPlanResult:
        """Agent 1: turn the LEADING hypotheses into focused, discriminating
        literature queries (replacing the old OR-of-condition-names blob).

        Only hypotheses with a meaningful prior are searched. Querying a
        2%-confidence hypothesis the interview already all but ruled out just
        injects high-quality-but-irrelevant literature about that condition, which
        the synthesis then mistakes for diagnostic support — the failure mode that
        let a 0.02 prior become an 85%-confident wrong working diagnosis."""
        diff: DifferentialResult = ctx.results["differential"]
        hyps = self._leading_hypotheses(diff)
        if not hyps:
            return QueryPlanResult()
        return await self._gateway.complete_typed(
            QueryPlanResult,
            system=_QUERY_PLAN_PROMPT,
            user=(
                "Differential:\n" + self._render_hypotheses(hyps)
                + "\n\nWrite one focused query per hypothesis that would best "
                "confirm it or distinguish it from the others."
            ),
            model=self._model,
        )

    def _make_retrieval_node(self, retriever: Retriever):
        async def node(ctx: TaskContext) -> list[EvidenceItem]:
            diff: DifferentialResult = ctx.results["differential"]
            plan: QueryPlanResult | None = ctx.results.get("query_plan")
            queries = self._queries_from_plan(plan, diff)
            # Run the planned queries concurrently; tolerate partial failure but
            # preserve the "retrieval failed" signal when every query errors.
            results = await asyncio.gather(
                *(retriever.search(q) for q in queries), return_exceptions=True
            )
            items: list[EvidenceItem] = []
            errors: list[BaseException] = []
            for r in results:
                if isinstance(r, BaseException):
                    errors.append(r)
                else:
                    items.extend(r)
            # Targeted queries can be too narrow and return nothing. Rather than let
            # RAG silently go dark, fall back to a broad OR-of-conditions search —
            # so evidence still flows whenever the literature plainly exists.
            if not items and plan and plan.queries:
                fallback = EvidenceQuery(
                    terms=[h.condition for h in self._leading_hypotheses(diff)]
                )
                try:
                    items = await retriever.search(fallback)
                    logger.info(
                        "%s: planned queries returned nothing; broad fallback found %d.",
                        retriever.source_name, len(items),
                    )
                except Exception as exc:  # noqa: BLE001 - recorded as a node failure
                    errors.append(exc)
            if errors and not items:
                raise errors[0]
            return [self._with_source(item, retriever.source_name) for item in items]
        return node

    async def _node_evidence_appraise(self, ctx: TaskContext) -> list[PaperAppraisal]:
        """Agent 2 (fan-out): appraise each retrieved paper on its own — one LLM
        call per paper, concurrently — reading the full abstract and extracting the
        concrete finding it contributes. Writes the kept, scored items to the
        blackboard and returns the per-paper appraisals for synthesis."""
        state: ConsultationState = ctx.state
        items = self._dedup(self._gather_evidence(ctx))[: self._max_evidence_items]
        if not items:
            state.visit.evidence = []
            return []

        diff: DifferentialResult = ctx.results["differential"]
        patient_desc = self._patient_block(state).strip()
        sem = asyncio.Semaphore(_APPRAISE_CONCURRENCY)

        async def appraise(item: EvidenceItem) -> tuple[EvidenceItem, PaperAppraisal] | None:
            async with sem:
                try:
                    appraisal = await self._gateway.complete_typed(
                        PaperAppraisal,
                        system=_APPRAISE_PROMPT,
                        user=self._render_one_paper(item, diff.hypotheses, patient_desc),
                        model=ModelTier.FAST,  # per-paper extraction is FAST-tier work
                    )
                except LLMError:
                    return None  # one paper failing never sinks the batch
            if not appraisal.keep:
                return None
            # Trust OUR identifier, not the model's echo of it.
            appraisal = appraisal.model_copy(update={"identifier": item.identifier})
            scored = item.model_copy(update={
                "quality_score": appraisal.quality_score,
                "quality_notes": appraisal.key_finding or appraisal.notes,
            })
            return scored, appraisal

        appraised = await asyncio.gather(*(appraise(it) for it in items))
        kept = [pair for pair in appraised if pair is not None]
        kept.sort(key=lambda pair: pair[0].quality_score or 0.0, reverse=True)
        state.visit.evidence = [item for item, _ in kept]
        return [appraisal for _, appraisal in kept]

    async def _node_evidence_synthesis(self, ctx: TaskContext) -> EvidenceSynthesisResult:
        """Agent 3: combine the per-paper appraisals into a decision-ready brief,
        organised by hypothesis. This is what grounds the working diagnosis."""
        appraisals: list[PaperAppraisal] = ctx.results.get("evidence_appraise") or []
        if not appraisals:
            return EvidenceSynthesisResult(overall="no external evidence retrieved")
        diff: DifferentialResult = ctx.results["differential"]
        items_by_id = {e.identifier: e for e in ctx.state.visit.evidence}
        return await self._gateway.complete_typed(
            EvidenceSynthesisResult,
            system=_SYNTHESIS_PROMPT,
            user=(
                "Differential:\n" + self._render_hypotheses(diff.hypotheses)
                + "\n\nAppraised papers:\n"
                + self._render_appraisals(appraisals, items_by_id)
            ),
            model=self._model,
        )

    async def _node_working_dx(self, ctx: TaskContext) -> WorkingDiagnosis:
        state: ConsultationState = ctx.state
        # Read the FINAL lead differential: refutation (if it ran) sits after KB
        # fusion, which sits after the raw differential. Each is best-effort, so fall
        # back down the chain — the raw differential is always present.
        diff: DifferentialResult = (
            ctx.results.get("refute")
            or ctx.results.get("kb_rerank")
            or ctx.results["differential"]
        )
        # synthesis and med_safety are optional: the lean accuracy-eval graph skips
        # them (they do not change the ranking under the non-destructive rule), so read
        # them defensively rather than requiring them.
        synthesis: EvidenceSynthesisResult | None = ctx.results.get("evidence_synthesis")
        med: MedicationSafetyResult | None = ctx.results.get("med_safety")
        test_rec: TestRecommendationResult | None = ctx.results.get("test_rec")

        # Flatten med_safety concerns in case smaller models returned dicts.
        if med is None:
            med_line = "none"
        elif med.summary:
            med_line = med.summary
        elif med.concerns:
            flat = [c.get("condition") or str(c) if isinstance(c, dict) else str(c)
                    for c in med.concerns]
            med_line = "concerns: " + "; ".join(flat)
        else:
            med_line = "none"

        result = await self._gateway.complete_typed(
            WorkingDiagnosisResult,
            system=_WORKING_DX_PROMPT,
            user=(
                "Differential:\n" + self._render_hypotheses(diff.hypotheses)
                + "\n\nExternal evidence synthesis (from the medical literature, "
                "by hypothesis):\n" + self._render_synthesis(synthesis)
                + "\n\nMedication-safety: " + med_line
                + "\n\nRecommended tests:\n"
                + (self._render_tests(test_rec.tests) if test_rec else "  (none available)")
            ),
            model=self._model,
        )

        # Deterministic safety floor: high stakes or low confidence REQUIRE a
        # concrete confirmation pathway, regardless of what the model returned.
        pathway = (result.confirmation_pathway or "").strip()
        needs_pathway = (
            result.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL)
            or result.confidence < self._closure_floor
        )
        if needs_pathway and not pathway:
            pathway = (
                "Please see a licensed clinician in person to confirm this before "
                "acting on it."
            )

        # NON-DESTRUCTIVE rule: unless evidence-override is explicitly enabled, the
        # lead is the top of the (fused/reranked) differential — the evidence step
        # cannot demote an already-correct lead. It still supplies the confidence,
        # risk, reasoning and pathway below.
        if self._evidence_can_override:
            condition = result.condition or (
                diff.hypotheses[0].condition if diff.hypotheses else "Uncertain"
            )
        else:
            condition = (
                diff.hypotheses[0].condition if diff.hypotheses
                else (result.condition or "Uncertain")
            )
        working = WorkingDiagnosis(
            condition=condition,
            confidence=result.confidence,
            risk_tier=result.risk_tier,
            reasoning=result.reasoning,
            differential=self._rerank_differential(condition, result.confidence, diff.hypotheses),
            evidence=list(state.visit.evidence),
            confirmation_pathway=pathway,
        )
        state.visit.working_diagnosis = working
        return working

    async def _node_report(self, ctx: TaskContext) -> Report:
        state: ConsultationState = ctx.state
        working: WorkingDiagnosis = ctx.results["working_diagnosis"]
        # The report prompt asks the model to summarise the tests, the appraised
        # evidence, and any medication-safety concerns — so it must be GIVEN them.
        # They already exist on the blackboard / as upstream node results; this node
        # was previously not passing them, which made the model omit or invent those
        # sections of the document a clinician signs.
        med: MedicationSafetyResult | None = ctx.results.get("med_safety")
        test_rec: TestRecommendationResult | None = ctx.results.get("test_rec")

        if med is None:
            med_line = "not assessed"
        elif med.summary:
            med_line = med.summary
        elif med.concerns:
            # Flatten concerns: smaller models may return dicts {condition, explanation}
            # instead of plain strings; convert to a readable list.
            flat = []
            for c in med.concerns:
                if isinstance(c, dict):
                    flat.append(c.get("condition") or str(c))
                else:
                    flat.append(str(c))
            med_line = "; ".join(flat)
        else:
            med_line = "no concerns identified"
        tests = test_rec.tests if test_rec is not None else state.visit.recommended_tests

        result = await self._gateway.complete_typed(
            ReportResult,
            system=_REPORT_PROMPT,
            user=(
                self._patient_block(state)
                + "Working diagnosis: " + working.condition
                + f" (confidence {working.confidence:.2f}, harm-if-wrong "
                + f"{working.risk_tier.value})\n"
                + "Reasoning: " + working.reasoning + "\n"
                + "Confirmation pathway: " + working.confirmation_pathway + "\n"
                + "Differential:\n" + self._render_hypotheses(working.differential)
                + "\n\nRecommended tests:\n" + self._render_tests(tests)
                + "\n\nMedication-safety: " + med_line
                + "\n\nAppraised evidence (highest-quality first):\n"
                + (self._render_evidence(working.evidence[:5])
                   if working.evidence else "  (none retained)")
            ),
            model=self._model,
        )
        report = Report(
            status=ReportStatus.PENDING_APPROVAL,
            content=result.content,
            generated_at=_utcnow(),
        )
        state.visit.report = report
        return report

    # ── Finalisation → patient-facing reply ───────────────────────────────────

    def _finalize(self, state: ConsultationState, report: ExecutionReport) -> PathResult:
        working = report.result("working_diagnosis")
        if not report.succeeded("differential") or working is None:
            failed = ", ".join(report.failures) or "unknown"
            logger.warning("Milestone could not complete (failed/skipped: %s)", failed)
            # Graceful, safe terminal: hand off to human care rather than retry-loop.
            state.session.phase = DiagnosticPhase.COMPLETE
            return PathResult(
                reply=(
                    "I'm sorry — I wasn't able to complete a full assessment right "
                    "now. Please follow up with a licensed clinician, and seek "
                    "urgent care if your symptoms get worse."
                ),
                transition=Transition.stay(),
            )

        state.session.phase = DiagnosticPhase.COMPLETE
        return PathResult(reply=_compose_patient_reply(working), transition=Transition.stay())

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _gather_evidence(self, ctx: TaskContext) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for name, value in ctx.results.items():
            if name.startswith(_EVIDENCE_PREFIX) and isinstance(value, list):
                items.extend(value)
        return items

    def _rerank_differential(
        self, condition: str, confidence: float, hypotheses: list[Hypothesis]
    ) -> list[Hypothesis]:
        """Return the differential with the CHOSEN working diagnosis at the front, so
        `working.differential[0]` always agrees with `working.condition`. The chosen
        hypothesis carries the (evidence-adjusted) working confidence; the remaining
        hypotheses follow in descending confidence. This makes the differential the
        single ranked source of truth — for the report and for any top-k scoring —
        instead of leaving `condition` and the differential's head disagreeing after
        the evidence step has re-picked the lead."""
        ranked = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
        norm = condition.strip().casefold()
        lead: Hypothesis | None = None
        remaining: list[Hypothesis] = []
        for h in ranked:
            if lead is None and h.condition.strip().casefold() == norm:
                lead = h.model_copy(update={"confidence": confidence})
            else:
                remaining.append(h)
        if lead is None:
            # The model named a condition not literally in the differential — make it
            # the lead so the stored list stays consistent with the chosen condition.
            lead = Hypothesis(condition=condition, confidence=confidence)
        return [lead, *remaining]

    def _leading_hypotheses(self, diff: DifferentialResult):
        """The hypotheses worth searching: ranked by prior confidence, above the
        minimum-confidence floor, capped at max_query_hypotheses. Searching only the
        leading hypotheses keeps RAG from dredging up irrelevant literature for
        conditions the interview already dismissed."""
        ranked = sorted(diff.hypotheses, key=lambda h: h.confidence, reverse=True)
        leading = [h for h in ranked if h.confidence >= self._min_query_confidence]
        return leading[: self._max_query_hypotheses]

    def _queries_from_plan(
        self, plan: QueryPlanResult | None, diff: DifferentialResult
    ) -> list[EvidenceQuery]:
        """The planned queries if any survived, else the OR-of-conditions fallback."""
        if plan and plan.queries:
            queries = [
                EvidenceQuery(query_string=pq.query.strip())
                for pq in plan.queries
                if pq.query and pq.query.strip()
            ]
            if queries:
                return queries
        terms = [h.condition for h in self._leading_hypotheses(diff)]
        return [EvidenceQuery(terms=terms)] if terms else []

    def _dedup(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        """Merge identical hits and retain every RAG-path provenance label."""
        seen: dict[str, EvidenceItem] = {}
        for item in items:
            existing = seen.get(item.identifier)
            if existing is None:
                seen[item.identifier] = item
                continue
            sources = [
                value.strip()
                for source in (existing.source, item.source)
                for value in source.split("|")
                if value.strip()
            ]
            merged = " | ".join(dict.fromkeys(sources))
            seen[item.identifier] = existing.model_copy(update={"source": merged})
        return list(seen.values())

    def _with_source(self, item: EvidenceItem, source: str) -> EvidenceItem:
        if item.source:
            return item
        return item.model_copy(update={"source": source})

    def _patient_block(self, state: ConsultationState) -> str:
        p = state.patient
        age = str(p.age) if p.age is not None else "unknown"
        conditions = ", ".join(p.chronic_conditions) or "none recorded"
        return (
            "Patient: "
            f"age {age}, sex {p.sex.value}, chronic conditions: {conditions}.\n\n"
        )

    def _transcript(self, state: ConsultationState, limit: int = 20) -> str:
        msgs = state.session.transcript[-limit:]
        return "\n".join(f"{m.role.value}: {m.content}" for m in msgs) or "none"

    def _render_hypotheses(self, hyps) -> str:
        if not hyps:
            return "  (none)"
        return "\n".join(
            f"  - {h.condition} (confidence {h.confidence:.2f})" for h in hyps
        )

    def _render_tests(self, tests) -> str:
        if not tests:
            return "  (none)"
        return "\n".join(f"  - {t.name}: {t.rationale}" for t in tests)

    def _render_evidence(self, items: list[EvidenceItem]) -> str:
        lines = []
        for it in items:
            year = it.published_year or "n/a"
            stype = it.study_type or "n/a"
            # Prefer the appraisal's extracted finding (set during appraisal) over a
            # raw abstract slice — it is the grounded, decision-relevant bit.
            detail = it.quality_notes or (it.summary[:200] if it.summary else "")
            tail = f" — {detail}" if detail else ""
            lines.append(f"  - [{it.identifier}] {it.title} ({year}, {stype}){tail}")
        return "\n".join(lines)

    def _render_one_paper(self, item: EvidenceItem, hyps, patient_desc: str = "") -> str:
        conditions = ", ".join(h.condition for h in hyps) or "the differential"
        year = item.published_year or "n/a"
        stype = item.study_type or "n/a"
        patient_line = f"{patient_desc}\n\n" if patient_desc else ""
        return (
            f"{patient_line}"
            f"Candidate conditions: {conditions}\n\n"
            f"Paper [{item.identifier}] {item.title} ({year}, {stype})\n"
            f"Abstract: {item.summary or '(no abstract available)'}"
        )

    def _render_appraisals(self, appraisals: list[PaperAppraisal], items_by_id) -> str:
        lines = []
        for a in appraisals:
            item = items_by_id.get(a.identifier)
            title = item.title if item is not None else ""
            supports = ", ".join(a.supports) or "-"
            refutes = ", ".join(a.refutes) or "-"
            lines.append(
                f"  - [{a.identifier}] {title} (score {a.quality_score:.2f}; "
                f"supports: {supports}; refutes: {refutes}) "
                f"finding: {a.key_finding or a.notes or '(none)'}"
            )
        return "\n".join(lines) or "  (none)"

    def _render_synthesis(self, synthesis: EvidenceSynthesisResult) -> str:
        if not synthesis or not synthesis.by_hypothesis:
            overall = synthesis.overall if synthesis else "none"
            return f"  (no external evidence) {overall}"
        lines = []
        for h in synthesis.by_hypothesis:
            supporting = ", ".join(h.supporting_refs) or "-"
            refuting = ", ".join(h.refuting_refs) or "-"
            lines.append(
                f"  - {h.condition} [{h.strength or 'n/a'}]: "
                f"supporting {supporting}; refuting {refuting} — {h.key_finding}"
            )
        if synthesis.contradictions:
            lines.append(f"  contradictions: {synthesis.contradictions}")
        if synthesis.overall:
            lines.append(f"  overall: {synthesis.overall}")
        return "\n".join(lines)


def _compose_patient_reply(working: WorkingDiagnosis) -> str:
    """Deterministically build the patient-facing summary from the working
    diagnosis. Kept rule-based so the safety-critical wording is never
    hallucinated, and so it always names the confirmation step and the clinician
    review."""
    lines = [
        f"Based on what you've told me, the most likely explanation is {working.condition}."
    ]
    if working.confidence < 0.4:
        lines.append("I want to be honest that I'm not certain about this.")
    elif working.confidence < 0.7:
        lines.append("There is still a fair amount of uncertainty, so it's worth confirming.")
    if working.confirmation_pathway:
        lines.append(f"The clearest next step is: {working.confirmation_pathway}")
    if working.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL):
        lines.append(
            "Because this could be serious, please arrange to be seen by a clinician "
            "promptly rather than waiting."
        )
    lines.append(
        "A licensed clinician will review this assessment before it is considered final."
    )
    return " ".join(lines)

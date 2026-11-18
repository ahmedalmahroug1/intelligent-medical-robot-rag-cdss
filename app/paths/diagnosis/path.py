"""
app/paths/diagnosis/path.py
---------------------------
The DIAGNOSIS path: the first and deepest clinical path. It is an ordinary peer —
it implements the same `Path` contract as every other path and registers with the
same router, with no special status. The router routes to it like any other.

This file builds the CHEAP LOOP — the ~80%-of-turns path that is one LLM call:

    Clinical Understanding → Hypothesis Generation → Sufficiency Check
        ── not sufficient ──► Question Planning ──► ask one question (Interview)
        ── sufficient ─────► hand off to the MILESTONE

All four of those steps happen inside a SINGLE structured call that returns an
`IntakeResult` (understanding + updated differential + sufficiency + next
question). Doing it in one call is what makes the loop cheap relative to the
milestone's evidence fan-out — "cheap" here means one call, not the cheapest
model (see the note on model tier below).

The deterministic sufficiency gate (the safety-relevant part) sits ON TOP of the
model's own `sufficient` flag:

    - confidence floor — do not converge on a weak top hypothesis just because the
      model is impatient; keep gathering until the leading hypothesis clears a
      minimum confidence, or the cap below forces a decision;
    - iteration cap — the interview can never loop forever; once the cap is hit
      the path converges regardless of the model, and the milestone (which is
      confidence- and risk-aware) produces a "recommend a test / see a clinician"
      outcome rather than a false closure.

The MILESTONE — differential → test recommendation → parallel evidence retrieval →
ranking/verification → medication-safety → calibrated, risk-gated working
diagnosis → report — is NOT built here. It is File 7 (it ports the V2 executor as
the engine). `_run_milestone` is the single seam where it drops in; this file
takes the conversation up to the point of convergence and no further.

Phases: this path drives DiagnosticPhase between INTAKE/INTERVIEW (looping) and
MILESTONE (converged). It never handles ESCALATED — the safety guard sits above
the router and deals with emergencies before a turn ever reaches a path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.core.enums import DiagnosticPhase, Route
from app.llm.base import LLMGateway, ModelTier
from app.paths.diagnosis.reasoning import reason_over_context
from app.paths.diagnosis.schemas import IntakeResult
from app.routing.contract import Path, PathResult, Transition

if TYPE_CHECKING:  # type-only; the path accesses state attributes at runtime
    from app.core.state import ConsultationState, SessionState, VisitState
    from app.paths.diagnosis.milestone import Milestone

logger = logging.getLogger(__name__)


# ── Tunables (constructor-overridable so tests can exercise the gate quickly) ──
DEFAULT_MAX_INTERVIEW_ITERATIONS = 8     # hard cap on interview turns before forcing the milestone
DEFAULT_CONFIDENCE_FLOOR = 0.5           # minimum top-hypothesis confidence to honour a "sufficient"

# Used when the gate decides to keep interviewing but the model — having called
# the case sufficient — offered no next question. A safe, open probe.
_FALLBACK_QUESTION = (
    "Could you tell me a little more about what you're experiencing — anything "
    "else you've noticed, even if it seems minor?"
)

# Said at the moment of convergence, while the (not-yet-built) milestone would run.
# Deliberately frames the system honestly: an assessment follows, and a clinician
# reviews it before it means anything.
_MILESTONE_BRIDGE = (
    "Thank you — I have enough to work through an assessment now. I'll look at the "
    "most likely explanations and what would help confirm them. A clinician then "
    "reviews the result before it's considered final."
)

_SYSTEM_PROMPT = """You are the interviewing component of a medical \
decision-support assistant, talking with a patient to understand their problem. \
You are NOT giving a final diagnosis and you are NOT a substitute for a clinician: \
a licensed clinician reviews every assessment before it carries any authority, and \
a separate safety system handles emergencies.

Each turn, do all of the following:
1. Update your clinical understanding of the case from the conversation so far and \
the patient's latest message.
2. Maintain a working differential — up to 8-10 plausible conditions ranked by a \
calibrated confidence between 0 and 1: the leading hypotheses first, then \
less-likely but clinically important alternatives as an honestly LOW-confidence \
tail. For each, give the findings that support it and the findings that argue \
against it. Be honest; do not inflate confidence, and let the confidences reflect \
real uncertainty. Weigh the WHOLE picture: a single non-specific finding (e.g. \
mucus colour, one low-grade symptom) must NOT override a hypothesis the rest of the \
presentation still supports — let confidence move gradually unless a finding is \
genuinely decisive.
3. Decide whether you have enough information to STOP asking questions and move on \
to a fuller assessment ("sufficient"). Lean towards gathering a little more when a \
single well-chosen question could meaningfully change the differential, but do not \
prolong the interview once the picture is reasonably clear.
4. If not sufficient, choose what to ask next, in this order of priority:
   (a) EARLY — when you have no differential yet, or only the patient's first \
message — ask the patient to enumerate, in their own words, ALL the symptoms they \
have noticed and when each started (e.g. "What symptoms have you noticed? Please \
tell me everything — every symptom, even small ones."), so you capture the full \
picture in one go before narrowing. Do NOT fire a narrow yes/no question yet, and \
do not settle for the first symptom they mention — invite the complete list.
   (b) BEFORE you start narrowing, take the brief history a patient will NOT \
volunteer on their own: their relevant ONGOING medical conditions and any regular \
MEDICATIONS or inhalers they take (and allergies). If these are still unknown, ask \
about them — they are frequently DECISIVE (a known asthma/inhaler history, or a \
drug that itself causes the presentation, e.g. an acute dystonic reaction). Assume \
nothing is in the record unless it was stated.
   (c) Once you have the picture AND that history, ask the SINGLE most informative \
question: the one whose answer would best separate your leading competing \
hypotheses.
   Ask exactly one question, in plain, warm language a patient can answer, avoiding \
medical jargon.

Tone and safety:
- Be warm, clear, and respectful. One question at a time.
- Never suggest the patient delay or avoid seeing a clinician for something \
potentially serious.
- Do not assert a definitive diagnosis in the question text — you are still \
gathering information.

Return ONLY a JSON object with exactly these keys:
{
  "understanding": "one or two sentences summarising the case so far",
  "chief_complaint": "a short phrase naming the main problem, or null if unclear",
  "hypotheses": [
    {"condition": "name", "confidence": 0.0, "supporting": ["..."], "against": ["..."]}
  ],
  "sufficient": true or false,
  "sufficiency_rationale": "one sentence on why you can or cannot stop yet",
  "next_question": "the single next question for the patient, or null if sufficient"
}"""


class DiagnosisPath(Path):
    """The diagnostic interview loop, as a router-registered Path peer."""

    route = Route.DIAGNOSIS

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        model: ModelTier = ModelTier.REASONING,
        max_interview_iterations: int = DEFAULT_MAX_INTERVIEW_ITERATIONS,
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
        milestone: Milestone | None = None,
    ) -> None:
        # REASONING by default: the interview maintains a differential and judges
        # sufficiency on real medical input, where answer quality matters more
        # than shaving latency. It is still ONE call per turn — the "cheap" in the
        # two-speed design is one-call-vs-milestone-fan-out, not the cheapest
        # model. Override to FAST if latency/cost wins for a given deployment.
        #
        # `milestone` is the run-once evidence pipeline (File 7b). It is optional:
        # without it, convergence returns the honest bridge line below (useful for
        # testing the loop in isolation); with it, convergence runs the pipeline.
        self._gateway = gateway
        self._model = model
        self._max_iterations = max_interview_iterations
        self._confidence_floor = confidence_floor
        self._milestone = milestone

    async def handle(self, *, turn: str, state: ConsultationState) -> PathResult:
        session = state.session

        # Re-entry once converged: the milestone owns the turn, not the loop.
        if session.phase is DiagnosticPhase.MILESTONE:
            return await self._run_milestone(state)
        if session.phase is DiagnosticPhase.COMPLETE:
            return self._already_complete_reply()

        # ── The cheap loop: one structured call does all four steps. ──────────
        session.interview_iterations += 1
        result = await reason_over_context(
            self._gateway,
            system_prompt=_SYSTEM_PROMPT,
            context=self._build_user_prompt(turn, state),
            model=self._model,
        )

        # Write the model's contribution onto the blackboard.
        state.visit.differential = result.hypotheses
        if result.chief_complaint and not state.visit.chief_complaint:
            state.visit.chief_complaint = result.chief_complaint

        # ── Deterministic sufficiency gate on top of the model's view. ────────
        top_confidence = max((h.confidence for h in result.hypotheses), default=0.0)
        hit_cap = session.interview_iterations >= self._max_iterations
        model_ready = result.sufficient and top_confidence >= self._confidence_floor

        if model_ready or hit_cap:
            if hit_cap and not model_ready:
                logger.info(
                    "Diagnosis interview hit the iteration cap (%d) before the "
                    "model was confident; converging to the milestone anyway.",
                    self._max_iterations,
                )
            session.phase = DiagnosticPhase.MILESTONE
            return await self._run_milestone(state)

        # Keep interviewing: ask the single most informative next question.
        session.phase = DiagnosticPhase.INTERVIEW
        question = result.next_question or _FALLBACK_QUESTION
        return PathResult(reply=question, transition=Transition.stay())

    # ── The milestone seam (File 7 drops in here) ─────────────────────────────

    async def _run_milestone(self, state: ConsultationState) -> PathResult:
        """
        The convergence step. When a `Milestone` is wired in (File 7b), this runs
        the run-once evidence pipeline — differential → test recommendation →
        parallel evidence retrieval → ranking/verification → medication-safety →
        calibrated, risk-gated working diagnosis → draft report — and returns the
        patient-facing summary it produces.

        When no milestone is configured, it returns an honest bridge line and
        leaves phase = MILESTONE. That keeps the loop testable in isolation and
        means wiring the pipeline is a constructor argument, not a code change to
        the loop or the router.
        """
        if self._milestone is None:
            return PathResult(reply=_MILESTONE_BRIDGE, transition=Transition.stay())
        return await self._milestone.run(state)

    # ── Small replies / prompt construction ───────────────────────────────────

    def _already_complete_reply(self) -> PathResult:
        return PathResult(
            reply=(
                "Your assessment is complete and is awaiting review by a clinician. "
                "If something new has come up, tell me and we can start from there."
            ),
            transition=Transition.stay(),
        )

    def _build_user_prompt(self, turn: str, state: ConsultationState) -> str:
        patient = state.patient
        visit = state.visit
        session = state.session

        age = str(patient.age) if patient.age is not None else "unknown"
        conditions = ", ".join(patient.chronic_conditions) or "none recorded"
        meds = ", ".join(m.name for m in patient.active_medications) or "none recorded"
        allergies = ", ".join(a.substance for a in patient.allergies) or "none recorded"

        differential = self._render_differential(visit)
        # Show the WHOLE interview, not just the last few turns: a deeper interview
        # (higher iteration cap) only helps if the diagnoser still sees the early
        # answers when it re-ranks — otherwise more questions lose the first symptoms.
        # ~2 messages per turn (patient + question), plus headroom.
        transcript = self._recent_transcript(session, limit=2 * self._max_iterations + 4)

        return (
            "Patient context:\n"
            f"  Age: {age}\n"
            f"  Sex: {patient.sex.value}\n"
            f"  Chronic conditions: {conditions}\n"
            f"  Current medications: {meds}\n"
            f"  Allergies: {allergies}\n\n"
            "Current working differential (update it; do not start from scratch):\n"
            f"{differential}\n\n"
            "Recent conversation (older to newer):\n"
            f"{transcript or 'none yet'}\n\n"
            f"Patient's latest message:\n\"{turn}\"\n\n"
            f"This is interview turn {session.interview_iterations} of at most "
            f"{self._max_iterations}. Respond with the JSON object."
        )

    def _render_differential(self, visit: VisitState) -> str:
        hyps = visit.differential
        if not hyps:
            return "  (none yet — this may be the first turn)"
        return "\n".join(
            f"  - {h.condition} (confidence {h.confidence:.2f})" for h in hyps
        )

    def _recent_transcript(self, session: SessionState, limit: int = 8) -> str:
        messages = session.transcript[-limit:]
        return "\n".join(f"{m.role.value}: {m.content}" for m in messages)

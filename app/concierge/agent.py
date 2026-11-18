"""
app/concierge/agent.py
----------------------
`ConciergeAgent` — OWNS the entire patient conversation and is the only visible persona. The
patient never talks to the CDSS directly. The concierge:

  * runs the interview itself (asks its own and rephrased-engine questions),
  * accumulates a structured `ClinicalPicture` across turns,
  * CONSULTS the CDSS reasoning engine (`ReasoningFacade.assess`) for the differential, the
    engine's suggested question, and the engine's sufficiency view,
  * applies SHARED sufficiency: the interview ends only when the engine is confident AND the
    concierge judges the information complete (no critical field missing),
  * handles meta / small-talk itself (never forwarded to the engine).

It does NOT diagnose, interpret tests, or make safety decisions — those belong to the CDSS.
An injected adapter runs the shared SafetyGuard before clinical extraction, allowing the
Concierge to present escalation advice without owning the emergency policy.

Stage A ends the interview with a bridge line; Stage B will replace that with the presented
working diagnosis / report from `facade.finalize`.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter

from app.concierge import prompts
from app.concierge.identity import (
    IdentityRegistry, detect_language, explicit_language_switch, parse_name,
)
from app.concierge.reasoning import ReasoningFacade
from app.concierge.schemas import (
    AssessResult, ClinicalPicture, FinalizeResult, IdentityParse, IntakeExtraction,
)
from app.llm.base import LLMError, LLMGateway, ModelTier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Turn:
    """What one patient turn decided, BEFORE the reply text is generated. Splitting the decision
    from the reply generation lets `respond()` (one complete string; the robot's path) and
    `respond_stream()` (chunks as they are generated; the web path) share the exact same flow —
    identity, extraction, engine consultation, shared sufficiency — with no duplicated logic."""
    kind: str                       # "static" | "smalltalk" | "question" | "present"
    reply: str = ""                 # the finished reply, for "static" turns only
    suggested: str = ""             # engine's suggested question, for "question" turns
    missing: tuple[str, ...] = ()   # concierge's missing fields, for "question" turns


@dataclass(frozen=True)
class ConversationSnapshot:
    """A read-only, public view of the conversation for persistence — so a caller (the robot or
    the web layer) can save the visit WITHOUT reaching into the agent's private fields."""
    name: str | None
    patient_id: str | None
    language: str
    complete: bool
    picture: ClinicalPicture
    transcript: list[tuple[str, str]]      # (role, text), role ∈ {"patient","assistant"}
    result: FinalizeResult | None          # the CDSS finalize result, once produced


class ConciergeAgent:
    """One conversation, owned end to end by the concierge."""

    def __init__(
        self,
        *,
        facade: ReasoningFacade,
        gateway: LLMGateway,
        identity: IdentityRegistry | None = None,
        safety_check: Callable[[str, ClinicalPicture], Awaitable[str | None]] | None = None,
        model: ModelTier = ModelTier.REASONING,
        fast_model: ModelTier = ModelTier.FAST,
    ) -> None:
        self._facade = facade
        self._gateway = gateway
        self._identity = identity or IdentityRegistry()
        self._safety_check = safety_check
        # REASONING tier is reserved for the medical reasoning (assess/finalize via the facade) and
        # the final presentation. The Concierge's own conversational calls — intake extraction,
        # question phrasing, small-talk, name parsing — are simple text transforms that run on the
        # FAST tier, so a turn is not three slow reasoning-model calls in a row (latency fix).
        self._model = model
        self._fast = fast_model

        self._identified = False
        self._name: str | None = None
        self._patient_id: str | None = None
        self._language = "en"
        # LANGUAGE LOCK: once True, the reply language never follows the message's script
        # again — only an explicit patient request (or preload from the record) changes it.
        # Following the script per turn made the model flip languages mid-interview.
        self._language_locked = False
        self._picture = ClinicalPicture()
        self._history: list[str] = []          # "patient: …" / "assistant: …"
        self._last_assess: AssessResult | None = None
        self._complete = False
        self._result = None                    # the CDSS finalize result, once produced

    def preload(
        self,
        *,
        name: str,
        patient_id: str | None = None,
        picture: ClinicalPicture | None = None,
        language: str | None = None,
    ) -> None:
        """Seed a KNOWN patient before the interview begins. The robot identifies the patient
        by face/name and loads their record from persistence, then calls this so the concierge
        does not re-ask the name and does not re-ask the engine for what the record already
        holds (age, sex, medications, allergies, chronic conditions). This is the PUBLIC seam
        for that hand-off, so callers never reach into the agent's private fields.

        `language` is the patient's SAVED preference ("ar"/"en") from their record: a returning
        patient continues in the language they chose before, locked from the first word."""
        self._identified = True
        self._name = name
        self._patient_id = patient_id
        if picture is not None:
            self._picture = picture
        if language in ("ar", "en"):
            self._language = language
            self._language_locked = True

    def snapshot(self) -> ConversationSnapshot:
        """A public, read-only view of the conversation (identity, accumulated picture, full
        transcript, and the finalize result). The persistence layer uses this to save the visit,
        so it never reaches into the agent's private fields."""
        turns: list[tuple[str, str]] = []
        for line in self._history:
            role, _, text = line.partition(": ")
            turns.append((role, text))
        return ConversationSnapshot(
            name=self._name, patient_id=self._patient_id, language=self._language,
            complete=self._complete, picture=self._picture, transcript=turns,
            result=self._result)

    def greeting(self) -> str:
        return prompts.greeting()

    async def respond(self, user_text: str) -> str:
        """One turn → one complete reply string (the robot's voice path needs the whole text
        before speaking). The flow itself lives in `_decide`, shared with `respond_stream`."""
        turn = await self._decide(user_text)
        if turn.kind == "static":
            return turn.reply
        if turn.kind == "smalltalk":
            return self._record(await self._smalltalk())
        if turn.kind == "present":
            return self._record(await self._present_result())
        return self._record(await self._compose_question(turn.suggested, list(turn.missing)))

    async def respond_stream(self, user_text: str) -> AsyncIterator[str]:
        """One turn → the reply as chunks, yielded WHILE the model generates them (the web
        channel's latency fix: the patient reads the first words instead of watching a spinner).
        Same flow as `respond` (`_decide`); only the reply generation streams. The full reply is
        recorded to history exactly once, when the stream ends."""
        turn = await self._decide(user_text)
        if turn.kind == "static":
            yield turn.reply
            return
        if turn.kind == "smalltalk":
            async for chunk in self._stream_reply(
                system=prompts.smalltalk_sys(self._language),
                user=self._smalltalk_ctx(),
                fallback=prompts.smalltalk_fallback(self._language),
            ):
                yield chunk
            return
        if turn.kind == "present":
            result = await self._finalize_result()
            if result.working_diagnosis is None:
                yield self._record(prompts.present_fallback(self._language))
                return
            async for chunk in self._stream_reply(
                system=prompts.present_sys(self._language),
                user=self._render_result(result),
                fallback=prompts.present_fallback(self._language),
                model=self._model,        # presenting the medical result stays on REASONING
            ):
                yield chunk
            return
        missing = list(turn.missing)
        async for chunk in self._stream_reply(
            system=prompts.question_sys(self._language),
            user=self._question_ctx(turn.suggested, missing),
            fallback=prompts.fallback_question(self._language, missing, turn.suggested),
        ):
            yield chunk

    async def _decide(self, user_text: str) -> _Turn:
        """The shared per-turn flow: identity → extraction → (engine consultation when the
        picture actually changed) → shared sufficiency. Returns WHAT to reply, not the text.
        Logs where the turn's time went, so latency is measured rather than guessed."""
        text = (user_text or "").strip()
        if not text:
            return _Turn("static", reply=self.greeting() if not self._identified else
                         prompts.fallback_question(self._language, self._missing_fields(), ""))
        if not self._identified:
            return _Turn("static", reply=await self._identify(text))

        # Safety is shared infrastructure: it runs before extraction or diagnosis,
        # while the Concierge remains the only persona that speaks to the patient.
        if self._safety_check is not None:
            advice = await self._safety_check(text, self._picture)
            if advice:
                self._history.append(f"patient: {text}")
                return _Turn("static", reply=self._record(advice))

        self._history.append(f"patient: {text}")

        # LANGUAGE: decided once, then LOCKED. First contact — an explicit choice ("عربي",
        # "English") or, failing that, the script of this first message picks the language.
        # After that the language NEVER follows the message's script again (that per-turn
        # following is what made the model flip languages mid-interview); only an explicit
        # request switches it, detected deterministically — the unlock must not depend on the
        # very model whose confusion we are locking out.
        switch = explicit_language_switch(text)
        if not self._language_locked:
            self._language = switch or detect_language(text)
            self._language_locked = True
        elif switch and switch != self._language:
            logger.info("concierge: explicit language switch -> %s", switch)
            self._language = switch

        t0 = perf_counter()
        intake = await self._extract(text)
        t_extract = perf_counter() - t0

        # Meta / non-clinical → answered by the concierge, NEVER sent to the engine.
        if not intake.is_clinical and not (intake.symptoms or intake.chief_complaint):
            logger.info("concierge turn: extract=%.2fs -> smalltalk", t_extract)
            return _Turn("smalltalk")

        # Clinical: grow the picture; consult the engine ONLY when this turn actually changed it.
        # A turn that repeats known facts ("like I said, my head hurts") reuses the last
        # assessment instead of paying a full reasoning call — the other half of the latency fix.
        # `language` is excluded from the comparison: it changes with the patient's script and
        # carries no clinical information.
        before = self._picture.model_dump(exclude={"language"})
        self._picture.merge(intake)
        self._picture.language = self._language
        changed = self._picture.model_dump(exclude={"language"}) != before

        if changed or self._last_assess is None:
            t1 = perf_counter()
            assess = await self._facade.assess(self._picture)
            self._last_assess = assess
            logger.info("concierge turn: extract=%.2fs assess=%.2fs",
                        t_extract, perf_counter() - t1)
        else:
            assess = self._last_assess
            logger.info("concierge turn: extract=%.2fs assess=skipped (no new clinical info)",
                        t_extract)

        cds_enough = assess.sufficient and assess.confidence >= self._facade.confidence_floor
        missing = self._missing_fields()
        if cds_enough and not missing:                     # SHARED sufficiency: both agree
            self._complete = True
            return _Turn("present")
        return _Turn("question", suggested=assess.suggested_question, missing=tuple(missing))

    async def _stream_reply(
        self, *, system: str, user: str, fallback: str, model: ModelTier | None = None,
    ) -> AsyncIterator[str]:
        """Stream one conversational reply, with the same guarantees the non-streaming helpers
        give: a failed or empty stream falls back to the deterministic fallback text, and the
        full reply lands in the history exactly once — including what was already sent if the
        client disconnects mid-stream (that IS what the patient saw)."""
        got = ""
        try:
            try:
                async for chunk in self._gateway.stream_text(
                        system=system, user=user, model=model or self._fast):
                    if chunk:
                        got += chunk
                        yield chunk
            except LLMError:
                pass                        # fall through to the fallback below
            if not got.strip():
                got = fallback
                yield fallback
        finally:
            if got:
                self._record(got)

    # ── identity (LLM-driven: a non-name is never taken as the name) ─────────────
    async def _identify(self, text: str) -> str:
        lang = detect_language(text)
        try:
            parsed = await self._gateway.complete_typed(
                IdentityParse, system=prompts.IDENTITY_SYS, user=text, model=self._fast)
        except LLMError:
            cand = parse_name(text)
            parsed = (IdentityParse(name=cand, has_name=True, language=lang)
                      if _looks_like_name(cand) else IdentityParse(has_name=False, language=lang))
        self._language = lang or parsed.language
        if parsed.has_name and parsed.name.strip():
            name = parsed.name.strip()
            self._name = name
            self._identified = True
            existing = self._identity.lookup(name)
            if existing:
                self._patient_id = existing
                return prompts.greet_known(name, self._language)
            self._patient_id = self._identity.register(name)
            return prompts.greet_unknown(name, self._language)
        return prompts.ask_name(self._language)        # no name yet → ask, stay unidentified

    # ── per-turn extraction ───────────────────────────────────────────────────────
    async def _extract(self, text: str) -> IntakeExtraction:
        try:
            intake = await self._gateway.complete_typed(
                IntakeExtraction, system=prompts.EXTRACT_SYS, user=text, model=self._fast)
        except LLMError:
            intake = IntakeExtraction()
        updates: dict = {}
        if not intake.original_text:
            updates["original_text"] = text
        if not intake.language:
            updates["language"] = detect_language(text)
        return intake.model_copy(update=updates) if updates else intake

    # ── the concierge's own completeness check (the shared half of sufficiency) ───
    def _missing_fields(self) -> list[str]:
        p = self._picture
        if not p.chief_complaint:
            return []                                       # nothing committed yet
        missing: list[str] = []
        if not p.duration:
            missing.append("duration")
        if not p.severity:
            missing.append("severity")
        if "pain" in (p.chief_complaint + " " + " ".join(p.symptoms)).lower() and not p.location:
            missing.append("location")
        return missing

    # ── compose the next question (the concierge owns the conversation) ───────────
    def _question_ctx(self, suggested: str, missing: list[str]) -> str:
        """The user prompt for the next-question call — shared by both reply paths."""
        return (
            self._context_block() +
            f"\n\nMedical engine suggested question: {suggested or '(none)'}"
            f"\nInformation still needed: {', '.join(missing) or '(none)'}"
            "\n\nCompose the next single question."
        )

    async def _compose_question(self, suggested: str, missing: list[str]) -> str:
        try:
            reply = await self._gateway.complete_text(
                system=prompts.question_sys(self._language),
                user=self._question_ctx(suggested, missing), model=self._fast)
            return reply.strip() or prompts.fallback_question(self._language, missing, suggested)
        except LLMError:
            return prompts.fallback_question(self._language, missing, suggested)

    # ── non-clinical reply (reception only, with the concierge's own memory) ──────
    def _smalltalk_ctx(self) -> str:
        """The user prompt for the small-talk call (the patient's line is already in the
        history) — shared by both reply paths."""
        return self._context_block() + "\n\nReply to the patient's latest message."

    async def _smalltalk(self) -> str:
        try:
            reply = await self._gateway.complete_text(
                system=prompts.smalltalk_sys(self._language), user=self._smalltalk_ctx(),
                model=self._fast)
            return reply.strip() or prompts.smalltalk_fallback(self._language)
        except LLMError:
            return prompts.smalltalk_fallback(self._language)

    # ── present the CDSS's finished assessment (Stage B) — relay, never invent ─────
    async def _finalize_result(self) -> FinalizeResult:
        """Run the engine's finalize over the mature picture and remember the result — shared
        by both reply paths (the milestone itself is not streamable; only its presentation is)."""
        diff = self._last_assess.hypotheses if self._last_assess else None
        result = await self._facade.finalize(self._picture, differential=diff)
        self._result = result
        return result

    async def _present_result(self) -> str:
        result = await self._finalize_result()
        if result.working_diagnosis is None:
            return prompts.present_fallback(self._language)
        try:
            reply = await self._gateway.complete_text(
                system=prompts.present_sys(self._language),
                user=self._render_result(result), model=self._model)
            return reply.strip() or prompts.present_fallback(self._language)
        except LLMError:
            return prompts.present_fallback(self._language)

    def _render_result(self, result) -> str:
        wd = result.working_diagnosis
        lines = [f"Leading consideration: {wd.condition} (confidence {wd.confidence:.0%})."]
        if wd.differential:
            others = ", ".join(h.condition for h in wd.differential[:4] if h.condition)
            if others:
                lines.append("Other possibilities considered: " + others + ".")
        if result.recommended_tests:
            names = ", ".join(t.name for t in result.recommended_tests if t.name)
            if names:
                lines.append("Suggested tests: " + names + ".")
        if wd.confirmation_pathway:
            lines.append("Confirmation pathway: " + wd.confirmation_pathway)
        return (self._known_context() + "\n\nMedical system result:\n" + "\n".join(lines) +
                "\n\nPresent this to the patient.")

    def _known_context(self) -> str:
        """What the concierge KNOWS — passed into its conversational calls so it answers
        consistently (e.g. it can tell the patient their own name)."""
        bits: list[str] = []
        if self._name:
            bits.append(f"Patient's name: {self._name}")
        p = self._picture
        if p.chief_complaint:
            bits.append(f"Chief complaint: {p.chief_complaint}")
        if p.symptoms:
            bits.append("Symptoms so far: " + ", ".join(p.symptoms))
        return "Known so far — " + ("; ".join(bits) if bits else "nothing yet") + "."

    def _context_block(self) -> str:
        """Known facts + the running conversation — so every reply REMEMBERS the dialogue
        (no re-greeting, natural flow). The latest patient line is already in the history."""
        return (self._known_context() + "\n\nConversation so far:\n" +
                ("\n".join(self._history[-10:]) or "(none)"))

    def _record(self, reply: str) -> str:
        self._history.append(f"assistant: {reply}")
        return reply


def _looks_like_name(s: str) -> bool:
    """Conservative check for the rare LLM-down fallback: 1-2 alphabetic tokens (so a phrase
    like 'speak in english' is NOT mistaken for a name)."""
    toks = s.split()
    return 1 <= len(toks) <= 2 and all(t.isalpha() for t in toks)

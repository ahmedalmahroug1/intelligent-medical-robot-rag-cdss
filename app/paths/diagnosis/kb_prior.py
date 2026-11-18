"""Optional knowledge-based re-ranking for diagnosis candidates.

The LLM still generates the candidate differential. This layer may reorder those
candidates using a local knowledge file, but it does not add new conditions and
does not produce calibrated probabilities. Its priors are development heuristics,
not empirical estimates or clinically validated likelihoods.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from app.core.state import Hypothesis
from app.llm.base import LLMGateway, ModelTier

DEFAULT_KB_PATH = "data/kb_curated.json"
_BACKGROUND = 0.06                 # p(symptom | disease) for symptoms a disease does not list
_CLAMP = (0.02, 0.98)
_DEFAULT_ALPHA = 0.5               # weight on the LLM confidence; (1-alpha) on the KB posterior

_EXTRACT_SYS = (
    "You are a clinical information extractor. Given a patient's account, decide for "
    "EACH symptom term in the provided controlled vocabulary whether the patient clearly "
    "HAS it (present) or it is not clearly present (omit it). Use ONLY terms from the "
    "vocabulary, verbatim. Demographic/history terms (e.g. child_age, older_age, male, "
    "female, smoking_history, asthma_history, recent_antipsychotic_med, recent_fish_ingestion) "
    "count too — infer them from the account. Return ONLY JSON: {\"present\": [...]}."
)

# Common LLM phrasings -> canonical curated-KB name (normalised keys). Keeps the
# re-ranker self-contained (no dependency on the evaluation synonym table).
_ALIASES: dict[str, str] = {
    "upper respiratory tract infection": "URTI", "upper respiratory infection": "URTI",
    "common cold": "URTI",
    "flu": "Influenza",
    "laryngitis": "Acute laryngitis",
    "pharyngitis": "Viral pharyngitis", "sore throat": "Viral pharyngitis",
    "otitis media": "Acute otitis media", "ear infection": "Acute otitis media",
    "sinusitis": "Acute rhinosinusitis", "rhinosinusitis": "Acute rhinosinusitis",
    "acute sinusitis": "Acute rhinosinusitis",
    "chronic sinusitis": "Chronic rhinosinusitis",
    "allergic rhinitis": "Allergic sinusitis", "hay fever": "Allergic sinusitis",
    "acute bronchitis": "Bronchitis",
    "asthma": "Bronchospasm / acute asthma exacerbation",
    "asthma exacerbation": "Bronchospasm / acute asthma exacerbation",
    "asthma attack": "Bronchospasm / acute asthma exacerbation",
    "acute asthma": "Bronchospasm / acute asthma exacerbation",
    "bronchospasm": "Bronchospasm / acute asthma exacerbation",
    "lung cancer": "Pulmonary neoplasm", "lung carcinoma": "Pulmonary neoplasm",
    "lung tumor": "Pulmonary neoplasm", "lung tumour": "Pulmonary neoplasm",
    "tb": "Tuberculosis",
    "pertussis": "Whooping cough",
    "laryngotracheobronchitis": "Croup",
    "pulmonary embolus": "Pulmonary embolism",
    "afib": "Atrial fibrillation",
    "svt": "PSVT", "supraventricular tachycardia": "PSVT",
    "paroxysmal supraventricular tachycardia": "PSVT",
    "angina": "Stable angina", "angina pectoris": "Stable angina",
    "dystonia": "Acute dystonic reactions", "acute dystonia": "Acute dystonic reactions",
    "dystonic reaction": "Acute dystonic reactions",
    "drug induced dystonia": "Acute dystonic reactions",
    "gastroesophageal reflux disease": "GERD", "gastroesophageal reflux": "GERD",
    "acid reflux": "GERD", "reflux": "GERD",
    "groin hernia": "Inguinal hernia",
    "pancreatic cancer": "Pancreatic neoplasm", "pancreatic carcinoma": "Pancreatic neoplasm",
    "pancreatic tumor": "Pancreatic neoplasm",
    "hiv": "HIV (initial infection)", "acute hiv": "HIV (initial infection)",
    "acute retroviral syndrome": "HIV (initial infection)",
    "chagas disease": "Chagas",
    "scombroid": "Scombroid food poisoning", "scombroid poisoning": "Scombroid food poisoning",
    "lupus": "SLE", "systemic lupus erythematosus": "SLE",
    "anaemia": "Anemia",
    "rib fracture": "Spontaneous rib fracture",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def _contig(a: list[str], b: list[str]) -> bool:
    """True if token list `a` appears as a contiguous run inside token list `b`."""
    if not a or len(a) > len(b):
        return False
    for i in range(len(b) - len(a) + 1):
        if b[i:i + len(a)] == a:
            return True
    return False


def _canonical_key(condition: str) -> str:
    """Fold common disease synonyms onto ONE key so ensemble votes for the same
    condition under different names ('lung cancer' / 'pulmonary neoplasm' / 'lung
    carcinoma') are counted together instead of splitting the reciprocal-rank vote.
    Uses the same alias table + contiguous-phrase matching as `KBReranker.match`, so
    fusion and re-ranking agree on what counts as the same condition. Open-world
    conditions the table does not know keep their normalised name as the key."""
    n = _norm(condition)
    if not n:
        return ""
    if n in _ALIASES:
        return _norm(_ALIASES[n])
    ntoks = n.split()
    for phrase, name in _ALIASES.items():
        if _contig(phrase.split(), ntoks):
            return _norm(name)
    return n


def fuse_llm_rankings(
    rankings: list[list[Hypothesis]], *, k0: int = 60
) -> list[Hypothesis]:
    """Reciprocal-rank-fuse K independent LLM differentials into one ranking.

    Self-consistency: sampling the differential K times (temperature > 0) and fusing
    the rankings reduces the variance of a weak model far more reliably than any
    single greedy call. Each candidate scores ``sum_s 1/(k0 + rank_in_sample_s)``, so
    a condition ranked highly by MANY samples rises and a one-off guess sinks.
    Candidates are keyed by normalised condition name; the representative kept is the
    occurrence with the highest LLM confidence, and its confidence is replaced by the
    normalised fused score so ordering reflects agreement, not one sample's number.

    Pure (no I/O) so it is unit-tested without a gateway. It never invents a
    condition: the output is exactly the union of the conditions the samples proposed
    — so fusing the LLM with itself can only reorder, never add outside knowledge.
    """
    agg: dict[str, dict] = {}
    for ranking in rankings:
        ordered = sorted(ranking, key=lambda h: float(h.confidence), reverse=True)
        for rank, h in enumerate(ordered, start=1):
            key = _canonical_key(h.condition)
            if not key:
                continue
            contrib = 1.0 / (k0 + rank)
            e = agg.get(key)
            if e is None:
                agg[key] = {"score": contrib, "rep": h, "rep_conf": float(h.confidence)}
            else:
                e["score"] += contrib
                if float(h.confidence) > e["rep_conf"]:
                    e["rep"], e["rep_conf"] = h, float(h.confidence)
    if not agg:
        return []
    ranked = sorted(agg.values(), key=lambda e: e["score"], reverse=True)
    hi = ranked[0]["score"] or 1.0
    return [
        e["rep"].model_copy(update={"confidence": round(min(1.0, e["score"] / hi), 4)})
        for e in ranked
    ]


class KBReranker:
    def __init__(
        self,
        kb_path: str = DEFAULT_KB_PATH,
        *,
        alpha: float = _DEFAULT_ALPHA,
        background: float = _BACKGROUND,
        model: ModelTier = ModelTier.FAST,
        mode: str = "rerank",
        fusion_kb_topk: int = 8,
        fusion_k0: int = 60,
        fusion_w_llm: float = 1.0,
        fusion_w_kb: float = 0.6,
    ) -> None:
        raw = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        self._kb: dict[str, dict[str, float]] = {}
        for name, d in raw.items():
            if name.startswith("_"):
                continue
            self._kb[name] = dict(d.get("symptoms", {}))     # positives only
        self._vocab = sorted({s for sym in self._kb.values() for s in sym})
        self._index = {_norm(n): n for n in self._kb}        # normalised name -> canonical
        self.alpha = alpha
        self.background = background
        self.model = model
        # "rerank": light blend that only reorders the LLM's own hypotheses (legacy).
        # "fuse":   reciprocal-rank fusion of the LLM ranking with the KB's Bayesian
        #           ranking — this CAN surface a KB condition the LLM missed (coverage),
        #           while LLM-favoured weights keep the LLM's top-1 protected.
        self.mode = mode
        self.fusion_kb_topk = fusion_kb_topk        # how many KB-only conditions may enter the tail
        self.fusion_k0 = fusion_k0                  # RRF damping constant (standard ~60)
        self.fusion_w_llm = fusion_w_llm            # weight on the LLM rank term
        self.fusion_w_kb = fusion_w_kb              # weight on the KB rank term (< w_llm ⇒ top-1 safe)

    # ── disease-name matching ────────────────────────────────────────────────
    def match(self, condition: str) -> str | None:
        """The curated-KB disease an LLM hypothesis refers to, or None (open-world)."""
        c = _norm(condition)
        if not c:
            return None
        if c in self._index:
            return self._index[c]
        if c in _ALIASES:
            return _ALIASES[c]
        ctoks = c.split()
        for phrase, name in _ALIASES.items():
            if _contig(phrase.split(), ctoks):
                return name
        for nk, name in self._index.items():
            ktoks = nk.split()
            if _contig(ktoks, ctoks) or _contig(ctoks, ktoks):
                return name
        return None

    # ── scoring ──────────────────────────────────────────────────────────────
    async def present_symptoms(self, gateway: LLMGateway, text: str) -> set[str]:
        user = (f"Controlled vocabulary:\n{', '.join(self._vocab)}\n\n"
                f"Patient account:\n{text[:3500]}\n\n"
                "Return the JSON with the present symptom terms.")
        try:
            data = await gateway.complete_structured(
                system=_EXTRACT_SYS, user=user, model=self.model
            )
        except Exception:  # noqa: BLE001 - extraction failure -> no re-rank, not a crash
            return set()
        vocab = set(self._vocab)
        return {s for s in (data.get("present") or []) if s in vocab}

    def _bayes_logscore(self, present: set[str], disease: str) -> float:
        sym = self._kb[disease]
        total = 0.0
        for s in present:
            p = sym.get(s, self.background)
            total += math.log(max(_CLAMP[0], min(_CLAMP[1], p)))
        return total

    async def rank_all(self, gateway: LLMGateway, text: str) -> tuple[list[str], int]:
        """STRONG-TRUST mode: rank ALL 36 curated-KB conditions by present-evidence
        naive Bayes from `text`, ignoring any LLM differential. Unlike `rerank`, this
        can surface a KB condition the LLM never proposed — the KB does the diagnosing,
        the caller only supplies the patient account (e.g. an interview transcript).
        Returns (ranked condition names, n_present); empty list if extraction found
        nothing to score on."""
        present = await self.present_symptoms(gateway, text)
        if not present:
            return [], 0
        ranked = sorted(self._kb, key=lambda d: self._bayes_logscore(present, d), reverse=True)
        return ranked, len(present)

    # ── reciprocal-rank fusion (LLM ranking ⊕ KB Bayesian ranking) ────────────
    def _fuse_ranks(
        self, hypotheses: list[Hypothesis], ranked_kb: list[str]
    ) -> list[Hypothesis]:
        """Pure fusion given the LLM hypotheses and the KB's ranked condition list.

        Split out (no I/O) so the fusion math is unit-tested without a gateway.

        Reciprocal-rank fusion: score(c) = w_llm/(k0+rank_llm) + w_kb/(k0+rank_kb).
        A condition ranked highly by BOTH sources is boosted; a KB-only condition
        (one the LLM never proposed) can enter the TAIL for coverage but, because
        w_kb < w_llm, can never outrank the LLM's #1 — so top-1 is protected by
        construction while top-k coverage improves.
        """
        k0, w_llm, w_kb = self.fusion_k0, self.fusion_w_llm, self.fusion_w_kb
        kb_rank = {name: i + 1 for i, name in enumerate(ranked_kb)}

        # LLM side: rank by confidence (desc), preserving original order on ties.
        order = sorted(
            range(len(hypotheses)),
            key=lambda i: hypotheses[i].confidence,
            reverse=True,
        )
        # A canonical key per candidate: the KB name when the condition maps into the
        # KB (so both sources score the SAME candidate), else the normalised LLM name.
        entries: dict[str, dict] = {}
        for rank_llm, i in enumerate(order, start=1):
            h = hypotheses[i]
            kb_name = self.match(h.condition)
            key = kb_name or _norm(h.condition)
            e = entries.get(key)
            if e is None:
                entries[key] = {"hyp": h, "kb_name": kb_name, "rank_llm": rank_llm}
            elif rank_llm < e["rank_llm"]:
                e.update({"hyp": h, "rank_llm": rank_llm})

        # KB-only candidates: the top KB conditions the LLM never proposed, added to
        # the tail so a correct condition the LLM missed can still be recovered.
        for name in ranked_kb[: self.fusion_kb_topk]:
            if name not in entries:
                entries[name] = {"hyp": None, "kb_name": name, "rank_llm": None}

        scored: list[tuple[float, Hypothesis]] = []
        for key, e in entries.items():
            score = 0.0
            if e["rank_llm"] is not None:
                score += w_llm / (k0 + e["rank_llm"])
            if e["kb_name"] in kb_rank:
                score += w_kb / (k0 + kb_rank[e["kb_name"]])
            hyp = e["hyp"] or Hypothesis(condition=e["kb_name"], confidence=0.0)
            scored.append((score, hyp))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        # Store the fused score as a normalised pseudo-confidence for ordering; the
        # working-diagnosis node sets the real confidence on the chosen lead.
        hi = scored[0][0] if scored else 1.0
        out: list[Hypothesis] = []
        for score, hyp in scored:
            conf = round(min(1.0, score / hi), 4) if hi > 0 else hyp.confidence
            out.append(hyp.model_copy(update={"confidence": conf}))
        return out

    async def fuse(
        self, gateway: LLMGateway, text: str, hypotheses: list[Hypothesis]
    ) -> list[Hypothesis]:
        """Fuse the LLM differential with the KB Bayesian ranking (see `_fuse_ranks`).
        Retrieves the KB ranking from the patient's present symptoms; on any failure
        (extraction empty, etc.) returns the LLM hypotheses unchanged — fusion can
        only help, never lose the LLM's ranking."""
        if not hypotheses:
            return hypotheses
        ranked_kb, n_present = await self.rank_all(gateway, text)
        if not ranked_kb:
            return hypotheses
        return self._fuse_ranks(hypotheses, ranked_kb)

    # ── the re-ranking layer ─────────────────────────────────────────────────
    async def rerank(
        self, gateway: LLMGateway, text: str, hypotheses: list[Hypothesis]
    ) -> list[Hypothesis]:
        """Reorder `hypotheses`: in-KB conditions blend LLM confidence with their
        naive-Bayes posterior; out-of-KB conditions keep their LLM confidence."""
        if len(hypotheses) < 2:
            return hypotheses
        present = await self.present_symptoms(gateway, text)
        if not present:
            return hypotheses                                # nothing to score on

        kb_log: dict[int, float] = {}
        for i, h in enumerate(hypotheses):
            name = self.match(h.condition)
            if name is not None:
                kb_log[i] = self._bayes_logscore(present, name)
        if not kb_log:
            return hypotheses                                # no hypothesis is in the KB

        # Normalise the in-KB log-scores to [0,1] so they are comparable to the
        # LLM confidence (min-max over the in-KB hypotheses; best -> 1, worst -> 0).
        lo, hi = min(kb_log.values()), max(kb_log.values())
        span = hi - lo
        kb_rel = {i: ((ls - lo) / span if span > 1e-9 else 1.0) for i, ls in kb_log.items()}

        rescored: list[tuple[float, Hypothesis]] = []
        for i, h in enumerate(hypotheses):
            if i in kb_rel:
                final = self.alpha * float(h.confidence) + (1.0 - self.alpha) * kb_rel[i]
            else:
                final = float(h.confidence)
            final = max(0.0, min(1.0, final))
            rescored.append((final, h.model_copy(update={"confidence": round(final, 4)})))
        rescored.sort(key=lambda pair: pair[0], reverse=True)
        return [h for _, h in rescored]


"""Staged, tool-using background curator for clinician-governed KB growth.

The patient-facing model never authors knowledge.  This independent agent plans
research, queries multiple evidence tools, writes diagnosis and treatment as
separate constrained drafts, verifies every claim against the retrieved text,
and attempts one evidence-bounded repair. A valid output becomes retrievable
immediately with an explicit unapproved label, while the proposal and audit
remain under ``pending/`` for later clinician approval, rejection, or deletion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Iterable

from app.config import settings
from app.core.state import ConsultationState, EvidenceItem
from app.knowledge.drug_labels import OpenFDALabelRetriever
from app.knowledge.guidelines import OfficialGuidelineCatalog
from app.knowledge.models import (
    DiagnosticDraft,
    DraftVerification,
    DynamicDiseaseRecord,
    EvidenceToolRun,
    GapAssessment,
    GapTarget,
    KnowledgeProposal,
    KnowledgeSource,
    ResearchAudit,
    ResearchPlan,
    TargetedRepair,
    TreatmentDraft,
    proposal_approval_errors,
    record_approval_errors,
)
from app.knowledge.repository import KnowledgeProposalRepository, normalise_condition
from app.knowledge.retrieval import ClinicalKnowledgeBase
from app.llm.base import LLMGateway, ModelTier
from app.paths.diagnosis.retrieval import EvidenceQuery, Retriever
from app.persistence.store import ConsultationStore

logger = logging.getLogger(__name__)


_GAP_SYSTEM = """You are the coverage-audit stage of a clinical knowledge curator.
You receive suspected conditions from one completed consultation and the complete inventory of
currently active records. Return only JSON matching {"gaps":[{"condition":"...",
"gap_kind":"missing|weak","reason":"...","matched_record_name":null}]}.
Rules:
- Evaluate only the supplied suspected conditions; never invent another condition.
- A synonym, abbreviation or subtype already represented by a sufficiently complete record is
  not missing. Prefer no gap over a duplicate record.
- Mark weak only when the inventory explicitly lists material missing operational fields.
- A hypothesis being present does not prove it is true; this is a coverage audit, not diagnosis.
- Return an empty gaps list when coverage is adequate."""


_PLAN_SYSTEM = """You plan evidence searches for a clinical knowledge curator.
Return only the requested ResearchPlan JSON. Do not write the disease record and do not assert
that the suspected diagnosis is correct. Produce focused PubMed queries for diagnostic criteria,
differentials/red flags and management guidelines. List official-guideline topics. List up to eight
generic medicine names worth checking in a regulatory label database; these are search leads, not
recommendations. Never include a dose, brand-only name, URL or invented citation."""


_DIAGNOSTIC_SYSTEM = """You are the diagnostic author inside a clinical knowledge curator.
Use ONLY the supplied evidence summaries and their exact source_ids. Return only DiagnosticDraft
JSON. Every clinical item must cite the source IDs that explicitly support it. Do not turn absence
of evidence into a negative finding. Include discriminating interview questions, explicit
diagnostic value for each symptom, important alternatives with distinguishing features, red flags,
and when testing or imaging is not routinely indicated. If evidence is insufficient, omit a claim
and describe the gap in diagnostic_limitations. Never invent a URL, prevalence, threshold or rule."""


_TREATMENT_SYSTEM = """You are the treatment and medication-safety author inside a clinical
knowledge curator. Use ONLY the supplied guideline, review and regulatory-label summaries. Return
only TreatmentDraft JSON. Treatment entries are decision-support options, never prescriptions:
do_not_autoprescribe must be true. For every medicine include explicit when-to-consider,
avoidance, caution, required-history and monitoring fields supported by cited sources. A guideline
supports therapeutic role; a regulatory label supports drug-specific contraindications/warnings.
Never invent a dose or transfer a warning from one medicine to another. State non-recommended
interventions clearly. Account for pregnancy, age, comorbidities and current medicines only when
the evidence supports those checks. Omit unsupported detail and record the gap."""


_VERIFY_SYSTEM = """You are an independent evidence auditor, not the record author.
Compare every operational claim in the supplied disease record with the supplied source summaries.
Return only DraftVerification JSON. A source ID existing is not enough: the cited summary must
actually entail the claim. Flag unsupported specificity, citation mismatch, internal contradiction,
unsafe medicine omissions, missing distinguishing features, and guideline-vs-label conflicts.
Use critical for potentially dangerous contradictions or fabricated claims, major for a material
clinical/traceability gap, and minor for wording that does not alter operation. This is a strict
reference-publication check: passed must be false whenever ANY issue remains, including minor
citation over-specificity. Do not repair the draft and do not use outside memory."""


_REPAIR_DIAGNOSTIC_SYSTEM = _DIAGNOSTIC_SYSTEM + """
This is a repair pass. Rewrite the complete DiagnosticDraft, addressing the supplied audit issues.
Remove an unsupported claim when the allowed evidence cannot repair it; never fill the gap from
memory. Preserve valid, well-cited content."""


_REPAIR_TREATMENT_SYSTEM = _TREATMENT_SYSTEM + """
This is a repair pass. Rewrite the complete TreatmentDraft, addressing the supplied audit issues.
Remove an unsupported option when the allowed evidence cannot repair it; never fill the gap from
memory. Preserve valid, well-cited content."""


_TARGETED_REPAIR_SYSTEM = """You are a precise clinical-record patcher.
Return only TargetedRepair JSON using RFC-6902-style JSON-pointer paths and the operations
"replace" or "remove". Resolve every verifier issue with the smallest possible change. Prefer
removing unsupported specificity; add no fact that is not explicitly present in the allowed
evidence summaries. To change an item inside a string list, replace that whole list field. To
remove an unsupported record item, remove its list index. You may replace an item's source_ids
only with IDs from allowed_evidence. Never alter the sources collection, source summaries,
schema_version, generated metadata, or the disease identity. Do not return a full record. The
patched full record will be schema-validated after your operations are applied. Resolve supplied
deterministic_gate_errors too, and never empty a required history, monitoring, caution or citation
list merely to silence a verifier issue."""


class KnowledgeCuratorAgent:
    """Background research, authoring, verification and proposal generation."""

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        repository: KnowledgeProposalRepository,
        knowledge: ClinicalKnowledgeBase,
        retriever: Retriever,
        provider_name: str,
        model_name: str,
        guideline_catalog: OfficialGuidelineCatalog | None = None,
        drug_labels: OpenFDALabelRetriever | None = None,
        min_confidence: float | None = None,
        max_sources: int | None = None,
        repair_attempts: int | None = None,
    ) -> None:
        self.gateway = gateway
        self.repository = repository
        self.knowledge = knowledge
        self.retriever = retriever
        self.guideline_catalog = guideline_catalog or OfficialGuidelineCatalog()
        self.drug_labels = drug_labels or OpenFDALabelRetriever()
        self.provider_name = provider_name
        self.model_name = model_name
        self.min_confidence = (
            settings.KNOWLEDGE_AGENT_MIN_CONFIDENCE
            if min_confidence is None else min_confidence
        )
        self.max_sources = max_sources or settings.KNOWLEDGE_AGENT_MAX_SOURCES
        self.repair_attempts = (
            settings.KNOWLEDGE_AGENT_REPAIR_ATTEMPTS
            if repair_attempts is None else repair_attempts
        )
        self._tasks: set[asyncio.Task] = set()
        self._submitted_visits: set[str] = set()

    def submit_state(self, state: ConsultationState) -> None:
        """Schedule work and return immediately; patient latency is unaffected."""
        visit_id = state.visit.visit_id
        if visit_id in self._submitted_visits:
            return
        self._submitted_visits.add(visit_id)
        snapshot = state.model_copy(deep=True)
        self._spawn(self.analyse(snapshot), label=f"visit {visit_id}")

    def submit_visit(self, store: ConsultationStore, visit_id: str) -> None:
        """Schedule a persisted Concierge visit without blocking its final reply."""
        if visit_id in self._submitted_visits:
            return
        self._submitted_visits.add(visit_id)
        self._spawn(self._load_and_analyse(store, visit_id), label=f"visit {visit_id}")

    async def _load_and_analyse(self, store: ConsultationStore, visit_id: str) -> None:
        visit = await store.visits.get(visit_id)
        if visit is None:
            return
        patient = await store.patients.get(visit.patient_id)
        sessions = await store.sessions.list_for_visit(visit_id)
        if patient is None or not sessions:
            return
        await self.analyse(ConsultationState(patient=patient, visit=visit, session=sessions[0]))

    async def analyse(self, state: ConsultationState) -> list[KnowledgeProposal]:
        candidates = self._candidates(state)
        if not candidates:
            return []
        inventory = self.knowledge.inventory()
        assessment = await self.gateway.complete_typed(
            GapAssessment,
            system=_GAP_SYSTEM,
            user=json.dumps(
                {"suspected_conditions": candidates, "active_inventory": inventory},
                ensure_ascii=False,
            ),
            model=ModelTier.FAST,
        )
        allowed = {normalise_condition(item["condition"]) for item in candidates}
        proposals: list[KnowledgeProposal] = []
        for gap in assessment.gaps:
            if normalise_condition(gap.condition) not in allowed:
                logger.warning("Knowledge curator ignored invented gap target %s", gap.condition)
                continue
            if self.repository.has_pending_for(gap.condition):
                continue
            proposal = await self._propose(gap, state, candidates)
            self.repository.save(proposal)
            proposals.append(proposal)
        return proposals

    async def repair_saved_proposal(self, proposal_id: str) -> KnowledgeProposal:
        """Re-author a pending draft from its existing evidence and verifier report.

        This is deliberately cheaper than repeating gap detection and retrieval.
        It never approves the record; it replaces the pending draft in place and
        preserves the accumulated research/repair audit.
        """
        proposal = self.repository.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown knowledge proposal '{proposal_id}'.")
        if proposal.status.value != "pending":
            raise ValueError(f"Proposal '{proposal_id}' is already {proposal.status.value}.")
        if proposal.record is None:
            raise ValueError(f"Proposal '{proposal_id}' has no record to repair.")

        record = proposal.record
        sources = record.sources
        diagnostic, treatment = self._split_record(record)
        verification = proposal.verification or DraftVerification(
            passed=False,
            summary="Independent verification was missing and must be rerun.",
        )
        audit = proposal.research_audit or ResearchAudit()
        gap = GapTarget(
            condition=proposal.target_condition,
            gap_kind=proposal.gap_kind,
            reason=proposal.gap_reason,
        )
        diagnostic_sources = self._for_scope(sources, "diagnosis")
        treatment_sources = self._for_scope(sources, "treatment")

        for _ in range(self.repair_attempts):
            issues = proposal_approval_errors(proposal)
            if proposal.verification is None:
                issues.append("independent verification is missing")
            if not issues:
                break
            if proposal.research_audit and not proposal.research_audit.official_guideline_coverage:
                break
            record, diagnostic, treatment, changed = await self._repair_record(
                gap, record, diagnostic, treatment, sources,
                diagnostic_sources, treatment_sources, issues, verification,
            )
            if not changed:
                break
            verification = await self._verify(record, sources)
            audit = audit.model_copy(update={"repair_attempts": audit.repair_attempts + 1})
            proposal = proposal.model_copy(update={
                "record": record,
                "research_audit": audit,
                "verification": verification,
            })

        errors = proposal_approval_errors(proposal)
        proposal = proposal.model_copy(update={
            "approvable": not errors,
            "generation_error": "; ".join(errors) if errors else None,
        })
        return self.repository.save(proposal)

    async def _propose(
        self,
        gap: GapTarget,
        state: ConsultationState,
        candidates: list[dict],
    ) -> KnowledgeProposal:
        base = dict(
            proposal_id=uuid.uuid4().hex,
            target_condition=gap.condition,
            gap_kind=gap.gap_kind,
            gap_reason=gap.reason,
            trigger_visit_id=state.visit.visit_id,
            trigger_hypotheses=candidates,
            model_provider=self.provider_name,
            model_name=self.model_name,
        )
        audit = ResearchAudit()
        try:
            plan = await self.gateway.complete_typed(
                ResearchPlan,
                system=_PLAN_SYSTEM,
                user=json.dumps({
                    "target_condition": gap.condition,
                    "gap_kind": gap.gap_kind,
                    "gap_reason": gap.reason,
                    "output_schema": ResearchPlan.model_json_schema(),
                }, ensure_ascii=False),
                model=ModelTier.FAST,
            )
            sources, audit = await self._research(gap.condition, plan)
            if not sources:
                return KnowledgeProposal(
                    **base,
                    research_audit=audit,
                    generation_error="No usable external evidence was retrieved.",
                )

            diagnostic_sources = self._for_scope(sources, "diagnosis")
            treatment_sources = self._for_scope(sources, "treatment")
            diagnostic_task = self._write_diagnostic(gap, diagnostic_sources)
            treatment_task = self._write_treatment(gap, treatment_sources)
            diagnostic, treatment = await asyncio.gather(diagnostic_task, treatment_task)
            record = self._assemble(diagnostic, treatment, sources)
            verification = await self._verify(record, sources)

            for _ in range(self.repair_attempts):
                provisional = KnowledgeProposal(
                    **base,
                    record=record,
                    research_audit=audit,
                    verification=verification,
                )
                issues = proposal_approval_errors(provisional)
                if not issues:
                    break
                if not audit.official_guideline_coverage:
                    break
                record, diagnostic, treatment, changed = await self._repair_record(
                    gap, record, diagnostic, treatment, sources,
                    diagnostic_sources, treatment_sources, issues, verification,
                )
                if not changed:
                    break
                audit = audit.model_copy(update={"repair_attempts": audit.repair_attempts + 1})
                verification = await self._verify(record, sources)

            proposal = KnowledgeProposal(
                **base,
                record=record,
                research_audit=audit,
                verification=verification,
            )
            errors = proposal_approval_errors(proposal)
            return proposal.model_copy(update={
                "approvable": not errors,
                "generation_error": "; ".join(errors) if errors else None,
            })
        except Exception as exc:  # noqa: BLE001 - persist the failed research attempt
            return KnowledgeProposal(
                **base,
                research_audit=audit,
                generation_error=(
                    f"Staged generation failed: {type(exc).__name__}: {str(exc)[:500]}"
                ),
            )

    async def _research(
        self, condition: str, plan: ResearchPlan,
    ) -> tuple[list[KnowledgeSource], ResearchAudit]:
        runs: list[EvidenceToolRun] = []
        sources: list[KnowledgeSource] = []

        try:
            official = await self.guideline_catalog.search(
                condition, plan.official_guideline_topics,
            )
            sources.extend(official)
            runs.append(EvidenceToolRun(
                tool=self.guideline_catalog.source_name,
                query=condition,
                source_ids=[source.source_id for source in official],
            ))
        except Exception as exc:  # noqa: BLE001 - audit one unavailable evidence tool
            runs.append(EvidenceToolRun(
                tool=self.guideline_catalog.source_name,
                query=condition,
                error=f"{type(exc).__name__}: {str(exc)[:240]}",
            ))

        searches: list[tuple[str, str, str]] = []
        # Two focused searches per clinical half are enough to diversify the
        # evidence without turning one proposal into a burst of NCBI requests.
        searches.extend(("PubMed diagnosis", query, "diagnosis")
                        for query in plan.diagnostic_queries[:2])
        searches.extend(("PubMed treatment", query, "treatment")
                        for query in plan.treatment_queries[:2])
        results = await asyncio.gather(*(
            self.retriever.search(EvidenceQuery(
                query_string=query,
                max_results=self.max_sources,
            ))
            for _, query, _ in searches
        ), return_exceptions=True)
        per_scope: dict[str, int] = {"diagnosis": 0, "treatment": 0}
        for (tool, query, scope), result in zip(searches, results, strict=True):
            if isinstance(result, BaseException):
                runs.append(EvidenceToolRun(
                    tool=tool,
                    query=query,
                    error=f"{type(result).__name__}: {str(result)[:240]}",
                ))
                continue
            converted = self._sources(result, scope=scope)
            remaining = max(0, self.max_sources - per_scope[scope])
            converted = converted[:remaining]
            per_scope[scope] += len(converted)
            sources.extend(converted)
            runs.append(EvidenceToolRun(
                tool=tool,
                query=query,
                source_ids=[source.source_id for source in converted],
            ))

        labels_found: list[str] = []
        catalog_medications: list[str] = []
        try:
            catalog_medications = await self.guideline_catalog.medication_terms(condition)
        except Exception:  # noqa: BLE001 - older/injected catalog implementations may omit it
            catalog_medications = []
        medication_terms = list(dict.fromkeys([
            *catalog_medications,
            *plan.medication_search_terms,
        ]))[:5]
        try:
            labels = await self.drug_labels.search_many(medication_terms)
            for term, source in labels.items():
                if source is not None:
                    labels_found.append(term)
                    sources.append(source)
                    source_ids = [source.source_id]
                else:
                    source_ids = []
                runs.append(EvidenceToolRun(
                    tool=self.drug_labels.source_name,
                    query=term,
                    source_ids=source_ids,
                ))
        except Exception as exc:  # noqa: BLE001 - labels are audited separately
            runs.append(EvidenceToolRun(
                tool=self.drug_labels.source_name,
                query=", ".join(medication_terms),
                error=f"{type(exc).__name__}: {str(exc)[:240]}",
            ))

        sources = self._dedupe_sources(sources)
        official_coverage = any(
            source.source_type.casefold().startswith("official") for source in sources
        )
        audit = ResearchAudit(
            plan=plan,
            tool_runs=runs,
            source_ids=[source.source_id for source in sources],
            official_guideline_coverage=official_coverage,
            medication_labels_requested=medication_terms,
            medication_labels_found=labels_found,
        )
        return sources, audit

    async def _write_diagnostic(
        self, gap: GapTarget, sources: list[KnowledgeSource],
    ) -> DiagnosticDraft:
        return await self.gateway.complete_typed(
            DiagnosticDraft,
            system=_DIAGNOSTIC_SYSTEM,
            user=self._author_payload(gap, sources, DiagnosticDraft),
            model=ModelTier.REASONING,
        )

    async def _write_treatment(
        self, gap: GapTarget, sources: list[KnowledgeSource],
    ) -> TreatmentDraft:
        return await self.gateway.complete_typed(
            TreatmentDraft,
            system=_TREATMENT_SYSTEM,
            user=self._author_payload(gap, sources, TreatmentDraft),
            model=ModelTier.REASONING,
        )

    async def _verify(
        self, record: DynamicDiseaseRecord, sources: list[KnowledgeSource],
    ) -> DraftVerification:
        verification = await self.gateway.complete_typed(
            DraftVerification,
            system=_VERIFY_SYSTEM,
            user=json.dumps({
                "record": record.model_dump(mode="json"),
                "allowed_evidence": [source.model_dump(mode="json") for source in sources],
                "output_schema": DraftVerification.model_json_schema(),
            }, ensure_ascii=False),
            model=ModelTier.REASONING,
        )
        if verification.blocking_issues() and verification.passed:
            verification = verification.model_copy(update={"passed": False})
        return verification

    async def _repair_record(
        self,
        gap: GapTarget,
        record: DynamicDiseaseRecord,
        diagnostic: DiagnosticDraft,
        treatment: TreatmentDraft,
        sources: list[KnowledgeSource],
        diagnostic_sources: list[KnowledgeSource],
        treatment_sources: list[KnowledgeSource],
        issues: list[str],
        verification: DraftVerification,
    ) -> tuple[DynamicDiseaseRecord, DiagnosticDraft, TreatmentDraft, bool]:
        # Once the record is structurally complete, claim-level verifier issues
        # should be fixed with small patches. Rewriting both sections is slower
        # and tends to introduce fresh unsupported details.
        deterministic_errors = record_approval_errors(record)
        if verification.issues:
            try:
                record = await self._targeted_repair(
                    record, verification, sources, deterministic_errors,
                )
                diagnostic, treatment = self._split_record(record)
                return record, diagnostic, treatment, True
            except Exception as exc:  # noqa: BLE001 - preserve the last verified draft
                logger.warning(
                    "Targeted knowledge repair failed; preserving the current draft: %s",
                    str(exc),
                )
                return record, diagnostic, treatment, False

        diagnostic, treatment = await self._repair(
            gap, diagnostic, treatment, diagnostic_sources, treatment_sources,
            issues, verification,
        )
        repaired_record = self._assemble(diagnostic, treatment, sources)
        changed = repaired_record.model_dump(mode="json") != record.model_dump(mode="json")
        return repaired_record, diagnostic, treatment, changed

    async def _targeted_repair(
        self,
        record: DynamicDiseaseRecord,
        verification: DraftVerification,
        sources: list[KnowledgeSource],
        deterministic_errors: list[str],
    ) -> DynamicDiseaseRecord:
        cited = {
            source_id
            for issue in verification.issues
            for source_id in issue.source_ids
        }
        issue_text = " ".join(
            f"{issue.message} {issue.suggested_fix}" for issue in verification.issues
        )
        cited.update(
            source.source_id for source in sources if source.source_id in issue_text
        )
        allowed = [source for source in sources if not cited or source.source_id in cited]
        patch = await self.gateway.complete_typed(
            TargetedRepair,
            system=_TARGETED_REPAIR_SYSTEM,
            user=json.dumps({
                "current_record": record.model_dump(mode="json"),
                "deterministic_gate_errors": deterministic_errors,
                "verifier_issues": [issue.model_dump(mode="json")
                                    for issue in verification.issues],
                "allowed_evidence": [source.model_dump(mode="json") for source in allowed],
                "output_schema": TargetedRepair.model_json_schema(),
            }, ensure_ascii=False),
            model=ModelTier.REASONING,
        )
        payload = record.model_dump(mode="json")
        for operation in patch.operations:
            self._apply_patch_operation(payload, operation.op, operation.path, operation.value)
        candidate = DynamicDiseaseRecord.model_validate(payload)
        before = set(deterministic_errors)
        after = set(record_approval_errors(candidate))
        introduced = after - before
        if introduced:
            raise ValueError(
                "Targeted repair introduced approval-gate errors: " + "; ".join(introduced)
            )
        if after:
            raise ValueError(
                "Targeted repair did not resolve approval-gate errors: " + "; ".join(after)
            )
        return candidate

    @staticmethod
    def _apply_patch_operation(payload: dict, op: str, path: str, value) -> None:
        # Models sometimes treat ``limitations`` like the sourced-text groups
        # and emit /limitations/N/text, although its items are plain strings.
        # Normalize that one unambiguous representation before pointer parsing.
        path = re.sub(r"^/limitations/(\d+)/text$", r"/limitations/\1", path)
        if not path.startswith("/"):
            raise ValueError(f"Invalid JSON pointer '{path}'.")
        tokens = [token.replace("~1", "/").replace("~0", "~")
                  for token in path[1:].split("/")]
        protected = {
            "schema_version", "name", "sources", "generated_at", "generated_by_model",
        }
        if not tokens or tokens[0] in protected:
            raise ValueError(f"Patch path '{path}' targets protected record data.")
        node = payload
        for token in tokens[:-1]:
            if isinstance(node, list):
                index = int(token)
                if index < 0 or index >= len(node):
                    raise IndexError(path)
                node = node[index]
            elif isinstance(node, dict) and token in node:
                node = node[token]
            else:
                raise KeyError(path)
        final = tokens[-1]
        if isinstance(node, list):
            index = int(final)
            if index < 0 or index >= len(node):
                raise IndexError(path)
            if op == "remove":
                node.pop(index)
            else:
                node[index] = value
        elif isinstance(node, dict) and final in node:
            if op == "remove":
                del node[final]
            else:
                node[final] = value
        else:
            raise KeyError(path)

    async def _repair(
        self,
        gap: GapTarget,
        diagnostic: DiagnosticDraft,
        treatment: TreatmentDraft,
        diagnostic_sources: list[KnowledgeSource],
        treatment_sources: list[KnowledgeSource],
        deterministic_issues: list[str],
        verification: DraftVerification,
    ) -> tuple[DiagnosticDraft, TreatmentDraft]:
        issues = {
            "deterministic_gate": deterministic_issues,
            "independent_verifier": [issue.model_dump(mode="json")
                                      for issue in verification.issues],
        }
        diagnostic_user = json.dumps({
            "target_condition": gap.condition,
            "current_draft": diagnostic.model_dump(mode="json"),
            "issues": issues,
            "allowed_evidence": [s.model_dump(mode="json") for s in diagnostic_sources],
            "output_schema": DiagnosticDraft.model_json_schema(),
        }, ensure_ascii=False)
        treatment_user = json.dumps({
            "target_condition": gap.condition,
            "current_draft": treatment.model_dump(mode="json"),
            "issues": issues,
            "allowed_evidence": [s.model_dump(mode="json") for s in treatment_sources],
            "output_schema": TreatmentDraft.model_json_schema(),
        }, ensure_ascii=False)
        repaired = await asyncio.gather(
            self.gateway.complete_typed(
                DiagnosticDraft,
                system=_REPAIR_DIAGNOSTIC_SYSTEM,
                user=diagnostic_user,
                model=ModelTier.REASONING,
            ),
            self.gateway.complete_typed(
                TreatmentDraft,
                system=_REPAIR_TREATMENT_SYSTEM,
                user=treatment_user,
                model=ModelTier.REASONING,
            ),
            return_exceptions=True,
        )
        if not isinstance(repaired[0], BaseException):
            diagnostic = repaired[0]
        if not isinstance(repaired[1], BaseException):
            treatment = repaired[1]
        return diagnostic, treatment

    def _assemble(
        self,
        diagnostic: DiagnosticDraft,
        treatment: TreatmentDraft,
        sources: list[KnowledgeSource],
    ) -> DynamicDiseaseRecord:
        return DynamicDiseaseRecord(
            **diagnostic.model_dump(exclude={"diagnostic_limitations"}),
            non_drug_management=treatment.non_drug_management,
            treatment_options=treatment.treatment_options,
            patient_advice=treatment.patient_advice,
            limitations=[
                *diagnostic.diagnostic_limitations,
                *treatment.treatment_limitations,
            ],
            sources=sources,
            generated_by_model=self.model_name,
        )

    @staticmethod
    def _split_record(
        record: DynamicDiseaseRecord,
    ) -> tuple[DiagnosticDraft, TreatmentDraft]:
        diagnostic_fields = {
            "name", "name_ar", "synonyms", "metadata_source_ids", "overview",
            "overview_source_ids", "symptoms", "negative_findings", "history_questions",
            "diagnostic_questions", "diagnostic_criteria", "tests",
            "differential_diagnosis", "red_flags",
        }
        diagnostic = DiagnosticDraft.model_validate({
            **record.model_dump(include=diagnostic_fields),
            "diagnostic_limitations": list(record.limitations),
        })
        treatment = TreatmentDraft(
            non_drug_management=record.non_drug_management,
            treatment_options=record.treatment_options,
            patient_advice=record.patient_advice,
            treatment_limitations=[],
        )
        return diagnostic, treatment

    @staticmethod
    def _author_payload(
        gap: GapTarget,
        sources: list[KnowledgeSource],
        schema: type[DiagnosticDraft] | type[TreatmentDraft],
    ) -> str:
        return json.dumps({
            "target_condition": gap.condition,
            "gap_kind": gap.gap_kind,
            "gap_reason": gap.reason,
            "allowed_evidence": [source.model_dump(mode="json") for source in sources],
            "output_schema": schema.model_json_schema(),
        }, ensure_ascii=False)

    @staticmethod
    def _for_scope(sources: list[KnowledgeSource], scope: str) -> list[KnowledgeSource]:
        return [
            source for source in sources
            if not source.evidence_scope
            or scope in source.evidence_scope
            or "general" in source.evidence_scope
            or (scope == "treatment" and "medication_safety" in source.evidence_scope)
        ]

    @staticmethod
    def _dedupe_sources(sources: list[KnowledgeSource]) -> list[KnowledgeSource]:
        unique: dict[str, KnowledgeSource] = {}
        for source in sources:
            previous = unique.get(source.source_id)
            if previous is None:
                unique[source.source_id] = source
                continue
            scopes = list(dict.fromkeys([*previous.evidence_scope, *source.evidence_scope]))
            unique[source.source_id] = previous.model_copy(update={"evidence_scope": scopes})
        return list(unique.values())

    def _candidates(self, state: ConsultationState) -> list[dict]:
        rows: list[dict] = []
        seen: set[str] = set()
        working = state.visit.working_diagnosis
        if working and working.condition:
            key = normalise_condition(working.condition)
            seen.add(key)
            rows.append({"condition": working.condition, "confidence": working.confidence,
                         "role": "working_diagnosis"})
        for hypothesis in sorted(
            state.visit.differential, key=lambda item: item.confidence, reverse=True
        ):
            key = normalise_condition(hypothesis.condition)
            if not key or key in seen or hypothesis.confidence < self.min_confidence:
                continue
            seen.add(key)
            rows.append({"condition": hypothesis.condition, "confidence": hypothesis.confidence,
                         "role": "differential"})
        return rows

    @staticmethod
    def _sources(items: Iterable[EvidenceItem], *, scope: str) -> list[KnowledgeSource]:
        sources: list[KnowledgeSource] = []
        for index, item in enumerate(items, 1):
            if not (item.summary or "").strip():
                continue
            pmid_match = re.search(r"PMID:(\d+)", item.identifier or "")
            url = (
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"
                if pmid_match else "https://pubmed.ncbi.nlm.nih.gov/"
            )
            sources.append(KnowledgeSource(
                source_id=f"PUBMED_{pmid_match.group(1) if pmid_match else index}",
                source_type="PubMed",
                title=item.title or "Untitled PubMed record",
                identifier=item.identifier or f"PubMed:{index}",
                url=url,
                published_year=item.published_year,
                study_type=item.study_type,
                summary=(item.summary or "")[:2000],
                evidence_scope=[scope],
            ))
        return sources

    def _spawn(self, coroutine, *, label: str) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self._tasks.discard(done)
            if done.cancelled():
                return
            try:
                done.result()
            except Exception:  # noqa: BLE001 - background errors are logged, not patient-facing
                logger.exception("Knowledge curator failed for %s", label)

        task.add_done_callback(finished)

    async def close(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._tasks.clear()

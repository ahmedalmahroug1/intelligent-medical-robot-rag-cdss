"""Filesystem repository for live provisional knowledge and clinician review."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app.config import settings
from app.knowledge.models import (
    DynamicDiseaseRecord,
    KnowledgeProposal,
    KnowledgeProposalStatus,
    proposal_approval_errors,
    record_approval_errors,
    utcnow,
)


class KnowledgeProposalError(RuntimeError):
    pass


def normalise_condition(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def safe_slug(value: str) -> str:
    slug = "_".join(re.findall(r"[a-z0-9]+", value.casefold()))
    return slug[:100] or "condition"


class KnowledgeProposalRepository:
    def __init__(
        self,
        *,
        pending_dir: str | Path | None = None,
        approved_dir: str | Path | None = None,
    ) -> None:
        self.pending_dir = Path(pending_dir or settings.KB_PENDING_DIR)
        self.approved_dir = Path(approved_dir or settings.KB_APPROVED_DIR)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.approved_dir.mkdir(parents=True, exist_ok=True)

    def save(self, proposal: KnowledgeProposal) -> KnowledgeProposal:
        self._write_json(
            self.pending_dir / f"{proposal.proposal_id}.json",
            proposal.model_dump(mode="json"),
        )
        return proposal

    def get(self, proposal_id: str) -> KnowledgeProposal | None:
        path = self.pending_dir / f"{proposal_id}.json"
        if not path.exists():
            return None
        proposal = KnowledgeProposal.model_validate_json(path.read_text(encoding="utf-8"))
        return self._apply_current_gate(proposal)

    def list(self, status: KnowledgeProposalStatus | None = None) -> list[KnowledgeProposal]:
        proposals: list[KnowledgeProposal] = []
        for path in self.pending_dir.glob("*.json"):
            try:
                proposal = KnowledgeProposal.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            proposal = self._apply_current_gate(proposal)
            if status is None or proposal.status is status:
                proposals.append(proposal)
        return sorted(proposals, key=lambda item: item.created_at, reverse=True)

    def has_pending_for(self, condition: str) -> bool:
        target = normalise_condition(condition)
        return any(
            normalise_condition(item.target_condition) == target
            for item in self.list(KnowledgeProposalStatus.PENDING)
        )

    def edit(
        self,
        proposal_id: str,
        *,
        record: DynamicDiseaseRecord,
        reviewed_by: str,
        notes: str | None = None,
    ) -> KnowledgeProposal:
        proposal = self._pending(proposal_id)
        errors = record_approval_errors(record)
        updated = proposal.model_copy(update={
            "record": record,
            # A clinician edit supersedes the machine verifier's review of the
            # old version. The deterministic content gate is still applied; the
            # explicit clinician action is the authority for the edited content.
            "verification": None,
            "approvable": not errors,
            "generation_error": "; ".join(errors) if errors else None,
            "reviewed_by": reviewed_by,
            "reviewer_notes": notes,
        })
        return self.save(updated)

    def approve(
        self,
        proposal_id: str,
        *,
        reviewed_by: str,
        notes: str | None = None,
        record: DynamicDiseaseRecord | None = None,
    ) -> KnowledgeProposal:
        proposal = self._pending(proposal_id)
        chosen = record or proposal.record
        candidate = proposal.model_copy(update={"record": chosen})
        errors = proposal_approval_errors(candidate)
        if chosen is None or errors:
            raise KnowledgeProposalError(
                "Proposal has no fully traceable, approvable record. Edit or regenerate it first. "
                + "; ".join(errors)
            )
        reviewed = proposal.model_copy(update={
            "record": chosen,
            "status": KnowledgeProposalStatus.APPROVED,
            "reviewed_at": utcnow(),
            "reviewed_by": reviewed_by,
            "reviewer_notes": notes,
        })
        # The active retriever reads only this directory and the seed directory.
        # Write the active record first; then persist the review decision.
        active_doc = {
            "approval": {
                "proposal_id": proposal_id,
                "reviewed_by": reviewed_by,
                "reviewed_at": reviewed.reviewed_at.isoformat() if reviewed.reviewed_at else None,
                "notes": notes,
            },
            "record": chosen.model_dump(mode="json"),
        }
        self._write_json(self.approved_dir / f"{safe_slug(chosen.name)}.json", active_doc)
        return self.save(reviewed)

    def reject(
        self,
        proposal_id: str,
        *,
        reviewed_by: str,
        notes: str | None = None,
    ) -> KnowledgeProposal:
        proposal = self._pending(proposal_id)
        rejected = proposal.model_copy(update={
            "status": KnowledgeProposalStatus.REJECTED,
            "reviewed_at": utcnow(),
            "reviewed_by": reviewed_by,
            "reviewer_notes": notes,
        })
        return self.save(rejected)

    def delete(
        self,
        proposal_id: str,
        *,
        reviewed_by: str,
        notes: str | None = None,
    ) -> KnowledgeProposal:
        """Remove an addition from the operational corpus while retaining its audit."""
        proposal = self.get(proposal_id)
        if proposal is None:
            raise KnowledgeProposalError(f"Unknown knowledge proposal '{proposal_id}'.")
        if proposal.status is KnowledgeProposalStatus.DELETED:
            raise KnowledgeProposalError(f"Proposal '{proposal_id}' is already deleted.")
        if proposal.record is not None:
            (self.approved_dir / f"{safe_slug(proposal.record.name)}.json").unlink(
                missing_ok=True,
            )
        deleted = proposal.model_copy(update={
            "status": KnowledgeProposalStatus.DELETED,
            "reviewed_at": utcnow(),
            "reviewed_by": reviewed_by,
            "reviewer_notes": notes,
        })
        return self.save(deleted)

    def _pending(self, proposal_id: str) -> KnowledgeProposal:
        proposal = self.get(proposal_id)
        if proposal is None:
            raise KnowledgeProposalError(f"Unknown knowledge proposal '{proposal_id}'.")
        if proposal.status is not KnowledgeProposalStatus.PENDING:
            raise KnowledgeProposalError(
                f"Proposal '{proposal_id}' is already {proposal.status.value}."
            )
        return proposal

    @staticmethod
    def _apply_current_gate(proposal: KnowledgeProposal) -> KnowledgeProposal:
        if proposal.status is not KnowledgeProposalStatus.PENDING or proposal.record is None:
            return proposal
        errors = proposal_approval_errors(proposal)
        return proposal.model_copy(update={
            "approvable": not errors,
            "generation_error": "; ".join(errors) if errors else None,
        })

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # OneDrive/Windows may hold the destination for a few milliseconds while
        # indexing it. Retry the atomic replace, then fall back to an ordinary
        # overwrite so a review action does not fail solely because of that lock.
        for attempt in range(3):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.02)
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink(missing_ok=True)

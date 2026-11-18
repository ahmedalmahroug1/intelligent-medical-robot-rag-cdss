"""Hot-reloading retrieval over seed, provisional, and approved knowledge."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from app.config import settings
from app.core.state import ConsultationState
from app.knowledge.models import DynamicDiseaseRecord
from app.knowledge.repository import normalise_condition


_STOP = {
    "the", "and", "with", "that", "this", "from", "have", "has", "had", "was",
    "were", "are", "for", "not", "but", "you", "your", "patient", "none", "normal",
    "about", "into", "when", "what", "which", "their", "they", "them", "been", "also",
}


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.split(r"[^a-z0-9]+", (value or "").casefold())
        if len(token) > 2 and token not in _STOP
    }


def _text(value: object, limit: int = 900) -> str:
    if value is None:
        return ""
    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return " ".join(rendered.split())[:limit]


class ClinicalKnowledgeBase:
    """Operational view of local knowledge with explicit review-state labels.

    Valid curator records are available to diagnosis as soon as they are
    generated.  They remain visibly labelled ``unapproved`` until a clinician
    approves them. Rejected or deleted records are excluded immediately.
    """

    def __init__(
        self,
        *,
        seed_dir: str | Path | None = None,
        pending_dir: str | Path | None = None,
        approved_dir: str | Path | None = None,
        top_n: int | None = None,
        relative_gate: float | None = None,
    ) -> None:
        self.seed_dir = Path(seed_dir or settings.KB_SEED_DIR)
        self.pending_dir = Path(pending_dir or settings.KB_PENDING_DIR)
        self.approved_dir = Path(approved_dir or settings.KB_APPROVED_DIR)
        self.top_n = top_n or settings.DYNAMIC_KB_TOPN
        self.relative_gate = relative_gate or settings.DYNAMIC_KB_GATE
        self._signature: tuple = ()
        self._records: list[dict] = []
        self._idf: dict[str, float] = {}
        self._refresh(force=True)

    @property
    def record_count(self) -> int:
        self._refresh()
        return len(self._records)

    def inventory(self) -> list[dict]:
        self._refresh()
        return [
            {
                "name": record["name"],
                "synonyms": list(record["synonyms"]),
                "origin": record["origin"],
                "review_status": record["review_status"],
                "proposal_id": record.get("proposal_id"),
                "gaps": self._gaps(record["data"], dynamic=record["dynamic"]),
            }
            for record in self._records
        ]

    def exact_record(self, condition: str) -> dict | None:
        self._refresh()
        target = normalise_condition(condition)
        for record in self._records:
            names = [record["name"], *record["synonyms"]]
            if any(normalise_condition(name) == target for name in names):
                return record
        return None

    def retrieve(self, query: str) -> list[dict]:
        self._refresh()
        query_tokens = _tokens(query)
        scored: list[tuple[float, dict]] = []
        for record in self._records:
            score = sum(self._idf.get(token, 0.0) for token in record["tokens"] & query_tokens)
            if score > 0:
                scored.append((score, record))
        if not scored:
            return []
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = scored[0][0]
        return [
            record for score, record in scored[: self.top_n]
            if score >= self.relative_gate * best
        ]

    def guidance(self, query: str) -> str:
        records = self.retrieve(query)
        if not records:
            return "No sufficiently relevant operational KB record was found. Use patient evidence alone."
        body = "\n\n--- LOCAL KB RECORD ---\n".join(
            self._render(record) for record in records
        )
        return (
            "These are retrieved decision-support records, not ground truth. Every record carries "
            "a review status. UNAPPROVED records may inform the differential, but their claims must "
            "be corroborated against the patient and cited evidence. Ignore non-matching records "
            "and decide whether to revise the differential, ask a discriminating question, request "
            "a test, or stop. Retrieval alone is never sufficient reason to diagnose.\n\n" + body
        )

    @staticmethod
    def query_from_state(state: ConsultationState, latest: str = "") -> str:
        transcript = "\n".join(
            f"{message.role.value}: {message.content}" for message in state.session.transcript
        )
        differential = " ".join(item.condition for item in state.visit.differential)
        return f"{latest}\n{transcript}\n{differential}"

    def _refresh(self, force: bool = False) -> None:
        paths = [
            *(sorted(self.seed_dir.glob("*.json")) if self.seed_dir.exists() else []),
            *(sorted(self.pending_dir.glob("*.json")) if self.pending_dir.exists() else []),
            *(sorted(self.approved_dir.glob("*.json")) if self.approved_dir.exists() else []),
        ]
        signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        if not force and signature == self._signature:
            return
        by_name: dict[str, dict] = {}
        for path in paths:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                dynamic = (
                    "record" in raw
                    and isinstance(raw.get("record"), dict)
                    and raw["record"].get("schema_version") == "dynamic-1.0"
                )
                if dynamic and "proposal_id" in raw:
                    # The proposal file doubles as the live provisional record.
                    # Rejection/deletion removes it from retrieval while keeping
                    # the complete research and review audit.
                    if (
                        raw.get("status", "pending") != "pending"
                        or raw.get("approvable") is not True
                    ):
                        continue
                    data = raw["record"]
                    review_status = "unapproved"
                    proposal_id = raw.get("proposal_id")
                elif dynamic:
                    data = raw["record"]
                    review_status = "clinician_approved"
                    proposal_id = raw.get("approval", {}).get("proposal_id")
                else:
                    data = raw
                    review_status = "seed"
                    proposal_id = None
                record = self._adapt(
                    data,
                    dynamic=dynamic,
                    origin=str(path),
                    review_status=review_status,
                    proposal_id=proposal_id,
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
            # A provisional dynamic record may improve a weak seed immediately;
            # a later clinician-approved copy overrides both.
            by_name[normalise_condition(record["name"])] = record
        self._records = list(by_name.values())
        frequency: dict[str, int] = {}
        for record in self._records:
            for token in record["tokens"]:
                frequency[token] = frequency.get(token, 0) + 1
        count = len(self._records)
        self._idf = {
            token: math.log((count + 1) / (seen + 0.5))
            for token, seen in frequency.items()
        }
        self._signature = signature

    def _adapt(
        self,
        data: dict,
        *,
        dynamic: bool,
        origin: str,
        review_status: str,
        proposal_id: str | None,
    ) -> dict:
        if dynamic:
            parsed = DynamicDiseaseRecord.model_validate(data)
            searchable = [parsed.name, *parsed.synonyms, parsed.overview]
            searchable.extend(item.text for item in parsed.symptoms)
            searchable.extend(item.question for item in parsed.history_questions)
            searchable.extend(item.question for item in parsed.diagnostic_questions)
            searchable.extend(item.condition for item in parsed.differential_diagnosis)
            return {
                "name": parsed.name,
                "synonyms": tuple(parsed.synonyms),
                "tokens": _tokens(" ".join(searchable)),
                "data": parsed.model_dump(mode="json"),
                "dynamic": True,
                "origin": origin,
                "review_status": review_status,
                "proposal_id": proposal_id,
            }

        metadata = data["metadata"]
        searchable = [metadata["name"], *metadata.get("synonyms", []), *metadata.get("abbreviations", [])]
        searchable.extend(item.get("complaint", "") for item in data.get("chief_complaints", []))
        searchable.extend(item.get("symptom", "") for item in data.get("symptoms", []))
        searchable.extend(item.get("question", "") for item in data.get("history_questions", []))
        searchable.extend(item.get("question", "") for item in data.get("diagnostic_questions", []))
        searchable.extend(item.get("disease", "") for item in data.get("differential_diagnosis", []))
        return {
            "name": metadata["name"],
            "synonyms": tuple([*metadata.get("synonyms", []), *metadata.get("abbreviations", [])]),
            "tokens": _tokens(" ".join(searchable)),
            "data": data,
            "dynamic": False,
            "origin": origin,
            "review_status": review_status,
            "proposal_id": proposal_id,
        }

    def _render(self, record: dict) -> str:
        data = record["data"]
        if record["review_status"] == "unapproved":
            review_note = (
                "UNAPPROVED - machine-generated and awaiting clinician review. "
                "It may support diagnostic reasoning, but must be corroborated."
            )
        elif record["review_status"] == "clinician_approved":
            review_note = "CLINICIAN APPROVED."
        else:
            review_note = "LOCAL SEED RECORD."
        if record["dynamic"]:
            symptoms = "; ".join(f"{x['text']} [{x.get('diagnostic_value', '')}]" for x in data["symptoms"][:6])
            negatives = "; ".join(x["text"] for x in data.get("negative_findings", [])[:4])
            questions = "; ".join(x["question"] for x in [*data["diagnostic_questions"], *data["history_questions"]][:7])
            differentials = "; ".join(f"vs {x['condition']}: {x.get('distinguishing_features', '')}" for x in data["differential_diagnosis"][:4])
            tests = "; ".join(f"{x['name']}: {x.get('indication', '')}" for x in data["tests"][:4])
            red_flags = "; ".join(x["text"] for x in data["red_flags"][:4])
            criteria = "; ".join(x["text"] for x in data["diagnostic_criteria"])
            source_ids = [source["source_id"] for source in data["sources"]]
            overview = data["overview"]
        else:
            symptoms = "; ".join(
                f"{x.get('symptom', '')} [{x.get('diagnostic_value', '')}]"
                for x in data.get("symptoms", [])[:6]
            )
            negatives = "; ".join(x.get("finding", "") for x in data.get("negative_findings", [])[:4])
            questions = "; ".join(
                x.get("question", "")
                for x in [*data.get("diagnostic_questions", []), *data.get("history_questions", [])][:7]
            )
            differentials = "; ".join(
                f"vs {x.get('disease', '')}: {x.get('differences', '')}"
                for x in data.get("differential_diagnosis", [])[:4]
            )
            tests = "; ".join(
                f"{x.get('test', '') or x.get('test_name', '')}: {x.get('why', '') or x.get('diagnostic_value', '')}"
                for x in [*data.get("next_best_test", []), *data.get("laboratory_tests", [])][:4]
            )
            red_flags = "; ".join(x.get("finding", "") for x in data.get("red_flags", [])[:4])
            criteria = _text(data.get("diagnostic_criteria", {}), 1000)
            source_ids = data.get("evidence", {}).get("verified_source_ids", [])
            overview = data.get("overview", {}).get("short_description", "")
        return (
            f"CONDITION: {record['name']}\n"
            f"REVIEW STATUS: {review_note}\n"
            f"Overview: {_text(overview)}\n"
            f"Typical/diagnostic clues: {_text(symptoms, 1200)}\n"
            f"Important negatives: {_text(negatives)}\n"
            f"Discriminating questions: {_text(questions, 1200)}\n"
            f"Distinguish from: {_text(differentials, 1200)}\n"
            f"Diagnostic criteria: {_text(criteria, 1000)}\n"
            f"Useful tests: {_text(tests)}\n"
            f"Red flags: {_text(red_flags)}\n"
            f"Source IDs: {_text(source_ids, 500)}"
        )

    @staticmethod
    def _gaps(data: dict, *, dynamic: bool) -> list[str]:
        if dynamic:
            checks = {
                "symptoms": len(data.get("symptoms", [])) >= 3,
                "diagnostic_questions": bool(data.get("diagnostic_questions")),
                "differentials": len(data.get("differential_diagnosis", [])) >= 2,
                "tests": bool(data.get("tests")),
                "red_flags": bool(data.get("red_flags")),
                "sources": bool(data.get("sources")),
                "treatment": bool(data.get("treatment_options") or data.get("non_drug_management")),
            }
        else:
            checks = {
                "symptoms": len(data.get("symptoms", [])) >= 3,
                "diagnostic_questions": bool(data.get("diagnostic_questions")),
                "differentials": len(data.get("differential_diagnosis", [])) >= 3,
                "tests": bool(data.get("laboratory_tests") or data.get("next_best_test")),
                "red_flags": len(data.get("red_flags", [])) >= 2,
                "sources": bool(data.get("evidence", {}).get("verified_source_ids")),
                "treatment": bool(data.get("treatment_guidance", {}).get("medication_options")),
            }
        return [name for name, complete in checks.items() if not complete]

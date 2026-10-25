"""Validate a blind audit and prepare source-grounded proposals without applying labels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from erag.pipeline import atomic_json

ROOT = Path("artifacts/label-audit")


def verify_audit(root: Path = ROOT) -> dict:
    provenance = json.loads((root / "audit-provenance.json").read_text())
    manifest = json.loads((root / "inputs/manifest.json").read_text())
    for name, expected in manifest["input_sha256"].items():
        if hashlib.sha256((root / "inputs" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"blind input changed: {name}")
    for name, expected in provenance["source_sha256"].items():
        if hashlib.sha256((Path("data/stage-1.1") / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"original source changed: {name}")
    for name, expected in provenance["original_label_sha256"].items():
        if hashlib.sha256((Path("data/stage-1.1") / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"original label file changed: {name}")
    questions = {
        q["question_id"]: q for q in json.loads((root / "inputs/questions.json").read_text())
    }
    units = {u["unit_id"]: u for u in json.loads((root / "inputs/units.json").read_text())}
    audit_path = root / "review/independent-judgments.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("input_hashes") != manifest["input_sha256"]:
        raise ValueError("auditor input hashes mismatch")
    if audit.get("coverage_question_count") != len(questions):
        raise ValueError("declared coverage mismatch")
    rows = audit["judgments"]
    actual_ids = [row["question_id"] for row in rows]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(questions):
        raise ValueError("question coverage is missing, duplicated or unknown")
    answerability_values = {"answerable", "partially_answerable", "unanswerable", "ambiguous"}
    coverage_values = {3: "full", 2: "partial", 1: "supporting"}
    positive_count = 0
    for row in rows:
        qid = row["question_id"]
        if row["answerability"] not in answerability_values:
            raise ValueError(f"invalid answerability: {qid}")
        if type(row["requires_human_review"]) is not bool:
            raise ValueError(f"review flag must be boolean: {qid}")
        if not isinstance(row["expected_facts"], list):
            raise ValueError(f"facts must be a list: {qid}")  # noqa: TRY004
        seen = set()
        for positive in row["positives"]:
            uid, grade = positive["unit_id"], positive["grade"]
            if uid not in units or uid in seen:
                raise ValueError(f"invalid or duplicate source unit: {qid}/{uid}")
            seen.add(uid)
            if type(grade) is not int or grade not in coverage_values:
                raise ValueError(f"invalid grade: {qid}/{uid}")
            if positive["coverage"] != coverage_values[grade]:
                raise ValueError(f"grade/coverage disagreement: {qid}/{uid}")
            evidence = positive["evidence"]
            if not isinstance(evidence, list) or not evidence:
                raise ValueError(f"missing source evidence: {qid}/{uid}")
            for quote in evidence:
                if (
                    not isinstance(quote, str)
                    or not quote.strip()
                    or quote not in units[uid]["text"]
                ):
                    raise ValueError(f"unsupported quotation: {qid}/{uid}: {quote!r}")
            if not positive["rationale"].strip():
                raise ValueError(f"missing rationale: {qid}/{uid}")
            positive_count += 1
        if row["answerability"] == "unanswerable" and row["positives"]:
            raise ValueError(f"unanswerable case has positive judgments: {qid}")
    return {
        "validated": True,
        "questions": len(rows),
        "source_units": len(units),
        "positive_judgments": positive_count,
        "answerability": dict(Counter(row["answerability"] for row in rows)),
        "human_review_flags": sum(row["requires_human_review"] for row in rows),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "originals_unchanged": True,
    }


def build_review(root: Path = ROOT) -> dict:
    validation = verify_audit(root)
    private = json.loads((root / "private-mapping.json").read_text())
    questions = {
        q["question_id"]: q for q in json.loads((root / "inputs/questions.json").read_text())
    }
    units = {u["unit_id"]: u for u in json.loads((root / "inputs/units.json").read_text())}
    audit = json.loads((root / "review/independent-judgments.json").read_text())
    comparisons = []
    counts = Counter()
    for row in sorted(audit["judgments"], key=lambda row: row["question_id"]):
        qid = row["question_id"]
        original = private["questions"][qid]["original_case"]
        grades = {}
        original_evidence = {}
        for evidence in original["evidence"]:
            matches = [
                uid
                for uid, unit in units.items()
                if unit["document_id"] == evidence["document_id"]
                and evidence["text"] in unit["text"]
            ]
            if not matches:
                raise ValueError(f"original evidence not covered by audit units: {qid}")
            for uid in matches:
                grades[uid] = max(grades.get(uid, 0), evidence["grade"])
                original_evidence.setdefault(uid, []).append(evidence["text"])
        proposed = {p["unit_id"]: p["grade"] for p in row["positives"]}
        changes = []
        for uid in sorted(grades.keys() | proposed.keys()):
            before, after = grades.get(uid, 0), proposed.get(uid, 0)
            if before == after:
                continue
            kind = (
                "added_full_answer"
                if before == 0 and after == 3
                else "added_partial_evidence"
                if before == 0 and after == 2
                else "added_supporting_evidence"
                if before == 0
                else "removed_positive"
                if after == 0
                else "grade_change"
            )
            counts[kind] += 1
            judgment = next((p for p in row["positives"] if p["unit_id"] == uid), None)
            changes.append(
                {
                    "kind": kind,
                    "unit_id": uid,
                    "chunk_id": private["units"][uid]["id"],
                    "document_id": units[uid]["document_id"],
                    "before_grade": before,
                    "proposed_grade": after,
                    "original_evidence": original_evidence.get(uid, []),
                    "proposed_judgment": judgment,
                    "source_text": units[uid]["text"],
                }
            )
        original_answerable = bool(original["evidence"])
        proposed_answerable = row["answerability"] != "unanswerable"
        comparison = {
            "question_id": qid,
            "case_ids": private["questions"][qid]["case_ids"],
            "source_file": private["questions"][qid]["source_file"],
            "query": questions[qid]["query"],
            "original_unit_grades": grades,
            "proposed_unit_grades": proposed,
            "answerability": row["answerability"],
            "answerability_changed": original_answerable != proposed_answerable,
            "expected_facts": row["expected_facts"],
            "changes": changes,
            "ambiguity": row["ambiguity"],
            "requires_human_review": row["requires_human_review"],
            "proposed_labels_applied": False,
        }
        comparisons.append(comparison)
    changed = [
        row
        for row in comparisons
        if row["changes"] or row["answerability_changed"] or row["requires_human_review"]
    ]
    summary = {
        **validation,
        "questions_with_label_differences": sum(bool(c["changes"]) for c in comparisons),
        "answerability_differences": sum(c["answerability_changed"] for c in comparisons),
        "change_counts": dict(counts),
        "review_queue_questions": len(changed),
        "applied_to_original_labels": False,
        "retrieval_or_tuning_executed": False,
        "quality_gate_recalculated": False,
    }
    atomic_json(root / "review/validated-summary.json", summary)
    atomic_json(root / "review/label-comparison.json", comparisons)
    atomic_json(root / "review/review-queue.json", changed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build_review(args.root), indent=2))


if __name__ == "__main__":
    main()

"""Reproducible baseline scoring from explicit reviewed annotations; no model calls."""

import hashlib
from pathlib import Path

from .answer_metrics import (
    citation_provenance,
    read_json,
    score_judgment,
    summarize,
    validate_judgment,
)
from .ingestion import ingest
from .pipeline import atomic_json, corpus_hash


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_reviewed_answers(answers_path, references_path, reviews_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError("preserve earlier reviewed baselines; choose a new output")
    answers, refs, reviews = (read_json(p) for p in (answers_path, references_path, reviews_path))
    if (
        reviews["answer_input_sha256"] != file_hash(answers_path)
        or reviews["reference_sha256"] != file_hash(references_path)
        or answers["dataset_hash"] != refs["parent_dataset_sha256"]
    ):
        raise ValueError("review annotations do not match the frozen answer/reference inputs")
    if (
        reviews["review_kind"] not in {"agent-reviewed", "human-reviewed"}
        or not reviews["reviewer"]
    ):
        raise ValueError("review provenance is required")
    documents = ingest(
        answers["retrieval_config"]["dataset"], answers["retrieval_config"].get("stage", "1.1")
    )
    if corpus_hash(documents) != answers["corpus_hash"]:
        raise ValueError("review corpus differs from the original answer run")
    docs = {d.id: d for d in documents}
    cases = {c["id"]: c for c in refs["cases"]}
    annotations = {c["id"]: c for c in reviews["cases"]}
    ids = [c["id"] for c in answers["cases"]]
    if (
        len(cases) != len(refs["cases"])
        or len(annotations) != len(reviews["cases"])
        or len(ids) != len(set(ids))
        or set(ids) != set(cases)
        or set(ids) != set(annotations)
    ):
        raise ValueError("every question requires exactly one reviewed annotation")
    rows = []
    for original in answers["cases"]:
        case, annotation = cases[original["id"]], annotations[original["id"]]
        result = original["result"]
        if (
            original["query"] != case["query"]
            or original["query"] != annotation["query"]
            or original["expected_status"] != case["expected_status"]
            or annotation["answer_claims"] != [c["text"] for c in result["claims"]]
            or annotation["required_facts"] != case["required_facts"]
            or not annotation["review_note"].strip()
        ):
            raise ValueError(
                "review question, claim text or required facts differ from frozen inputs"
            )
        for evidence in case["source_evidence"]:
            if evidence["text"] not in docs[evidence["document_id"]].text:
                raise ValueError("reference evidence absent from original source")
        status = result["status"]
        row = {
            "id": original["id"],
            "expected_status": case["expected_status"],
            "actual_status": status,
            "abstention_decision_correct": status == case["expected_status"],
            "review_note": annotation["review_note"],
        }
        if status == "answered":
            if not result["claims"] or any(not c["citations"] for c in result["claims"]):
                raise ValueError("answered case requires cited claims")
            value = validate_judgment(
                annotation["judgment"], result["claims"], case["required_facts"]
            )
            provenance = citation_provenance(result, docs)
            row.update(
                judgment=value,
                citation_provenance=provenance,
                scores=score_judgment(value, provenance),
            )
        elif annotation["judgment"] is not None:
            raise ValueError("unanswered cases must not have factual judgments")
        rows.append(row)
    report = {
        "phase": "answer-" + answers["retrieval_config"].get("stage", "1.1"),
        "documents": len(documents),
        "answer_model": answers["model"],
        "answer_model_digest": answers["model_digest"],
        "review_kind": reviews["review_kind"],
        "reviewer": reviews["reviewer"],
        "review_rubric": reviews["rubric"],
        "answer_input_sha256": file_hash(answers_path),
        "reference_sha256": file_hash(references_path),
        "review_sha256": file_hash(reviews_path),
        "corpus_hash": answers["corpus_hash"],
        "summary": summarize(rows),
        "cases": rows,
        "limitations": [
            "Agent-reviewed development baseline, not independent human validation.",
            f"Observed {len(rows)}-case set; no production or held-out accuracy claim.",
            "Strict qualifier interpretation and atomic decomposition affect scores.",
        ],
        "quality_gate": "No new acceptance thresholds applied.",
    }
    atomic_json(output, report)
    return report

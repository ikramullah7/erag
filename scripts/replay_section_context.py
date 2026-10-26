"""Replay recorded source selections through deterministic section completion; no model calls."""

import argparse
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answer_evaluation import summarize_answer_rows
from erag.answer_metrics import citation_provenance
from erag.config import digest
from erag.ingestion import ingest
from erag.pipeline import atomic_json, corpus_hash
from erag.section_context import complete_sections


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("answers", "cases", "snapshot", "answer-config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve earlier evaluations; choose a new output")
    source = json.loads(args.answers.read_text())
    cases = json.loads(args.cases.read_text())
    snapshot = json.loads(args.snapshot.read_text())
    cfg = load_answer_config(args.answer_config)
    if not cfg.section_context_completion:
        raise ValueError("replay requires section_context_completion")
    old = {**source["config"], "section_context_completion": True}
    if old != asdict(cfg):
        raise ValueError("only the deterministic completion flag may change")
    if source["dataset_hash"] != hashlib.sha256(args.cases.read_bytes()).hexdigest():
        raise ValueError("question inputs differ from the recorded model run")
    docs = ingest(source["retrieval_config"]["dataset"], source["retrieval_config"]["stage"])
    if (
        source["corpus_hash"] != snapshot["corpus_hash"]
        or snapshot["corpus_hash"] != corpus_hash(docs)
        or source["chunk_hash"] != snapshot["chunk_hash"]
        or snapshot["chunk_hash"] != digest(snapshot["chunks"])
    ):
        raise ValueError("recorded run, snapshot and current corpus differ")
    refs = {c["id"]: c for c in cases["cases"]}
    rows = deepcopy(source["cases"])
    if len(refs) != len(rows) or set(refs) != {c["id"] for c in rows}:
        raise ValueError("each recorded case needs exactly one original input")
    for case in rows:
        ref, before = refs[case["id"]], case["result"]
        if (case["query"], case["expected_status"]) != (ref["query"], ref["expected_status"]):
            raise ValueError("question or expected status changed")
        case["result"] = complete_sections(before, snapshot)
        after = case["result"]
        # Every original source assertion must remain covered by an exact same-version superset.
        for claim in before["claims"]:
            for cite in claim["citations"]:
                if not any(
                    e["document_id"] == cite["document_id"]
                    and e["version"] == cite["version"]
                    and e["start"] <= cite["start"]
                    and e["end"] >= cite["end"]
                    for c in after["claims"]
                    for e in c["citations"]
                ):
                    raise ValueError("completion dropped a selected source assertion")
        provenance = citation_provenance(after, {d.id: d for d in docs})
        case["citations_valid"] = bool(provenance) and all(all(row) for row in provenance)
        case["fact_checks"] = [
            any(re.search(p, after["answer"], re.IGNORECASE) for p in alternatives)
            for alternatives in ref["fact_patterns"]
        ]
        case["passed"] = after["status"] == ref["expected_status"] and (
            after["status"] != "answered" or (case["citations_valid"] and all(case["fact_checks"]))
        )
    report = deepcopy(source)
    report.update(
        config=asdict(cfg),
        cases=rows,
        evaluation_kind="Observed development questions; recorded local model generations and deterministic bounded source-section completion. No new model calls or held-out claim.",
        timing_scope="Parent cached retrieval and actual model generation reused; only section_context_ms is newly measured.",
        completion_provenance={
            "parent_answers": str(args.answers),
            "model_calls": 0,
            "original_selected_evidence_preserved": True,
            "sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    args.answers,
                    args.cases,
                    args.snapshot,
                    args.answer_config,
                    Path(__file__),
                    Path("src/erag/section_context.py"),
                )
            },
        },
    )
    report["summary"], report["checks"] = summarize_answer_rows(rows)
    report["passed"] = all(report["checks"].values())
    atomic_json(args.output, report)
    print(json.dumps({"cases": len(rows), "checks": report["checks"], "model_calls": 0}))


if __name__ == "__main__":
    main()

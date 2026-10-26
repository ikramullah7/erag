"""Fresh coverage review of frozen source-fact selections; never a held-out claim."""

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answer_evaluation import summarize_answer_rows
from erag.answer_metrics import citation_provenance
from erag.config import load_config
from erag.evaluation import environment
from erag.evidence_answer import REVIEW_SYSTEM, apply_coverage_review
from erag.generators import create_generator
from erag.ingestion import ingest
from erag.pipeline import atomic_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--answer-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", nargs="+")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise ValueError("preserve earlier coverage reviews")
    root.mkdir(parents=True)
    raw = Path(args.answers).read_bytes()
    source = json.loads(raw)
    cfg = replace(load_answer_config(args.answer_config), coverage_check=True)
    if source.get("provider", "ollama") != cfg.provider:
        raise ValueError("coverage provider must match the primary answer provider")
    retrieval_cfg = load_config(cfg.retrieval_config)
    docs = {d.id: d for d in ingest(retrieval_cfg.dataset, retrieval_cfg.stage)}
    cases = deepcopy(source["cases"])
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [c for c in cases if c["id"] in selected]
        if len(cases) != len(selected):
            raise ValueError("unknown/uncompleted coverage probe ID")
    import re

    expected = json.loads(Path("data/real-world/stage1/answer-evaluation-v1.json").read_text())
    references = {c["id"]: c for c in expected["cases"]}
    generator = create_generator(cfg)
    for case in cases:
        ref = references[case["id"]]
        if ref["query"] != case["query"] or ref["expected_status"] != case["expected_status"]:
            raise ValueError("question/label differs from frozen evaluation input")
        citation_provenance(case["result"], docs)
        case["result"] = apply_coverage_review(case["result"], generator, cfg)
        provenance = citation_provenance(case["result"], docs)
        case["fact_checks"] = [
            any(re.search(p, case["result"]["answer"], re.IGNORECASE) for p in alternatives)
            for alternatives in ref["fact_patterns"]
        ]
        case["citations_valid"] = (
            case["result"]["status"] == "answered"
            and bool(provenance)
            and all(all(row) for row in provenance)
        )
        case["passed"] = (
            (
                case["result"]["status"] == "answered"
                and all(case["fact_checks"])
                and case["citations_valid"]
            )
            if ref["expected_status"] == "answered"
            else case["result"]["status"] == "insufficient_evidence"
        )
        case["primary_generation_reused"] = True
        atomic_json(
            root / "answers.partial.json",
            {"completed": sum("primary_generation_reused" in c for c in cases), "cases": cases},
        )
        print(case["id"], case["result"]["status"], case["passed"], flush=True)
    report = {
        "evaluation_kind": "Observed-set development replay; frozen primary generation and fresh coverage review, no fresh retrieval/full-pipeline latency claim.",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "input_path": args.answers,
        "provider": cfg.provider,
        "phase": "answer-" + retrieval_cfg.stage,
        "documents": len(docs),
        "model": source.get("model"),
        "model_digest": source.get("model_digest"),
        "dataset_hash": source.get("dataset_hash"),
        "corpus_hash": source.get("corpus_hash"),
        "chunk_hash": source.get("chunk_hash"),
        "retrieval_config": asdict(retrieval_cfg),
        "environment": environment(),
        "coverage_prompt_sha256": hashlib.sha256(REVIEW_SYSTEM.encode()).hexdigest(),
        "config": asdict(cfg),
        "cases": cases,
    }
    if not args.case_ids:
        if len(cases) != 22:
            raise ValueError("full coverage evaluation requires all22 cases")
        report["summary"], report["checks"] = summarize_answer_rows(cases)
        report["passed"] = all(report["checks"].values())
    atomic_json(root / "answers.json", report)
    (root / "answers.partial.json").unlink()


if __name__ == "__main__":
    main()

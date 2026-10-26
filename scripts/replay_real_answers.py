"""Fresh local generation over hash-validated cached real retrieval; development only."""

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answer_evaluation import evaluate_answers, format_answer_checks
from erag.chunking import validate_chunks
from erag.config import digest, load_config
from erag.generators import create_generator
from erag.ingestion import ingest
from erag.models import CrossEncoderReranker
from erag.pipeline import atomic_json, corpus_hash
from erag.types import Chunk


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--retrieval-run", required=True)
    parser.add_argument("--answer-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", nargs="+")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise ValueError("preserve prior experiments; choose a new output directory")
    root.mkdir(parents=True)
    answer_cfg = load_answer_config(args.answer_config)
    cfg = load_config(answer_cfg.retrieval_config)
    source = Path(args.retrieval_run)
    frozen = json.loads((source / "frozen-inputs.json").read_text())
    if frozen["retrieval"] != asdict(cfg):
        raise ValueError("cached retrieval belongs to a different configuration")
    snapshot = json.loads((Path(cfg.state) / "manifest.json").read_text())
    docs = ingest(cfg.dataset, cfg.stage)
    if (
        snapshot["config_hash"] != cfg.fingerprint
        or snapshot["corpus_hash"] != corpus_hash(docs)
        or snapshot["chunk_hash"] != digest(snapshot["chunks"])
    ):
        raise ValueError("source/index hashes do not match cached retrieval")
    by_id = {c["id"]: c for c in snapshot["chunks"]}
    for doc in docs:
        validate_chunks(
            doc,
            [Chunk(**c) for c in snapshot["chunks"] if c["document_id"] == doc.id],
            cfg.max_words,
        )
    ranking = json.loads((source / "heldout.json").read_text())
    cached = {}
    for row in ranking["queries"]:
        if row["mode"] == "hybrid" and row["reranked"]:
            diagnostics = row["diagnostics"]
            if any(r["chunk"] != by_id.get(r["chunk"]["id"]) for r in diagnostics["results"]):
                raise ValueError("cached evidence differs from indexed source chunks")
            cached[diagnostics["query"]] = diagnostics

    class Cached:
        def __init__(self):
            self.cfg, self.snapshot = cfg, snapshot
            self.reranker = CrossEncoderReranker(cfg) if answer_cfg.fact_candidates else None

        def search(self, query, mode, rerank, top_k):
            return deepcopy(cached[query])

    cases_path = Path("data/real-world/stage1/answer-evaluation-v1.json")
    if args.case_ids:
        dataset = json.loads(cases_path.read_text())
        selected = set(args.case_ids)
        dataset["cases"] = [c for c in dataset["cases"] if c["id"] in selected]
        if len(dataset["cases"]) != len(selected):
            raise ValueError("unknown diagnostic case ID")
        cases_path = root / "selected-cases.json"
        atomic_json(cases_path, dataset)
    report = evaluate_answers(
        Cached(), create_generator(answer_cfg), answer_cfg, cases_path, root / "answers.json"
    )
    report["evaluation_kind"] = (
        "observed-set development replay; fresh generation, cached retrieval"
    )
    report["prompt_version"] = sorted(
        {c["result"].get("prompt_version", "provider-error") for c in report["cases"]}
    )
    atomic_json(root / "answers.json", report)
    print(format_answer_checks(report))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

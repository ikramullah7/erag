"""Frozen five-real-article retrieval comparison and installed-14B answer evaluation."""

import argparse
import hashlib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answer_evaluation import evaluate_answers, format_answer_checks
from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.evaluation import evaluate, format_target_table, quality_gate
from erag.generators import create_generator
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/real-world/stage1/run-001")
    parser.add_argument("--retrieval-config", default="config/real-stage1.yaml")
    parser.add_argument("--answer-config", default="config/answer-real-stage1.yaml")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise ValueError("Preserve existing real-world runs; choose a new run ID")
    root.mkdir(parents=True)
    configure_logging()
    cfg = load_config(args.retrieval_config)
    answer_cfg = load_answer_config(args.answer_config)
    if load_config(answer_cfg.retrieval_config) != cfg:
        raise ValueError("answer and retrieval profiles must agree")
    atomic_json(
        root / "frozen-inputs.json",
        {
            "retrieval": asdict(cfg),
            "answer": asdict(answer_cfg),
            "inputs": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(cfg.dataset).glob("*")
                if p.is_file()
            },
            "selection": (
                "Observed-set development replay; not fresh held-out validation."
                if args.development
                else "Fixed existing retrieval/14B profile before first real-corpus query; no tuning on supplied questions."
            ),
            "validation_limit": "Twenty questions supplied by user before outputs; two agent-authored controls. Labels source-reviewed by agent before model outputs, not independently human-reviewed.",
        },
    )
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        cached = {}
        ranking_checks = []
        for split in ("heldout",):
            report = evaluate(
                engine,
                Path("data/real-world/stage1") / (split + ".json"),
                root / (split + ".json"),
                "real-world-stage1-fixed-001",
                split,
                strategies=(("bm25", False), ("vector", False), ("hybrid", False), ("hybrid", True))
                if args.development
                else None,
            )
            if args.development:
                report["evaluation_kind"] = "observed-set development replay"
                atomic_json(root / (split + ".json"), report)
            ranking_checks.append(quality_gate(report["summary"]["hybrid+rerank"], True)["passed"])
            print(split, flush=True)
            print(format_target_table(report["summary"]["hybrid+rerank"]), flush=True)
            for row in report["queries"]:
                if row["mode"] == "hybrid" and row["reranked"]:
                    cached[row["diagnostics"]["query"]] = row["diagnostics"]

        class SameEvidence:
            cfg = engine.cfg
            snapshot = engine.snapshot
            reranker = engine.reranker

            def search(self, question, mode, rerank, top_k):
                return deepcopy(cached[question])

        report = evaluate_answers(
            SameEvidence(),
            create_generator(answer_cfg),
            answer_cfg,
            Path("data/real-world/stage1/answer-evaluation-v1.json"),
            root / "answers.json",
        )
        report["retrieval_results_reused_from_ranking_evaluation"] = True
        report["evaluation_kind"] = (
            "observed-set development replay" if args.development else "initial frozen evaluation"
        )
        report["timing_scope"] = (
            "Generation timings are fresh; retrieval timing was measured in the ranking evaluation and reused."
        )
        atomic_json(root / "answers.json", report)
        print(format_answer_checks(report), flush=True)
        return 0 if report["passed"] and all(ranking_checks) else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())

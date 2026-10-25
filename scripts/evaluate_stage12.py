"""Frozen ten-document retrieval comparison and installed-14B answer evaluation."""

import argparse
import hashlib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answer_evaluation import evaluate_answers, format_answer_checks
from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.evaluation import evaluate, format_target_table
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.ollama import OllamaGenerator
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/stage-1.2/run-001")
    root = Path(parser.parse_args().output)
    if root.exists():
        raise ValueError("Preserve existing stage 1.2 runs; choose a new run ID")
    root.mkdir(parents=True)
    configure_logging()
    cfg = load_config("config/stage-1.2.yaml")
    answer_cfg = load_answer_config("config/answer-stage-1.2.yaml")
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
            "selection": "Existing Stage 1.1 retrieval and 14B generation profile fixed before first Stage 1.2 query; no parameter tuning or model selection.",
            "validation_limit": "Questions and source labels authored by the implementing agent; fresh questions are prospective, not independently human-labelled.",
        },
    )
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        cached = {}
        for split in ("tuning", "heldout"):
            report = evaluate(
                engine,
                Path(cfg.dataset) / (split + ".json"),
                root / (split + ".json"),
                "stage12-fixed-001",
                split,
            )
            print(split, flush=True)
            print(format_target_table(report["summary"]["hybrid+rerank"]), flush=True)
            for row in report["queries"]:
                if row["mode"] == "hybrid" and row["reranked"]:
                    cached[row["diagnostics"]["query"]] = row["diagnostics"]

        class SameEvidence:
            cfg = engine.cfg
            snapshot = engine.snapshot

            def search(self, question, mode, rerank, top_k):
                return deepcopy(cached[question])

        report = evaluate_answers(
            SameEvidence(),
            OllamaGenerator(answer_cfg),
            answer_cfg,
            Path(cfg.dataset) / "answer-evaluation-v1.json",
            root / "answers.json",
        )
        report["retrieval_results_reused_from_ranking_evaluation"] = True
        report["timing_scope"] = (
            "Generation timings are fresh; retrieval timing was measured in the ranking evaluation and reused."
        )
        atomic_json(root / "answers.json", report)
        print(format_answer_checks(report), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()

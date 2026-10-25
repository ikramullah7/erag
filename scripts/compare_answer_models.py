"""Fixed-question comparison of already-installed Qwen models; never pulls models."""

import argparse
import hashlib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answer_evaluation import evaluate_answers, format_answer_checks, load_answer_cases
from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.ollama import OllamaGenerator
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/answer-evaluation-v1")
    root = Path(parser.parse_args().output)
    if root.exists():
        raise ValueError("Preserve earlier answer comparisons; use a new version")
    root.mkdir()
    configure_logging()
    cfgs = [load_answer_config(p) for p in ["config/answer.yaml", "config/answer-8b.yaml"]]
    retrieval_cfg = load_config(cfgs[0].retrieval_config)
    cases_path = Path("data/stage-1.1/answer-evaluation-v1.json")
    atomic_json(
        root / "frozen-inputs.json",
        {
            "models": [asdict(c) for c in cfgs],
            "cases_hash": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "retrieval_config": asdict(retrieval_cfg),
            "selection": "Compare fixed models; no parameter tuning",
            "timing_scope": "Generation times only; identical cached retrieval results supplied to each model",
        },
    )
    store = QdrantStore(path=str(Path(retrieval_cfg.state) / "qdrant"))
    reports = []
    try:
        real = Engine(
            retrieval_cfg,
            store,
            SentenceEmbedder(retrieval_cfg),
            CrossEncoderReranker(retrieval_cfg),
        )
        cases = load_answer_cases(cases_path, real)
        cached = {
            c["query"]: real.search(
                c["query"], retrieval_cfg.mode, retrieval_cfg.rerank, retrieval_cfg.candidates
            )
            for c in cases
        }

        class SameEvidence:
            cfg = real.cfg
            snapshot = real.snapshot

            def search(self, question, mode, rerank, top_k):
                return deepcopy(cached[question])

        for cfg in cfgs:
            report = evaluate_answers(
                SameEvidence(),
                OllamaGenerator(cfg),
                cfg,
                cases_path,
                root / (cfg.model.replace(":", "-") + ".json"),
            )
            report["retrieval_results_reused_for_equal_context"] = True
            report["retrieval_executed_once_per_question"] = True
            atomic_json(root / (cfg.model.replace(":", "-") + ".json"), report)
            reports.append(report)
            print(format_answer_checks(report), flush=True)
        atomic_json(
            root / "comparison.json",
            {
                "models": [
                    {k: r[k] for k in ["model", "model_digest", "summary", "checks", "passed"]}
                    for r in reports
                ],
                "same_sources_for_both_models": True,
                "prompt_version": reports[0]["prompt_version"],
            },
        )
        return 0 if reports[0]["passed"] else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())

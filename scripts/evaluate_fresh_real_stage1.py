"""One-shot fresh-question validation against the previously archived five-doc pipeline."""

import json
from copy import deepcopy
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answer_evaluation import evaluate_answers
from erag.config import load_config
from erag.evaluation import evaluate, load_cases
from erag.evidence_evaluation import analyze_evidence, validate_pool, verify_frozen_files
from erag.generators import create_generator
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


def main():
    packet = Path("data/real-world/stage1/fresh-validation-v1")
    frozen = json.loads((packet / "frozen-pipeline.json").read_text())
    verify_frozen_files(frozen["pipeline_hashes"])
    verify_frozen_files(frozen["label_hashes"])
    root = Path("artifacts/real-world/stage1/fresh-validation-v1")
    if root.exists():
        raise ValueError("fresh validation already claimed; never rerun silently")
    cfg = load_config("config/real-stage1-v2.yaml")
    acfg = load_answer_config("config/answer-real-stage1-v2.yaml")
    snapshot = json.loads((Path(cfg.state) / "manifest.json").read_text())
    pool = json.loads((packet / "evidence-pool.json").read_text())
    validate_pool(pool, snapshot)
    root.mkdir(parents=True)
    atomic_json(root / "frozen-inputs.json", frozen)
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        load_cases(packet / "heldout.json", engine)
        ranking = evaluate(
            engine,
            packet / "heldout.json",
            root / "ranking.json",
            "fresh-user-questions-v1",
            "heldout",
            strategies=(("bm25", False), ("vector", False), ("hybrid", False), ("hybrid", True)),
        )
        ranking["evaluation_kind"] = (
            "Fresh question wordings; pre-existing frozen pipeline; provisional source-reviewed labels; topics supplied by user."
        )
        atomic_json(root / "ranking.json", ranking)
        cached = {
            r["diagnostics"]["query"]: r["diagnostics"]
            for r in ranking["queries"]
            if r["mode"] == "hybrid" and r["reranked"]
        }

        class SameEvidence:
            def __init__(self):
                self.cfg, self.snapshot, self.reranker = cfg, engine.snapshot, engine.reranker

            def search(self, question, mode, rerank, top_k):
                return deepcopy(cached[question])

        answers = evaluate_answers(
            SameEvidence(),
            create_generator(acfg),
            acfg,
            packet / "answer-cases.json",
            root / "answers.json",
        )
        answers["evaluation_kind"] = ranking["evaluation_kind"]
        answers["timing_scope"] = (
            "Fresh generation and fact-reranking; ranking diagnostics reused. Not fresh full-pipeline latency."
        )
        atomic_json(root / "answers.json", answers)
        atomic_json(
            root / "evidence-alignment.json",
            analyze_evidence(pool, engine.snapshot, ranking, answers),
        )
        verify_frozen_files(frozen["pipeline_hashes"])
        verify_frozen_files(frozen["label_hashes"])
        print(json.dumps(answers["summary"], indent=2), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()

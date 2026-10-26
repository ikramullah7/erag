"""Re-evaluate observed questions with cached raw ranking and fresh answer evidence."""

import argparse
import gzip
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answer_evaluation import evaluate_answers, format_answer_checks
from erag.config import digest, load_config
from erag.generators import create_generator
from erag.ingestion import ingest
from erag.models import CrossEncoderReranker
from erag.pipeline import atomic_json, corpus_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ranking", "cases", "answer-config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--case-ids", nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve earlier runs; choose a new output directory")
    acfg = load_answer_config(args.answer_config)
    cfg = load_config(acfg.retrieval_config)
    snapshot_path = Path(cfg.state) / "manifest.json"
    snapshot = json.loads(snapshot_path.read_text())
    ranking = json.loads(args.ranking.read_text())
    if (
        ranking["config"] != asdict(cfg)
        or ranking["corpus_hash"] != snapshot["corpus_hash"]
        or ranking["chunk_hash"] != snapshot["chunk_hash"]
        or snapshot["chunk_hash"] != digest(snapshot["chunks"])
        or snapshot["corpus_hash"] != corpus_hash(ingest(cfg.dataset, cfg.stage))
    ):
        raise ValueError("cached ranking, index and current source do not agree")
    by_id = {c["id"]: c for c in snapshot["chunks"]}
    cached = {}
    for row in ranking["queries"]:
        if row["mode"] == "hybrid" and row["reranked"]:
            d = row["diagnostics"]
            if any(r["chunk"] != by_id.get(r["chunk"]["id"]) for r in d["results"]):
                raise ValueError("cached retrieval contains altered chunks")
            cached[d["query"]] = d
    dataset = json.loads(args.cases.read_text())
    if args.case_ids:
        selected = set(args.case_ids)
        dataset["cases"] = [c for c in dataset["cases"] if c["id"] in selected]
        if {c["id"] for c in dataset["cases"]} != selected:
            raise ValueError("unknown case ID")
    if any(c["query"] not in cached for c in dataset["cases"]):
        raise ValueError("case query is missing from cached ranking")
    args.output.mkdir(parents=True)
    atomic_json(args.output / "cases.json", dataset)
    paths = [
        Path(__file__),
        args.ranking,
        args.cases,
        args.answer_config,
        snapshot_path,
        *Path("src/erag").glob("*.py"),
    ]
    atomic_json(
        args.output / "inputs.json",
        {
            "retrieval": asdict(cfg),
            "answer": asdict(acfg),
            "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "evaluation_kind": "Observed development questions; no independent validation or raw ranking changes.",
        },
    )
    with gzip.open(args.output / "source.json.gz", "wt") as archive:
        json.dump({str(p): p.read_text() for p in paths if p.suffix in {".py", ".yaml"}}, archive)
    generator = create_generator(acfg)
    if acfg.provider == "ollama":
        generator.verify_model()

    class Cached:
        def __init__(self):
            self.cfg, self.snapshot = cfg, snapshot
            self.reranker = CrossEncoderReranker(cfg)

        def search(self, question, mode, rerank, top_k):
            if mode not in (None, "hybrid") or not rerank:
                raise ValueError("cache contains hybrid reranked retrieval only")
            return deepcopy(cached[question])

    report = evaluate_answers(
        Cached(),
        generator,
        acfg,
        args.output / "cases.json",
        args.output / "answers.json",
    )
    report["evaluation_kind"] = (
        "Observed-set development replay; raw chunk retrieval cached, document fact search and generation fresh."
    )
    report["timing_scope"] = (
        "Application elapsed time includes fresh document/fact search and generation; initial chunk retrieval timings are reused."
    )
    atomic_json(args.output / "answers.json", report)
    inputs = json.loads((args.output / "inputs.json").read_text())
    if any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in inputs["sha256"].items()
    ):
        raise ValueError("evaluation inputs changed during generation; do not promote this run")
    print(format_answer_checks(report), flush=True)


if __name__ == "__main__":
    main()

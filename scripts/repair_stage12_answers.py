"""Revalidate measured first-pass outputs and make only the two necessary local repair calls."""

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answer_evaluation import evaluate_answers, format_answer_checks
from erag.answering import Generation
from erag.config import load_config
from erag.ollama import OllamaGenerator
from erag.pipeline import atomic_json
from erag.reviewed_metrics import file_hash


def main():
    previous = Path("artifacts/stage-1.2/run-001")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/stage-1.2/repair-001")
    root = Path(parser.parse_args().output)
    if root.exists():
        raise ValueError("Preserve earlier repair evaluations")
    root.mkdir()
    cfg = load_config("config/stage-1.2.yaml")
    answer_cfg = load_answer_config("config/answer-stage-1.2.yaml")
    original = json.loads((previous / "answers.json").read_text())
    outputs = {r["query"]: r for r in original["cases"]}
    cached = {}
    for split in ("tuning", "heldout"):
        for row in json.loads((previous / (split + ".json")).read_text())["queries"]:
            if row["mode"] == "hybrid" and row["reranked"]:
                cached[row["diagnostics"]["query"]] = row["diagnostics"]

    class Evidence:
        snapshot = json.loads((Path(cfg.state) / "manifest.json").read_text())

        def __init__(self):
            self.cfg = cfg

        def search(self, question, mode, rerank, top_k):
            return deepcopy(cached[question])

    class RecordedFirstAttempt:
        def __init__(self):
            self.seen = set()
            self.new_requests = 0
            self.live = OllamaGenerator(answer_cfg)

        def generate(self, messages, schema):
            question = json.loads(messages[1]["content"])["question"]
            if question in self.seen:
                self.new_requests += 1
                return self.live.generate(messages, schema)
            self.seen.add(question)
            row = outputs[question]
            if row["result"]["status"] == "invalid_response":
                return Generation(**row["rejected_generation"])
            result = row["result"]
            # Reconstruct the already-validated structured payload, retaining actual identity,
            # usage and measured generation duration; do not regenerate passed answers.
            payload = {
                "status": result["status"],
                "claims": [
                    {
                        "text": c["text"],
                        "citations": [
                            {k: e[k] for k in ("source_id", "quote")} for e in c["citations"]
                        ],
                    }
                    for c in result["claims"]
                ],
            }
            return Generation(
                payload, result["model_identity"], result["usage"], result["generation_ms"]
            )

    generator = RecordedFirstAttempt()
    report = evaluate_answers(
        Evidence(),
        generator,
        answer_cfg,
        Path(cfg.dataset) / "answer-evaluation-v1.json",
        root / "answers.json",
    )
    report.update(
        first_attempts_replayed_from=str(previous / "answers.json"),
        first_attempt_input_sha256=file_hash(previous / "answers.json"),
        new_local_repair_requests=generator.new_requests,
        first_pass_method="Raw rejected generations replayed exactly; passed payloads reconstructed from their already-validated claims. No expected answers supplied to the model.",
        timing_scope="Fresh repair durations plus original measured generation durations; retrieval measured in run-001. Replayed elapsed_ms is not fresh end-to-end latency.",
        config=asdict(answer_cfg),
    )
    atomic_json(root / "answers.json", report)
    print(format_answer_checks(report))
    print("New local repair requests:", generator.new_requests)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

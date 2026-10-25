"""Reproducible five-document answer checks, distinct from retrieval ranking metrics."""

import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .answering import PROMPT_VERSION, answer_question
from .evaluation import environment
from .pipeline import atomic_json


def load_answer_cases(path: str | Path, engine) -> list[dict]:
    dataset = json.loads(Path(path).read_text())
    cases = dataset["cases"]
    documents = {d["id"]: d["text"] for d in engine.snapshot["documents"]}
    seen = set()
    for case in cases:
        if case["id"] in seen or not case["query"].strip():
            raise ValueError("duplicate answer case or empty question")
        seen.add(case["id"])
        if case["expected_status"] not in {"answered", "insufficient_evidence"}:
            raise ValueError("unknown expected answer status")
        if (case["expected_status"] == "answered") != bool(case["fact_patterns"]):
            raise ValueError("answerable cases require expected fact patterns")
        for alternatives in case["fact_patterns"]:
            if not alternatives:
                raise ValueError("empty expected fact alternatives")
            for pattern in alternatives:
                re.compile(pattern, re.IGNORECASE)
        if case["expected_status"] == "answered" and not case["source_evidence"]:
            raise ValueError("answerable cases require source evidence")
        for evidence in case["source_evidence"]:
            if (
                evidence["document_id"] not in documents
                or evidence["text"] not in documents[evidence["document_id"]]
            ):
                raise ValueError(f"unsupported expected evidence: {case['id']}")
    return cases


def evaluate_answers(
    engine,
    generator,
    cfg,
    cases_path: str | Path,
    output: str | Path,
    mode: str | None = None,
    rerank: bool | None = None,
) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError("answer evaluation output already exists; preserve earlier results")
    cases = load_answer_cases(cases_path, engine)
    documents = {d["id"]: d["text"] for d in engine.snapshot["documents"]}
    rows = []

    class Capture:
        last = None
        history = None

        def generate(self, messages, schema):
            self.last = generator.generate(messages, schema)
            self.history.append(asdict(self.last))
            return self.last

    captured = Capture()
    for case in cases:
        captured.last = None
        captured.history = []
        try:
            result = answer_question(engine, captured, cfg, case["query"], mode, rerank)
        except (ValueError, RuntimeError, OSError) as exc:
            result = {"status": "provider_error", "answer": "", "claims": [], "error": str(exc)}
        fact_checks = [
            any(re.search(pattern, result["answer"], re.IGNORECASE) for pattern in alternatives)
            for alternatives in case["fact_patterns"]
        ]
        citations_valid = result["status"] == "answered" and bool(result["claims"])
        for claim in result["claims"]:
            for evidence in claim["citations"]:
                doc = documents.get(evidence["document_id"], "")
                citations_valid &= doc[evidence["start"] : evidence["end"]] == evidence["quote"]
        passed = (
            result["status"] == "answered" and all(fact_checks) and citations_valid
            if case["expected_status"] == "answered"
            else result["status"] == "insufficient_evidence"
        )
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected_status": case["expected_status"],
                "fact_checks": fact_checks,
                "citations_valid": bool(citations_valid),
                "passed": bool(passed),
                "result": result,
            }
        )
        if len(captured.history) > 1:
            rows[-1]["generation_attempts"] = captured.history
        if result["status"] == "invalid_response" and captured.last is not None:
            # Evaluation diagnostics only; never rendered as the user's factual answer.
            rows[-1]["rejected_generation"] = asdict(captured.last)
        # Preserve partial evidence if a later local inference fails or is interrupted.
        atomic_json(
            output.with_suffix(".partial.json"),
            {"model": cfg.model, "completed": len(rows), "cases": rows},
        )
    known = [r for r in rows if r["expected_status"] == "answered"]
    unknown = [r for r in rows if r["expected_status"] == "insufficient_evidence"]
    answered = [r for r in rows if r["result"]["status"] == "answered"]
    if not known or not unknown:
        raise ValueError("answer checks require both answerable and unsupported questions")
    generation_times = [
        r["result"]["generation_ms"] for r in rows if "generation_ms" in r["result"]
    ]
    summary = {
        "cases": len(rows),
        "known_fact_cases_passed": sum(r["passed"] for r in known),
        "known_fact_cases": len(known),
        "known_fact_case_rate": sum(r["passed"] for r in known) / len(known),
        "correct_abstentions": sum(r["passed"] for r in unknown),
        "unsupported_cases": len(unknown),
        "abstention_rate": sum(r["passed"] for r in unknown) / len(unknown),
        "valid_responses": sum(
            r["result"]["status"] in {"answered", "insufficient_evidence"} for r in rows
        ),
        "invalid_responses": sum(r["result"]["status"] == "invalid_response" for r in rows),
        "provider_errors": sum(r["result"]["status"] == "provider_error" for r in rows),
        "unsupported_answers": sum(r["result"]["status"] == "answered" for r in unknown),
        "citation_provenance_rate": sum(r["citations_valid"] for r in answered) / len(answered)
        if answered
        else 0,
        "generation_ms_p50": float(np.percentile(generation_times, 50))
        if generation_times
        else None,
        "generation_ms_p95": float(np.percentile(generation_times, 95))
        if generation_times
        else None,
        "failed_cases": [r["id"] for r in rows if not r["passed"]],
    }
    checks = {
        "known_fact_cases_at_least_90_percent": summary["known_fact_case_rate"] >= 0.90,
        "all_unsupported_questions_abstained": summary["abstention_rate"] == 1,
        "no_invalid_or_error_responses": summary["invalid_responses"] == 0
        and summary["provider_errors"] == 0,
        "all_answered_citations_have_valid_provenance": summary["citation_provenance_rate"] == 1,
    }
    report = {
        "phase": "answer-" + engine.cfg.stage,
        "documents": len(engine.snapshot["documents"]),
        "model": cfg.model,
        "model_digest": cfg.model_digest,
        "config": asdict(cfg),
        "retrieval_config": asdict(engine.cfg),
        "prompt_version": PROMPT_VERSION,
        "dataset_hash": hashlib.sha256(Path(cases_path).read_bytes()).hexdigest(),
        "corpus_hash": engine.snapshot["corpus_hash"],
        "chunk_hash": engine.snapshot["chunk_hash"],
        "environment": environment(),
        "created_unix": time.time(),
        "summary": summary,
        "checks": checks,
        "passed": all(checks.values()),
        "cases": rows,
        "limitation": "Expected-fact regexes and quote provenance do not prove full semantic entailment; inspect claims against evidence.",
    }
    atomic_json(output, report)
    output.with_suffix(".partial.json").unlink(missing_ok=True)
    return report


def format_answer_checks(report: dict) -> str:
    s = report["summary"]
    median = s["generation_ms_p50"]
    generation_time = f"{median / 1000:.2f}s" if median is not None else "unavailable"
    return (
        f"{report['model']} | {report['phase']} | {report['documents']} documents\n"
        f"Known-fact cases: {s['known_fact_cases_passed']}/{s['known_fact_cases']} (target >=90%)\n"
        f"Correct abstentions: {s['correct_abstentions']}/{s['unsupported_cases']} (target 100%)\n"
        f"Citation provenance: {s['citation_provenance_rate']:.0%} (target 100%)\n"
        f"Invalid responses: {s['invalid_responses']}; provider errors: {s['provider_errors']}\n"
        f"Generation median: {generation_time}\n"
        f"Gate: {'PASS' if report['passed'] else 'FAIL'}"
    )

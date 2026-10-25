"""Answer checks distinguish fact matching, provenance and abstention."""

import json
from dataclasses import replace
from typing import ClassVar

import pytest

from erag.answer_config import AnswerConfig
from erag.answer_evaluation import evaluate_answers, format_answer_checks, load_answer_cases
from erag.answering import Generation
from erag.config import load_config

QUOTE = "The ledger address is 10.42.8.17."


class Engine:
    cfg = replace(load_config("config/default.yaml"), rerank=False)
    snapshot: ClassVar[dict] = {
        "documents": [{"id": "azure-networking", "text": QUOTE}],
        "corpus_hash": "test-corpus",
        "chunk_hash": "test-chunks",
    }

    def search(self, *args, **kwargs):
        return {
            "mode": "hybrid",
            "reranked": False,
            "timings": {},
            "results": [
                {
                    "final_rank": 1,
                    "chunk": {
                        "id": "one",
                        "document_id": "azure-networking",
                        "source": "azure-networking.txt",
                        "version": "one",
                        "heading": "Ledger",
                        "text": QUOTE,
                        "start": 0,
                        "end": len(QUOTE),
                    },
                }
            ],
        }


class Generator:
    def __init__(self, abstain=True):
        self.abstain = abstain
        self.calls = 0

    def generate(self, messages, schema):
        self.calls += 1
        payload = {
            "status": "answered",
            "claims": [{"text": QUOTE, "citations": [{"source_id": "S1", "quote": QUOTE}]}],
        }
        if self.calls > 1 and self.abstain:
            payload = {"status": "insufficient_evidence", "claims": []}
        return Generation(payload, "test", {}, 1)


def cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "known",
                        "query": "Ledger IP?",
                        "expected_status": "answered",
                        "fact_patterns": [[r"10\.42\.8\.17"]],
                        "source_evidence": [{"document_id": "azure-networking", "text": QUOTE}],
                    },
                    {
                        "id": "unknown",
                        "query": "Database password?",
                        "expected_status": "insufficient_evidence",
                        "fact_patterns": [],
                        "source_evidence": [],
                    },
                ]
            }
        )
    )
    return path


def test_fact_and_abstention_checks_preserve_evidence(tmp_path):
    path = tmp_path / "result.json"
    report = evaluate_answers(
        Engine(), Generator(), AnswerConfig("qwen3:14b", "a" * 64), cases(tmp_path), path
    )
    assert report["passed"] and report["summary"]["known_fact_cases_passed"] == 1
    assert report["summary"]["correct_abstentions"] == 1
    assert report["cases"][0]["result"]["claims"][0]["citations"][0]["start"] == 0
    assert path.exists() and not path.with_suffix(".partial.json").exists()
    with pytest.raises(ValueError, match="preserve"):
        evaluate_answers(
            Engine(), Generator(), AnswerConfig("qwen3:14b", "a" * 64), cases(tmp_path), path
        )


def test_valid_citation_does_not_pass_an_unsupported_answer(tmp_path):
    report = evaluate_answers(
        Engine(),
        Generator(abstain=False),
        AnswerConfig("qwen3:14b", "a" * 64),
        cases(tmp_path),
        tmp_path / "bad.json",
    )
    assert report["summary"]["citation_provenance_rate"] == 1
    assert not report["passed"] and report["summary"]["unsupported_answers"] == 1


def test_evaluation_expected_facts_must_exist_in_sources(tmp_path):
    path = cases(tmp_path)
    data = json.loads(path.read_text())
    data["cases"][0]["source_evidence"][0]["text"] = "invented evidence"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unsupported expected evidence"):
        load_answer_cases(path, Engine())


def test_provider_failures_remain_reportable_without_latency(tmp_path):
    class FailedProvider:
        def generate(self, *args):
            raise RuntimeError("local runtime unavailable")

    report = evaluate_answers(
        Engine(),
        FailedProvider(),
        AnswerConfig("qwen3:14b", "a" * 64),
        cases(tmp_path),
        tmp_path / "errors.json",
    )
    assert not report["passed"] and report["summary"]["provider_errors"] == 2
    assert "Generation median: unavailable" in format_answer_checks(report)


def test_rejected_payload_is_diagnostic_and_never_a_factual_answer(tmp_path):
    class BadProvider(Generator):
        def generate(self, *args):
            generated = super().generate(*args)
            if self.calls == 1:
                generated.payload["claims"][0]["text"] = "The ledger address is 10.42.8.99."
            return generated

    report = evaluate_answers(
        Engine(),
        BadProvider(),
        AnswerConfig("qwen3:14b", "a" * 64),
        cases(tmp_path),
        tmp_path / "rejected.json",
    )
    row = report["cases"][0]
    assert row["result"]["status"] == "invalid_response"
    assert "10.42.8.99" not in row["result"]["answer"]
    assert "10.42.8.99" in row["rejected_generation"]["payload"]["claims"][0]["text"]

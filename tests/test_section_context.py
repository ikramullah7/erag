import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from erag.answer_config import AnswerConfig
from erag.answering import Generation, answer_question
from erag.section_context import complete_sections, section_ranges


def fixture(text, quote):
    a = text.index(quote)
    chunk = {
        "id": "c",
        "document_id": "d",
        "version": "v",
        "source": "manual",
        "heading": "Rules",
        "position": 0,
        "start": a,
        "end": a + len(quote),
        "text": quote,
    }
    cite = {
        "source_id": "S1",
        "document_id": "d",
        "version": "v",
        "start": a,
        "end": a + len(quote),
        "quote": quote,
    }
    return {
        "status": "answered",
        "claims": [{"text": quote, "citations": [cite]}],
        "sources": [{"source_id": "S1", "chunk": chunk}],
    }, {"documents": [{"id": "d", "version": "v", "text": text}], "chunks": [chunk]}


def test_default_and_alternative_are_retained_in_one_exact_parent_section():
    text = "# Manual\n\n## Management\n\nThe policy controls ordering.\n\n### Default\n\nSequential is the default.\n\n### Alternative\n\nConcurrent mode permits simultaneous termination.\n\n## Unrelated\n\nAnother rule applies here."
    result, snapshot = fixture(text, "The policy controls ordering.")
    before = copy.deepcopy(result)
    out = complete_sections(result, snapshot)
    assert result == before
    assert (
        "Sequential is the default" in out["answer"] and "Concurrent mode permits" in out["answer"]
    )
    assert out["claims"][0]["text"] in text and "Unrelated" not in out["answer"]
    cite = out["claims"][0]["citations"][0]
    assert text[cite["start"] : cite["end"]] == cite["quote"]
    assert complete_sections(out, snapshot) == out


def test_large_sections_end_on_a_closed_subsection_without_truncation():
    text = (
        "# Manual\n\n## Policy\n\nBase rule applies.\n\n### First\n\nA complete first alternative.\n\n### Second\n\n"
        + "Long unrelated content. " * 100
    )
    ranges = section_ranges(text, 130)
    policy = next(r for r in ranges if r["heading"].endswith("Policy"))
    quote = text[policy["start"] : policy["end"]]
    assert quote.endswith("A complete first alternative.") and len(quote) <= 130


def test_completion_never_exposes_fenced_examples_or_changes_failure_status():
    text = "# Manual\n\n## Rules\n\nA standalone source rule applies.\n\n```\nsecret: example\n```"
    result, snapshot = fixture(text, "A standalone source rule applies.")
    assert complete_sections(result, snapshot)["answer"] == result["claims"][0]["text"] + " [S1]"
    failed = {"status": "provider_error", "claims": [], "error": "timeout"}
    assert complete_sections(failed, snapshot) == failed
    result["claims"][0]["citations"][0]["version"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        complete_sections(result, snapshot)


@pytest.mark.skipif(
    not Path("artifacts/real-world/stage1/index-v2/manifest.json").exists(),
    reason="requires local real-document snapshot",
)
def test_real_short_policy_sections_preserve_parallel_and_default_context():
    snapshot = json.loads(Path("artifacts/real-world/stage1/index-v2/manifest.json").read_text())
    doc = next(d for d in snapshot["documents"] if d["id"].endswith("statefulset"))
    ranges = section_ranges(doc["text"])
    policy = next(r for r in ranges if r["heading"].endswith("Pod Management Policies"))
    quote = doc["text"][policy["start"] : policy["end"]]
    assert "OrderedReady pod management is the default" in quote
    assert "Parallel pod management tells" in quote


def test_clipped_paragraph_is_completed_without_including_fenced_examples():
    text = (
        "# Manual\n\nThe entire first sentence describes desired state. A second sentence defines actual state.\n\n```\n"
        + "example " * 400
        + "\n```"
    )
    result, snapshot = fixture(text, "desired state. A second sentence defines actual state.")
    out = complete_sections(result, snapshot)
    assert out["claims"][0]["text"].startswith("The entire first sentence")
    assert "```" not in out["answer"]
    assert out["model_source_ids"] == ["S1"]
    with pytest.raises(ValueError, match="budget"):
        complete_sections(result, snapshot, 2001)


def test_completion_flag_preserves_actual_generation_inputs_and_adds_policy_context():
    text = "# Manual\n\n## Management\n\nThe policy controls ordering.\n\n### Default\n\nSequential is the default.\n\n### Alternative\n\nConcurrent mode permits simultaneous termination."
    original, snapshot = fixture(text, "The policy controls ordering.")
    engine = SimpleNamespace(
        cfg=SimpleNamespace(rerank=False, candidates=20),
        snapshot=snapshot,
        search=lambda *a, **k: {
            "results": [{"chunk": original["sources"][0]["chunk"], "final_rank": 1}],
            "mode": "hybrid",
            "reranked": False,
            "timings": {},
        },
    )

    class Generator:
        def __init__(self):
            self.calls = []

        def generate(self, messages, schema):
            self.calls.append(copy.deepcopy((messages, schema)))
            return Generation({"evidence_ids": ["E1"]}, "fixture", {}, 1)

    cfg = AnswerConfig(model="local", model_digest="a" * 64, evidence_selection=True)
    before, after = Generator(), Generator()
    raw = answer_question(engine, before, cfg, "How is ordering controlled?")
    completed = answer_question(
        engine, after, replace(cfg, section_context_completion=True), "How is ordering controlled?"
    )
    assert before.calls == after.calls and len(after.calls) == 1
    assert completed["model_claims"] == raw["claims"]
    assert completed["catalog"] == raw["catalog"]
    assert (
        "Concurrent mode" in completed["answer"]
        and "Sequential is the default" in completed["answer"]
    )
    with pytest.raises(ValueError, match="boolean"):
        replace(cfg, section_context_completion=1)
    with pytest.raises(ValueError, match="requires"):
        replace(cfg, evidence_selection=False, section_context_completion=True)

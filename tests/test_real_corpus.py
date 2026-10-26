"""The approved five-page pilot requires its frozen acquisition manifest."""

import hashlib
import json

import pytest

from erag.ingestion import REAL_STAGE1_IDS, ingest


def test_verified_real_corpus_has_its_own_dataset_and_original_urls(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = []
    for key in sorted(REAL_STAGE1_IDS):
        text = f"Unique source text for {key}.\n"
        (corpus / (key + ".txt")).write_text(text)
        rows.append(
            {
                "id": key,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "http_status": 200,
                "final_url": "https://kubernetes.io/" + key + "/",
            }
        )
    manifest = tmp_path / "acquisition.json"
    manifest.write_text(json.dumps({"status": "corpus_frozen", "documents": rows}))
    docs = ingest(corpus, "real-1")
    assert len(docs) == 5 and all(d.dataset == "real-world-stage1-v1" for d in docs)
    assert all(d.source.startswith("https://kubernetes.io/") for d in docs)
    file = corpus / (min(REAL_STAGE1_IDS) + ".txt")
    file.write_text("Changed source")
    with pytest.raises(ValueError, match="frozen"):
        ingest(corpus, "real-1")


def test_real_stage_rejects_unverified_or_synthetic_sources(tmp_path):
    with pytest.raises(ValueError):
        ingest("data/stage-1.1", "real-1")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for key in REAL_STAGE1_IDS:
        (corpus / (key + ".txt")).write_text(key)
    with pytest.raises(FileNotFoundError):
        ingest(corpus, "real-1")

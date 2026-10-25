from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from erag.chunking import chunk_document, validate_chunks
from erag.ingestion import ingest


def test_exactly_five_and_deterministic(config):
    docs = ingest(config.dataset)
    assert len(docs) == 5
    assert docs == ingest(config.dataset)
    assert len({d.version for d in docs}) == 5


@pytest.mark.parametrize(
    "problem", ["empty", "binary", "invalid_utf8", "duplicate", "extra", "missing", "symlink"]
)
def test_reject_invalid_ingestion(config, problem):
    root = Path(config.dataset)
    target = root / "aks-platform.txt"
    if problem == "empty":
        target.write_text(" ")
    elif problem == "binary":
        target.write_bytes(b"abc\x00def")
    elif problem == "invalid_utf8":
        target.write_bytes(b"\xff")
    elif problem == "duplicate":
        target.write_bytes((root / "observability.txt").read_bytes())
    elif problem == "extra":
        (root / "sixth.txt").write_text("another document")
    elif problem == "missing":
        target.unlink()
    else:
        target.unlink()
        target.symlink_to(root / "observability.txt")
    with pytest.raises((ValueError, UnicodeError)):
        ingest(root)


@pytest.mark.parametrize("size,overlap", [(100, 0), (100, 20), (140, 20), (180, 20), (140, 40)])
def test_chunk_integrity_offsets_and_stable_ids(config, size, overlap):
    for doc in ingest(config.dataset):
        chunks = chunk_document(doc, size, overlap)
        assert chunks == chunk_document(doc, size, overlap)
        validate_chunks(doc, chunks, size)
        assert all(c.text == doc.text[c.start : c.end] for c in chunks)
        assert all(len(c.text.split()) <= size for c in chunks)
        assert all(c.heading for c in chunks)
        if overlap:
            assert all(b.start < a.end for a, b in pairwise(chunks))
        updated = replace(doc, version="new-version")
        assert {c.id for c in chunks}.isdisjoint(
            c.id for c in chunk_document(updated, size, overlap)
        )


def test_chunk_metadata_corruption_is_rejected(config):
    doc = ingest(config.dataset)[0]
    chunks = chunk_document(doc, 140, 20)
    chunks[0] = replace(chunks[0], position=99)
    with pytest.raises(ValueError):
        validate_chunks(doc, chunks, 140)


def test_missing_content_is_rejected(config):
    doc = ingest(config.dataset)[0]
    chunks = chunk_document(doc, 140, 20)
    with pytest.raises(ValueError, match="missing"):
        validate_chunks(doc, chunks[:-1], 140)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_settings(config, size, overlap):
    with pytest.raises(ValueError):
        chunk_document(ingest(config.dataset)[0], size, overlap)


def test_forged_chunk_id_rejected(config):
    doc = ingest(config.dataset)[0]
    chunks = chunk_document(doc, 140, 20)
    chunks[0] = replace(chunks[0], id="forged")
    with pytest.raises(ValueError):
        validate_chunks(doc, chunks, 140)


def test_oversized_unicode_paragraph_splits_without_content_loss(config):
    from erag.types import Document

    text = "# Résumé\n\n" + " ".join(f"mémoire-{i}" for i in range(500)) + "\n"
    doc = Document("unit-only", "in-memory", "unit-version", text)
    chunks = chunk_document(doc, 100, 20)
    assert len(chunks) > 1
    validate_chunks(doc, chunks, 100)
    assert all(len(c.text.split()) <= 100 for c in chunks)
    assert all(c.heading == "Résumé" for c in chunks)
    assert all(b.start < a.end for a, b in pairwise(chunks))


def test_section_chunks_do_not_carry_neighbouring_section_fragments(config):
    import re

    for doc in ingest(config.dataset):
        chunks = chunk_document(doc, 140, 20, section_boundaries=True)
        assert len(chunks) == 6
        validate_chunks(doc, chunks, 140)
        boundaries = [m.start() for m in re.finditer(r"(?m)^## ", doc.text)][1:]
        assert all(not any(c.start < b < c.end for b in boundaries) for c in chunks)
        assert all(a.end <= b.start for a, b in pairwise(chunks))

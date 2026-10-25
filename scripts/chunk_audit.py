"""Integrity-only overlap/structure audit; never tunes against held-out labels."""

import json
import re
from dataclasses import asdict
from pathlib import Path

from erag.chunking import chunk_document
from erag.ingestion import ingest
from erag.pipeline import atomic_json


def main() -> None:
    documents = ingest("data/stage-1.1")
    rows = []
    for size, overlap in ((100, 0), (100, 20), (140, 0), (140, 20), (140, 40), (180, 20)):
        details = []
        for doc in documents:
            chunks = chunk_document(doc, size, overlap)
            words = list(re.finditer(r"\S+", doc.text))
            duplications = [
                sum(c.start <= w.start() and c.end >= w.end() for c in chunks) for w in words
            ]
            paragraphs = list(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", doc.text, re.DOTALL))
            preserved = sum(
                any(c.start <= p.start() and c.end >= p.end() for c in chunks) for p in paragraphs
            )
            headings = list(re.finditer(r"(?m)^#{1,6} .+$", doc.text))
            headings_preserved = sum(
                any(c.start <= h.start() and c.end >= h.end() for c in chunks) for h in headings
            )
            details.append(
                {
                    "document_id": doc.id,
                    "chunks": len(chunks),
                    "words": len(words),
                    "missing_words": sum(x == 0 for x in duplications),
                    "overlap_word_occurrences": sum(max(0, x - 1) for x in duplications),
                    "paragraphs": len(paragraphs),
                    "paragraphs_preserved": preserved,
                    "headings": len(headings),
                    "headings_preserved": headings_preserved,
                    "source_offsets_valid": all(
                        c.text == doc.text[c.start : c.end] for c in chunks
                    ),
                    "duplicate_exact_chunks": len(chunks) - len({(c.start, c.end) for c in chunks}),
                    "chunk_metadata": [asdict(c) for c in chunks],
                }
            )
        rows.append({"max_words": size, "overlap_words": overlap, "documents": details})
    atomic_json(Path("artifacts/chunk-audit.json"), rows)
    print(
        json.dumps(
            [
                {
                    **{k: r[k] for k in ("max_words", "overlap_words")},
                    "chunks": sum(d["chunks"] for d in r["documents"]),
                    "missing_words": sum(d["missing_words"] for d in r["documents"]),
                    "overlap_word_occurrences": sum(
                        d["overlap_word_occurrences"] for d in r["documents"]
                    ),
                    "preserved_paragraphs": sum(d["paragraphs_preserved"] for d in r["documents"]),
                    "total_paragraphs": sum(d["paragraphs"] for d in r["documents"]),
                }
                for r in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

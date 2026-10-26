"""Freeze user-selected real-document inputs before any retrieval/model evaluation."""

import hashlib
import json
from pathlib import Path

from .pipeline import atomic_json


def prepare_real_validation(documents, questions, output):
    """Prepare an input manifest only; this does not bypass synthetic-stage ingestion guards."""
    directory, output = Path(documents), Path(output)
    if output.exists():
        raise ValueError("preserve earlier input manifests; choose a new output")
    files = sorted(directory.glob("*.txt"))
    if not 1 <= len(files) <= 20:
        raise ValueError("select between one and twenty plain-text documents for this pilot")
    sources, seen = [], set()
    for path in files:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError("document must be a regular UTF-8 text file below the size limit")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if not text.strip() or "\x00" in text:
            raise ValueError("empty or binary document")
        normalized = hashlib.sha256(text.strip().encode()).hexdigest()
        if normalized in seen:
            raise ValueError("duplicate document content")
        seen.add(normalized)
        sources.append(
            {
                "id": path.stem,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    if not isinstance(questions, dict) or not isinstance(questions.get("questions"), list):
        raise ValueError("questions must contain a questions list and an author")  # noqa: TRY004
    author = questions.get("author")
    if not isinstance(author, str) or not author.strip() or not questions["questions"]:
        raise ValueError("an author and at least one independently supplied question are required")
    rows, ids, queries = [], set(), set()
    for item in questions["questions"]:
        if not isinstance(item, dict):
            raise ValueError("question entries must be objects")  # noqa: TRY004
        key, query = item.get("id"), item.get("query")
        if (
            not isinstance(key, str)
            or not key.strip()
            or key in ids
            or not isinstance(query, str)
            or not query.strip()
            or len(query) > 4000
            or query.strip().casefold() in queries
        ):
            raise ValueError("questions require unique nonempty IDs and distinct valid queries")
        ids.add(key)
        queries.add(query.strip().casefold())
        rows.append({"id": key, "query": query})
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    manifest = {
        "track": "real-document-validation",
        "status": "prepared_not_indexed",
        "sources": sources,
        "questions": rows,
        "question_author": author,
        "question_sha256": digest,
        "independence": "Author supplied; not independently verified.",
        "evaluation_status": "Not run; expected facts/source labels still require source review.",
        "model_profile": "config/answer-stage-1.3.yaml",
        "stage_policy": "Separate pilot, capped at twenty selected sources; synthetic stages unchanged.",
    }
    atomic_json(output, manifest)
    return manifest

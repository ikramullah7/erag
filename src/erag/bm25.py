"""Lexical retrieval, preserving technical IDs as well as identifier components."""

import re

from rank_bm25 import BM25Okapi

from .types import Chunk


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[._:/=-][a-z0-9]+)*", text.lower())
    return [
        part
        for token in tokens
        for part in (
            [token] + re.findall(r"[a-z0-9]+", token) if re.search(r"[._:/=-]", token) else [token]
        )
    ]


class BM25:
    def __init__(self, chunks: list[Chunk], texts: list[str] | None = None):
        if not chunks:
            raise ValueError("cannot index empty chunks")
        self.chunks = chunks
        texts = [c.text for c in chunks] if texts is None else texts
        if len(texts) != len(chunks):
            raise ValueError("retrieval texts must match chunks")
        self.index = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        scores = self.index.get_scores(tokenize(query))
        rows = [(c.id, float(s)) for c, s in zip(self.chunks, scores, strict=True) if s > 0]
        return sorted(rows, key=lambda row: (-row[1], row[0]))[:limit]

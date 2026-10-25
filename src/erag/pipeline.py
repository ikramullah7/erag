"""Canonical snapshots, atomic publication, and retrieval orchestration."""

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .bm25 import BM25
from .chunking import chunk_document, validate_chunks
from .config import Config, digest
from .diagnostics import event
from .fusion import rrf
from .ingestion import approved_ids, ingest
from .types import Chunk, Embedder, Reranker, VectorStore


def corpus_hash(documents) -> str:
    return digest([(d.id, d.version) for d in documents])


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp.open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build_index(
    cfg: Config, embedder: Embedder, store: VectorStore, deleted_ids: list[str] | None = None
) -> dict:
    started = time.perf_counter()
    documents = ingest(cfg.dataset, cfg.stage)
    manifest = Path(cfg.state) / "manifest.json"
    if deleted_ids is None:
        deleted_ids = (
            json.loads(manifest.read_text()).get("deleted_documents", [])
            if manifest.exists()
            else []
        )
    if not set(deleted_ids) <= approved_ids(cfg.stage):
        raise ValueError("unknown document deletion ID")
    active = [d for d in documents if d.id not in deleted_ids]
    if not active:
        raise ValueError("cannot delete every document")
    chunks = [
        c
        for d in active
        for c in chunk_document(d, cfg.max_words, cfg.overlap_words, cfg.section_boundaries)
    ]
    vectors = embedder.encode([c.text for c in chunks])
    collection = f"erag_{uuid4().hex}"
    store.build(collection, chunks, vectors)
    snapshot = {
        "schema": 1,
        "deleted_documents": sorted(set(deleted_ids)),
        "collection": collection,
        "config_hash": cfg.fingerprint,
        "corpus_hash": corpus_hash(documents),
        "embedding_identity": embedder.identity,
        "chunks": [asdict(c) for c in chunks],
        "documents": [asdict(d) for d in documents],
        "chunk_hash": digest([asdict(c) for c in chunks]),
        "index_ms": (time.perf_counter() - started) * 1000,
    }
    # Publication is the only mutation visible to new readers. Old collections remain valid.
    atomic_json(Path(cfg.state) / "manifest.json", snapshot)
    event(
        "index_published",
        collection=collection,
        documents=len(documents),
        chunks=len(chunks),
        index_ms=snapshot["index_ms"],
    )
    return snapshot


class Engine:
    def __init__(
        self,
        cfg: Config,
        store: VectorStore,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.cfg, self.store, self.embedder, self.reranker = cfg, store, embedder, reranker
        self.snapshot = json.loads((Path(cfg.state) / "manifest.json").read_text())
        if self.snapshot["schema"] != 1 or self.snapshot["config_hash"] != cfg.fingerprint:
            raise ValueError("stale or unsupported index configuration; rebuild")
        documents = ingest(cfg.dataset, cfg.stage)
        if self.snapshot["corpus_hash"] != corpus_hash(documents):
            raise ValueError("source documents changed; rebuild before retrieval")
        self.chunks = [Chunk(**c) for c in self.snapshot["chunks"]]
        if digest(self.snapshot["chunks"]) != self.snapshot["chunk_hash"]:
            raise ValueError("corrupt chunk snapshot")
        if self.snapshot["documents"] != [asdict(d) for d in documents]:
            raise ValueError("document snapshot mismatch")
        deleted = self.snapshot.get("deleted_documents", [])
        if not set(deleted) <= approved_ids(cfg.stage) or not self.chunks:
            raise ValueError("invalid deletion manifest")
        if len({c.id for c in self.chunks}) != len(self.chunks):
            raise ValueError("duplicate chunk IDs")
        if {c.document_id for c in self.chunks} != approved_ids(cfg.stage) - set(deleted):
            raise ValueError("active document collection mismatch")
        for doc in documents:
            if doc.id in deleted:
                continue
            validate_chunks(doc, [c for c in self.chunks if c.document_id == doc.id], cfg.max_words)
        store.validate(self.snapshot["collection"], self.chunks)
        if embedder is not None and embedder.identity != self.snapshot["embedding_identity"]:
            raise ValueError("embedding identity mismatch")
        self.by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25(self.chunks)

    def search(
        self, query: str, mode: str | None = None, rerank: bool = False, top_k: int = 10
    ) -> dict:
        mode = mode or self.cfg.mode
        if not query.strip() or len(query) > 4000 or top_k < 1:
            raise ValueError("query must be nonempty, <=4000 characters, top_k positive")
        if mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError("unknown retrieval mode")
        started = time.perf_counter()
        timings = {}
        lexical, vector = [], []
        if mode in {"bm25", "hybrid"}:
            t = time.perf_counter()
            lexical = self.bm25.search(query, self.cfg.candidates)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000
        if mode in {"vector", "hybrid"}:
            if self.embedder is None:
                raise ValueError("vector retrieval requires embedding provider")
            t = time.perf_counter()
            encoded = self.embedder.encode([query])[0].tolist()
            vector = self.store.search(self.snapshot["collection"], encoded, self.cfg.candidates)
            timings["vector_ms"] = (time.perf_counter() - t) * 1000
        fused = (
            rrf([lexical, vector], self.cfg.rrf_k, [self.cfg.bm25_weight, self.cfg.vector_weight])
            if mode == "hybrid"
            else []
        )
        initial = fused if mode == "hybrid" else lexical if mode == "bm25" else vector
        initial = initial[: self.cfg.candidates]
        final = initial
        reranked = {}
        if rerank:
            if self.reranker is None:
                raise ValueError("reranking requires reranker provider")
            t = time.perf_counter()
            scores = self.reranker.score(query, [self.by_id[cid].text for cid, _ in initial])
            if len(scores) != len(initial) or not all(math.isfinite(s) for s in scores):
                raise ValueError("invalid reranker scores")
            reranked = {cid: float(s) for (cid, _), s in zip(initial, scores, strict=True)}
            final = sorted(reranked.items(), key=lambda row: (-row[1], row[0]))
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000
        components = {"bm25": lexical, "cosine": vector, "rrf": fused}
        scores = {name: dict(rows) for name, rows in components.items()}
        initial_ranks = {cid: rank for rank, (cid, _) in enumerate(initial, 1)}
        rows = []
        for rank, (cid, _) in enumerate(final[:top_k], 1):
            rows.append(
                {
                    "chunk": asdict(self.by_id[cid]),
                    "initial_rank": initial_ranks[cid],
                    "final_rank": rank,
                    "scores": {
                        **{name: values.get(cid) for name, values in scores.items()},
                        "cross_encoder": reranked.get(cid)
                        if getattr(self.reranker, "score_type", "cross_encoder") == "cross_encoder"
                        else None,
                        "reranker_rrf": reranked.get(cid)
                        if getattr(self.reranker, "score_type", "cross_encoder") == "reranker_rrf"
                        else None,
                    },
                }
            )
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        result = {
            "query": query,
            "mode": mode,
            "reranked": rerank,
            "timings": timings,
            "components": components,
            "initial_ranking": initial,
            "final_ranking": final,
            "results": rows,
        }
        event("retrieval", mode=mode, reranked=rerank, candidates=len(initial), **timings)
        return result

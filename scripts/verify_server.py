"""Real-model server smoke test; uses fixture copies and never reads held-out labels."""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import yaml

from erag.config import load_config
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json, build_index
from erag.storage import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:16333")
    args = parser.parse_args()
    root = Path("artifacts/server-check")
    root.mkdir(exist_ok=True)
    shutil.copytree("data/stage-1.1", root / "data", dirs_exist_ok=True)
    cfg = replace(
        load_config("config/default.yaml"),
        dataset=str(root / "data"),
        state=str(root / "index"),
        qdrant_url=args.url,
    )
    embedder, reranker = SentenceEmbedder(cfg), CrossEncoderReranker(cfg)
    store = QdrantStore(url=args.url)
    collections = []
    results = []
    local = QdrantStore()
    local_cfg = replace(cfg, state=str(root / "local-index"), qdrant_url=None)
    try:
        snapshot = build_index(cfg, embedder, store, [])
        collections.append(snapshot["collection"])
        engine = Engine(cfg, store, embedder, reranker)
        build_index(local_cfg, embedder, local, [])
        reference = Engine(local_cfg, local, embedder, reranker)
        for mode in ("bm25", "vector", "hybrid"):
            for rerank in (False, True):
                result = engine.search("What IP address does pe-ledger-prod-01 have?", mode, rerank)
                expected = reference.search(result["query"], mode, rerank)
                assert [x[0] for x in result["final_ranking"]] == [
                    x[0] for x in expected["final_ranking"]
                ]
                # Source relevance cutoff follows agreed document Recall@3, not an assumed top-1.
                rank = next(
                    x["final_rank"]
                    for x in result["results"]
                    if x["chunk"]["document_id"] == "azure-networking"
                )
                assert rank <= 3
                result["expected_source_rank"] = rank
                result["matches_embedded_ranking"] = True
                results.append(result)
        rebuilt = build_index(cfg, embedder, store)
        collections.append(rebuilt["collection"])
        assert rebuilt["chunk_hash"] == snapshot["chunk_hash"]
        source = Path(cfg.dataset) / "aks-platform.txt"
        original = source.read_text()
        source.write_text(original + "\n\nServer integration update fixture.\n")
        changed = build_index(cfg, embedder, store)
        collections.append(changed["collection"])
        old_ids = {c["id"] for c in snapshot["chunks"] if c["document_id"] == "aks-platform"}
        changed_ids = {c["id"] for c in changed["chunks"] if c["document_id"] == "aks-platform"}
        assert old_ids.isdisjoint(changed_ids)
        Engine(cfg, store, embedder, reranker)
        source.write_text(original)
        deleted = build_index(cfg, embedder, store, ["certificate-management"])
        collections.append(deleted["collection"])
        engine = Engine(cfg, store, embedder, reranker)
        assert len({c.document_id for c in engine.chunks}) == 4
        assert all(c.document_id != "certificate-management" for c in engine.chunks)
        restored = build_index(cfg, embedder, store, [])
        collections.append(restored["collection"])
        assert restored["chunk_hash"] == snapshot["chunk_hash"]
        Engine(cfg, store, embedder, reranker)
        image = yaml.safe_load(Path("compose.yaml").read_text())["services"]["qdrant"]["image"]
        atomic_json(
            Path("artifacts/server-verification.json"),
            {
                "passed": True,
                "image": image,
                "server_version": store.client.info().version,
                "launch": "docker run (Compose CLI unavailable)",
                "six_retrieval_variants": True,
                "idempotent_rebuild": True,
                "updates": True,
                "logical_deletion_restore": True,
                "source_fixture_count": 5,
                "corpus_hash": snapshot["corpus_hash"],
                "chunk_hash": snapshot["chunk_hash"],
                "initial_smoke_observation": "Unreranked BM25 ranks the expected source second for the smoke paraphrase on both backends; the initial top-1 assumption failed. No retrieval parameters or official labels changed.",
                "heldout_labels_used": False,
                "model_identity": embedder.identity,
                "reranker_identity": reranker.identity,
                "initial_index_ms": snapshot["index_ms"],
                "queries": results,
            },
        )
    finally:
        for collection in collections:
            store.client.delete_collection(collection)
        store.close()
        local.close()


if __name__ == "__main__":
    main()

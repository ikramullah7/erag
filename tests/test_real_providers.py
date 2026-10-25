"""Opt-in real model and Qdrant server integration; defaults never download models."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEmbedder

from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, build_index
from erag.storage import QdrantStore


@pytest.mark.skipif(
    os.getenv("ERAG_MODEL_TESTS") != "1", reason="set ERAG_MODEL_TESTS=1 for cached real models"
)
@pytest.mark.parametrize("section_boundaries", [False, True])
def test_real_model_index_retrieval_and_reranking(config, section_boundaries):
    config = replace(config, section_boundaries=section_boundaries, bm25_weight=3)
    embedder, reranker = SentenceEmbedder(config), CrossEncoderReranker(config)
    store = QdrantStore(path=str(Path(config.state) / "real-qdrant"))
    try:
        build_index(config, embedder, store)
        engine = Engine(config, store, embedder, reranker)
        for mode in ("bm25", "vector", "hybrid"):
            result = engine.search("What IP address does pe-ledger-prod-01 have?", mode, True)
            assert result["results"][0]["chunk"]["document_id"] == "azure-networking"
        vectors = embedder.encode(["ledger private DNS", "TLS certificate renewal"])
        assert vectors.shape == (2, 384)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    finally:
        store.close()


@pytest.mark.skipif(
    not os.getenv("ERAG_QDRANT_URL"), reason="set ERAG_QDRANT_URL for Docker server integration"
)
def test_server_parity_and_payload_consistency(config):
    embedder = FakeEmbedder()
    local = QdrantStore()
    remote = QdrantStore(url=os.environ["ERAG_QDRANT_URL"])
    remote_cfg = replace(config, state=str(Path(config.state) / "server"))
    snapshots = []
    try:
        a = build_index(config, embedder, local)
        b = build_index(remote_cfg, embedder, remote)
        snapshots.append(b["collection"])
        assert a["chunk_hash"] == b["chunk_hash"]
        left = Engine(config, local, embedder)
        right = Engine(remote_cfg, remote, embedder)
        for mode in ("bm25", "vector", "hybrid"):
            q = "DNS ledger certificate"
            ids_a = [x[0] for x in left.search(q, mode)["final_ranking"]]
            ids_b = [x[0] for x in right.search(q, mode)["final_ranking"]]
            assert ids_a == ids_b
    finally:
        for collection in snapshots:
            remote.client.delete_collection(collection)
        local.close()
        remote.close()

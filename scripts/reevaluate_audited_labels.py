"""Re-evaluate fixed profiles on the approved label version; no tuning or promotion."""

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from erag.config import Config, load_config
from erag.diagnostics import configure_logging, event
from erag.evaluation import (
    environment,
    evaluate,
    format_target_table,
    load_cases,
    load_quality_targets,
    quality_gate,
    ranking_metrics,
)
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore

LABEL_ROOT = Path("data/stage-1.1/evaluations/source-audit-v1")
OUT = Path("artifacts/source-audit-v1-reevaluation")
# All choices predate the audit. Their parameters are loaded unchanged.
PROFILES = (
    ("original-bm25", "artifacts/experiments/selected.yaml", "bm25", False, None),
    ("original-hybrid", "artifacts/experiments/selected.yaml", "hybrid", False, None),
    ("weighted-hybrid", "artifacts/revision-v2/selected.yaml", "hybrid", False, None),
    ("bge-full-context", "artifacts/revision-v5/selected.yaml", "hybrid", True, None),
    (
        "bge-windows",
        "artifacts/revision-v6/selected.yaml",
        "hybrid",
        True,
        "artifacts/revision-v6/validation.json",
    ),
    (
        "bge-rank-fusion",
        "artifacts/revision-v7/selected.yaml",
        "hybrid",
        True,
        "artifacts/revision-v7/validation.json",
    ),
)


def rescore_cached(engine, cases_path, cached_path, mode, rerank):
    cached = json.loads(Path(cached_path).read_text())
    if asdict(Config(**cached["config"])) != asdict(engine.cfg):
        raise ValueError("cached model/configuration mismatch")
    if (
        cached["corpus_hash"] != engine.snapshot["corpus_hash"]
        or cached["chunk_hash"] != engine.snapshot["chunk_hash"]
    ):
        raise ValueError("cached source/chunk snapshot mismatch")
    cases = {c["id"]: c for c in load_cases(cases_path, engine)}
    selected = [r for r in cached["queries"] if r["mode"] == mode and r["reranked"] == rerank]
    if len(selected) != len(cases) or {r["case_id"] for r in selected} != set(cases):
        raise ValueError("cached question coverage mismatch")
    report = deepcopy(cached)
    rows = []
    for old in selected:
        case = cases[old["case_id"]]
        if case["query"] != old["diagnostics"]["query"]:
            raise ValueError("query text changed; cached rankings cannot be reused")
        row = deepcopy(old)
        chunks = [cid for cid, _ in row["diagnostics"]["final_ranking"]]
        docs = list(dict.fromkeys(engine.by_id[cid].document_id for cid in chunks))
        dm = ranking_metrics(docs, case["relevant_documents"], [3, 5])
        cm = ranking_metrics(chunks, case["relevant_chunks"], [3, 5, 10])
        dm.pop("ndcg@3", None)
        cm.pop("ndcg@3", None)
        row.update(
            document_metrics=dm,
            chunk_metrics=cm,
            relevant_documents=case["relevant_documents"],
            relevant_chunks=case["relevant_chunks"],
            answerable=bool(dm),
            failed=bool(dm and (dm["recall@3"] < 1 or cm["recall@5"] < 1)),
        )
        rows.append(row)
    answerable = [r for r in rows if r["answerable"]]

    def means(group, kind):
        return {k: float(np.mean([r[kind][k] for r in group])) for k in group[0][kind]}

    categories = {}
    for category in sorted({r["category"] for r in answerable}):
        group = [r for r in answerable if r["category"] == category]
        categories[category] = {
            "document": means(group, "document_metrics"),
            "chunk": means(group, "chunk_metrics"),
        }
    key = mode + ("+rerank" if rerank else "")
    summary = {
        "document": means(answerable, "document_metrics"),
        "chunk": means(answerable, "chunk_metrics"),
        "latency_ms_p50": cached["summary"][key]["latency_ms_p50"],
        "latency_ms_p95": cached["summary"][key]["latency_ms_p95"],
        "failed_queries": [r["case_id"] for r in answerable if r["failed"]],
        "no_answer_queries": [r["case_id"] for r in rows if not r["answerable"]],
        "answerable_count": len(answerable),
        "by_category": categories,
    }
    report.update(
        queries=rows,
        summary={key: summary},
        created_unix=time.time(),
        labels_hash=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        ranking_measurement_source=cached_path,
        reevaluation_environment=environment(),
        inference_reexecuted=False,
        latency_source="original frozen run; not remeasured",
    )
    return report


def main():
    if OUT.exists():
        raise ValueError("Do not overwrite the audited-label re-evaluation")
    manifest = json.loads((LABEL_ROOT / "manifest.json").read_text())
    if manifest["status"] != "USER_APPROVED_LABEL_VERSION":
        raise ValueError("label version is not approved")
    for name, expected in manifest["original_hashes"].items():
        if hashlib.sha256((Path("data/stage-1.1") / name).read_bytes()).hexdigest() != expected:
            raise ValueError("original labels changed")
    for name, expected in manifest["proposed_hashes"].items():
        if hashlib.sha256((LABEL_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("approved labels changed")
    OUT.mkdir()
    configure_logging()
    labels = LABEL_ROOT / "heldout-v4.json"
    frozen = {
        name: {
            "config": asdict(load_config(config_path)),
            "mode": mode,
            "rerank": rerank,
            "cached_rankings": cache,
        }
        for name, config_path, mode, rerank, cache in PROFILES
    }
    atomic_json(
        OUT / "frozen-profiles.json",
        {
            "profiles": frozen,
            "label_version": "source-audit-v1",
            "targets": load_quality_targets(),
            "label_hash": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "tuning_authorized_or_executed": False,
        },
    )
    embedder = None
    summaries = {}
    for name, config_path, mode, rerank, cache in PROFILES:
        cfg = load_config(config_path)
        store = QdrantStore(path=str(Path(cfg.state) / "qdrant"), url=cfg.qdrant_url)
        try:
            if cache:
                engine = Engine(cfg, store)
                report = rescore_cached(engine, labels, cache, mode, rerank)
            else:
                if mode != "bm25" and embedder is None:
                    embedder = SentenceEmbedder(cfg)
                ranker = CrossEncoderReranker(cfg) if rerank else None
                engine = Engine(cfg, store, embedder if mode != "bm25" else None, ranker)
                report = evaluate(
                    engine,
                    labels,
                    OUT / f"{name}.json",
                    f"source-audit-v1-{name}",
                    "heldout-v4",
                    strategies=((mode, rerank),),
                )
                report["inference_reexecuted"] = True
                report["latency_source"] = "measured in this run"
            report.update(
                label_version="source-audit-v1",
                label_manifest=str(LABEL_ROOT / "manifest.json"),
                experiment_id=f"source-audit-v1-{name}",
                query_set_already_observed=True,
                parameters_changed=False,
                independent_validation_claimed=False,
            )
            atomic_json(OUT / f"{name}.json", report)
            key = mode + ("+rerank" if rerank else "")
            summary = report["summary"][key]
            gate = quality_gate(summary, rerank)
            summaries[name] = {
                "chunk": summary["chunk"],
                "document": summary["document"],
                "gate": gate,
                "answerable_count": summary["answerable_count"],
                "inference_reexecuted": report["inference_reexecuted"],
                "latency_ms_p95": summary["latency_ms_p95"],
                "latency_source": report["latency_source"],
            }
            event("audited_labels_profile_complete", profile=name, gate_passed=gate["passed"])
            print(
                f"{name}: {json.dumps({k: summary['chunk'][k] for k in load_quality_targets()})}",
                flush=True,
            )
        finally:
            store.close()
    atomic_json(
        OUT / "comparison.json",
        {
            "label_version": "source-audit-v1",
            "profiles": summaries,
            "parameters_tuned": False,
            "profile_promoted": False,
            "original_labels_changed": False,
            "disputed_proposals_applied": False,
            "independent_validation_claimed": False,
        },
    )
    print("\nLatest frozen candidate | approved corrected labels | 60 answerable questions")
    print(format_target_table(summaries["bge-rank-fusion"]))
    return 0 if any(s["gate"]["passed"] for s in summaries.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

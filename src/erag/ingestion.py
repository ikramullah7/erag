"""Validated plain-text ingestion; explicit approved document stages are enforced."""

import hashlib
from pathlib import Path

from .types import Document

APPROVED_IDS = frozenset(
    {
        "aks-platform",
        "kubernetes-troubleshooting",
        "azure-networking",
        "certificate-management",
        "observability",
    }
)


STAGE_12_IDS = APPROVED_IDS | frozenset(
    {
        "postgresql-recovery",
        "redis-availability",
        "workload-identity",
        "release-management",
        "incident-response",
    }
)


STAGE_13_IDS = STAGE_12_IDS | frozenset(
    {
        "servicebus-recovery",
        "vault-rotation",
        "gitops-reconciliation",
        "patch-compliance",
        "storage-access",
        "disaster-recovery",
        "logging-governance",
        "secret-delivery",
        "capacity-management",
        "ingress-routing",
    }
)


def approved_ids(stage: str):
    if stage == "1.1":
        return APPROVED_IDS
    if stage == "1.2":
        return STAGE_12_IDS
    if stage == "1.3":
        return STAGE_13_IDS
    raise ValueError("unapproved document stage")


def ingest(directory: str | Path, stage: str = "1.1") -> list[Document]:
    root = Path(directory)
    files = sorted(root.glob("*.txt"))
    expected = approved_ids(stage)
    if {p.stem for p in files} != expected or len(files) != len(expected):
        raise ValueError(
            f"Stage {stage} requires exactly the {len(expected)} approved document IDs"
        )
    documents, seen = [], set()
    for p in files:
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"unsafe document: {p}")
        if p.stat().st_size > 2_000_000:
            raise ValueError(f"document too large: {p}")
        text = p.read_bytes().decode("utf-8")
        if not text.strip() or "\x00" in text:
            raise ValueError(f"empty or binary document: {p}")
        version = hashlib.sha256(text.encode()).hexdigest()
        normalized = hashlib.sha256(text.strip().encode()).hexdigest()
        if normalized in seen:
            raise ValueError(f"duplicate document content: {p}")
        seen.add(normalized)
        documents.append(Document(p.stem, p.name, version, text))
    return documents

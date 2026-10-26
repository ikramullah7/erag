"""Fetch only the five authorized manifest entries; preserve raw and extracted internal copies."""

import argparse
import csv
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from erag.article_extraction import KubernetesArticle
from erag.pipeline import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="artifacts/real-world/stage1/acquisition-001")
root = Path(parser.parse_args().output)
if root.exists():
    raise ValueError("Preserve earlier acquisitions; choose a new run ID")
(root / "raw").mkdir(parents=True)
(root / "corpus").mkdir()
manifest = Path("data/real-world/stage1/rag_official_docs_30.csv")
rows = list(csv.DictReader(manifest.open()))[:5]
expected = ["Deployments", "StatefulSets", "DaemonSet", "Service", "Ingress"]


def fetch(item):
    i, row = item
    try:
        url = row["url"]
        if urlparse(url).hostname != "kubernetes.io" or urlparse(url).scheme != "https":
            raise ValueError("unapproved first-stage host")
        with urlopen(
            Request(url, headers={"User-Agent": "EnterpriseRAG-local-validation/1.0"}), timeout=45
        ) as response:
            status = response.status
            final = response.geturl()
            headers = dict(response.headers)
            raw = response.read(8_000_001)
        if status != 200 or len(raw) > 8_000_000:
            raise ValueError("non-200 or oversized response")
        if (
            urlparse(final).hostname != "kubernetes.io"
            or urlparse(final).path != urlparse(url).path
        ):
            raise ValueError("unexpected redirect; no page substitution")
        html = raw.decode("utf-8")
        parser = KubernetesArticle()
        parser.feed(html)
        title, text = parser.result()
        h1 = text.splitlines()[0].removeprefix("# ").strip()
        if h1 != expected[i]:
            raise ValueError("unexpected article heading: " + h1)
        cid = f"kubernetes-{row['id']}-{urlparse(url).path.rstrip('/').split('/')[-1]}"
        (root / "raw" / (cid + ".html")).write_bytes(raw)
        (root / "corpus" / (cid + ".txt")).write_text(text)
        meta = {
            "id": cid,
            "manifest_id": row["id"],
            "manifest_title": row["title"],
            "title": title,
            "article_heading": h1,
            "requested_url": url,
            "final_url": final,
            "http_status": status,
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "last_modified_header": headers.get("Last-Modified"),
            "html_sha256": hashlib.sha256(raw).hexdigest(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "words": len(text.split()),
            "characters": len(text),
            "product_version": "Unversioned rolling Kubernetes documentation; no product version inferred from navigation.",
            "version_mentions": sorted(set(re.findall(r"Kubernetes v\d+\.\d+", text))),
            "extraction": "main div.td-content; headings, fenced command blocks and table rows retained; navigation/feedback excluded",
            "title_note": "Manifest plural vs official singular title recorded, without changing page"
            if row["title"] != h1
            else None,
            "status": "verified",
        }
        atomic_json(root / (cid + ".metadata.json"), meta)
        return meta
    except (ValueError, OSError, UnicodeError) as exc:
        return {
            "manifest_id": row["id"],
            "requested_url": row["url"],
            "status": "failed",
            "error": str(exc),
        }


with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(fetch, enumerate(rows)))
report = {
    "track": "real-world-stage1",
    "status": "corpus_frozen"
    if all(r["status"] == "verified" for r in results)
    else "acquisition_failed",
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "authorized_documents": 5,
    "documents": results,
    "evaluation_status": "Not evaluated; user questions and source-grounded labels required.",
}
atomic_json(root / "acquisition.json", report)
for row in results:
    print(
        json.dumps(
            {
                k: row.get(k)
                for k in ("manifest_id", "status", "title", "http_status", "words", "error")
            }
        )
    )
raise SystemExit(0 if report["status"] == "corpus_frozen" else 2)

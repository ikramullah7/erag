"""Preserve measured quality attempts as data, including failures; no narrative report."""

import gzip
import hashlib
import json
from pathlib import Path

from erag.pipeline import atomic_json


def main() -> None:
    destination = Path("docs/results/improvements")
    destination.mkdir(exist_ok=True)
    manifest = {}
    for root in sorted(Path("artifacts").glob("revision-v*")):
        if not root.is_dir():
            continue
        target = destination / root.name
        target.mkdir(exist_ok=True)
        for source in sorted(root.glob("*.json")):
            raw = source.read_bytes()
            output = target / (source.name + ".gz")
            output.write_bytes(gzip.compress(raw, mtime=0))
            manifest[str(output.relative_to(destination))] = {
                "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
                "compressed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        for source in sorted(root.glob("*.yaml")):
            (target / source.name).write_bytes(source.read_bytes())
    atomic_json(destination / "archive-manifest.json", manifest)
    for name in ("current-tests.xml", "current-tests.log"):
        source = Path("artifacts") / name
        if source.exists():
            (destination / name).write_bytes(source.read_bytes())
    print(json.dumps({"archived_measurements": len(manifest), "includes_failed_attempts": True}))


if __name__ == "__main__":
    main()

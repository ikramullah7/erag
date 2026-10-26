"""Analyze saved retrieval and answer evidence, without inference or label mutation."""

import argparse
import hashlib
import json
from pathlib import Path

from erag.answer_metrics import read_json
from erag.evidence_evaluation import analyze_evidence
from erag.pipeline import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "manifest", "ranking", "answers", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve previous analyses; choose a new output")
    report = analyze_evidence(
        read_json(args.pool),
        read_json(args.manifest),
        read_json(args.ranking),
        read_json(args.answers),
    )
    report["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.pool, args.manifest, args.ranking, args.answers)
    }
    atomic_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

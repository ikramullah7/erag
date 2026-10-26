"""Prepare immutable real-document/question metadata without indexing or model calls."""

import argparse
import json
from pathlib import Path

from erag.real_validation import prepare_real_validation

parser = argparse.ArgumentParser()
parser.add_argument("--documents", required=True)
parser.add_argument("--questions", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
manifest = prepare_real_validation(
    args.documents, json.loads(Path(args.questions).read_text()), args.output
)
print(
    f"Prepared {len(manifest['sources'])} documents and {len(manifest['questions'])} questions; evaluation not run."
)

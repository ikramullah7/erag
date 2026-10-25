"""Calculate the reviewed five-document baseline without an LLM or active index."""

import argparse

from erag.answer_metrics import format_metrics
from erag.reviewed_metrics import score_reviewed_answers

parser = argparse.ArgumentParser()
parser.add_argument("--answers", default="docs/results/answers-phase-1.1/v2-qwen3-14b.json.gz")
parser.add_argument("--references", default="data/stage-1.1/answer-semantic-references-v1.json")
parser.add_argument("--reviews", default="data/stage-1.1/answer-reviewed-baseline-v1.json")
parser.add_argument("--output", required=True)
args = parser.parse_args()
report = score_reviewed_answers(args.answers, args.references, args.reviews, args.output)
print(format_metrics(report).replace("local semantic judge estimate", "agent-reviewed baseline"))

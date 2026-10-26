# Real-document input preparation utility

The selected real pilot now uses `data/real-world/stage1/`: five verified official Kubernetes
articles, twenty user-supplied questions and two labelled agent-authored controls. Initial results
are in `docs/results/real-world-stage1/summary.json`; quality gates fail and human label/scope
adjudication is pending. Approval is required before expanding beyond the first five articles.

The generic utility here prepares source/question metadata only. Its twenty-file cap does not
authorize a twenty-document pilot or bypass the narrower real-1 ingestion guard. Source inputs
must be selected by the user; author provenance is recorded rather than independently verified.

`questions.template.json` is an empty format example, not an authored benchmark. Each actual
question entry contains an `id` and `query`, with an author recorded at the document level. Plain
language questions may be formatted by the assistant without changing their wording. Keep
provided topic/context metadata separately and record whether it is passed to retrieval.
Do not generate questions from source wording and label them independent human questions.

```sh
uv run python scripts/prepare_real_validation.py --documents /selected/document/folder --questions /selected/questions.json --output artifacts/real-validation/input-manifest.json
```

This validates UTF-8 `.txt` files, duplicate content, question IDs/queries and output preservation,
then freezes source/question hashes. It does not index or evaluate the inputs. Actual ingestion
requires a separately authorized, verified manifest/index profile. Keep publisher copies internal,
source versions frozen, and expected facts separate from model-generated answers. Never tune
on an observed evaluation set and call its replay fresh independent validation.

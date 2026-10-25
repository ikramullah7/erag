# Twenty-document Phase 1.3

All ten stage-1.2 source `.txt` files and their chunk IDs are unchanged. Added Service Bus
recovery, vault rotation, regional DR, audit storage, capacity, ingress routing, secret delivery,
GitOps, logging governance and patch compliance. `stage-manifest.json` binds source and
pre-evaluation label hashes.

There are 36 fixed questions: ten retained regressions, twenty prospective supported questions
and six unsupported questions. Tuning: fifteen supported; heldout: fifteen supported and six
unsupported. Labels/reference facts were authored and source-checked before retrieval outputs.
They are synthetic, agent-authored fixtures rather than independent human evidence; source
facts can overlap across questions. No profile tuning/selection uses heldout results.

Exact evidence spans define canonical chunk relevance. Complete evidence has grade 3; required
partial evidence in cross-document questions or secret polling/reload has grade 2. Incidental
topic mentions are excluded. The required facts are limited to requested information and
separate policy settings/objectives from actual live results. New secret delivery documentation
supplies a mount path, not a credential value, so the mixed path/password question remains
unsupported under the all-material-parts abstention policy.

Use explicit profiles; prior-stage commands remain available:

```sh
HF_HUB_OFFLINE=1 uv run erag --config config/stage-1.3.yaml index
HF_HUB_OFFLINE=1 uv run erag answer 'Who approves dead-letter replay?' --answer-config config/answer-stage-1.3.yaml
```

The ranking profile is unchanged from 1.2. Generation remains local Qwen3 14B, with 12,288
context, at most 1,024 output tokens, one validation retry and unique literal quote-fragment
alignment. No sources beyond the approved twenty or external services are added.

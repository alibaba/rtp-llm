# Anonymous master request fixture

`master-request-templates.json` is a deterministic view of the pinned anonymous
prefix trace used by online_eval. It contains request lengths and anonymous
block labels; original token IDs and deployment identifiers are absent.
Its transformations record the source SHA256, request limit, input clipping,
and fixed output length. Shared labels become shared synthetic token blocks
in `MasterBatchEndToEndPerformanceTest`.

Regenerate from the FlexLB directory:

```bash
python3 tools/online_eval/scripts/pipeline/derive_master_templates.py \
  --out flexlb-api/src/test/resources/master-request-templates.json
```

The online_eval dataset test verifies this resource against the pinned source.

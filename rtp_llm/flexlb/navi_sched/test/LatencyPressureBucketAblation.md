# Pressure-matched history ablation

These are offline counterfactuals using the deployed latency fitter and the
logged training sequence. The baseline prediction matches the online log
within 0.000001 ms for every included prediction. MAE is mean absolute
prediction error, in milliseconds. The pressure selector uses the logged
sample's `running + waiting` count at admission, **not** inflight at fit time.
The `low` and `high` results below mean at or below the per-node first
quartile and at or above the per-node third quartile of that count within the
window; they are not fixed QPS bands shared across hardware.

| Window | Baseline base scale / residual rate | MI / RTX predictions | Baseline MI / RTX MAE | Add one current-sample-bucket history item MI / RTX MAE |
| --- | --- | ---: | ---: | ---: |
| A, changing pressure | 16 / 0.8 | 4474 / 3110 | 5931 / 1507 | 5962 / 1514 |
| B, short changing pressure | 0.7 / 0.08 | 2095 / 1229 | 2079 / 1626 | 2082 / 1627 |
| C, mostly steady | 0.7 / 0.08 | 10195 / 5501 | 1995 / 1691 | 1991 / 1693 |

The comparisons hold each window's own learning rates constant. Thus B/C's
residual-0.08 results must not be compared directly with the residual-0.8
error-rate experiments. In A, adding one item increases high-pressure MI MAE
from 3965 to 4182 ms and low-pressure RTX MAE from 1482 to 1506 ms. Adding
two or four items raises full-window MI/RTX MAE to 5966/1514 and 6044/1521
ms. Selecting from the latest update's pressure instead also worsens the
full-window result. Rebuilding cross-bucket coverage with one item from each
other populated bucket, plus one, two, or four same-bucket items, is worse
still: A MI MAE spans 6193-6306 ms and RTX
MAE spans 1525-1558 ms across the replacement variants.

This rejects the simple admission-pressure proxy and these sampling rules on
the current traces; it does **not** reject selecting by true fit-time inflight.
The one-item intervention changed all 1900/836/3925 fitting updates in A/B/C
(four fit batches per update), so the negative result is not due to a no-op.
Only 7/7600, 6/3344, and 20/15700 appended items respectively duplicated
an item already in their original batch.
On 2026-10-08, the sampled online MI node had 7770 training updates but no
`fit_inflight_before_requests` field, so that experiment still needs an image
containing the audit field and fresh logs. No online configuration was changed
for this ablation.

The fit-inflight selector now assigns updates to an experiment window using
`fit_inflight_before_time_us` from the training-update log, not the sample's
earlier `event_time_us`. Missing timestamps and reversed capture times fail
validation. It uses the valid snapshot taken immediately before fitting even
if inflight changes before the post-fit snapshot; the stable/unstable counts
remain separate diagnostics. Excluding unstable updates would preferentially
discard the dynamic-pressure cases being tested. A missing or invalid pre-fit
snapshot is never substituted with the sample's admission-time count. The
online image has not yet supplied this field, so there is still no measured
fit-time-pressure result to compare against the admission-pressure proxy above.

Generated summaries: `/tmp/navi-pressure-bucket-{A,B,C}-20261008.jsonl`.
Tool: `CompareLatencyPressureBuckets.py`. The full A grid uses both `augment`
and `replace`, current-sample and update-latest selectors, and 1/2/4 extra
items. B/C use `augment`, current-sample and one extra item. The baseline's
maximum log difference is 1.8e-7 ms in A and below 1e-9 ms in B/C.

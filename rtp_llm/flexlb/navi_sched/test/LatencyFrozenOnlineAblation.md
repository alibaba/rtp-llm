# Frozen online-learning comparison

Window A (2026-10-07, changing replay pressure) was replayed with the logged
weight 4, base learning-rate scale 16 and residual learning rate 0.8. The
counterfactual anchor for each node is its last logged model update before
the first sample in the window whose observed latency is at least 20 seconds.
Only subsequent samples with observed latency at least 20 seconds are scored.
This is a **latency-threshold anchor**, not a model fitted exclusively on
low-pressure data. `freeze` stops both updates, `freeze_base` stops base-model
updates but continues residual updates, and `freeze_residual` does the reverse.

MAE is mean absolute final-prediction error, in milliseconds. The online
column is the same matched request set as its counterfactual, not an
independently selected traffic slice.

| Hardware | Matched samples | Online MAE | Freeze both MAE | Freeze base MAE | Freeze residual MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| MI308 | 3971 | 6629 | 18588 | 6861 | 12879 |
| RTX5000 | 2859 | 1559 | 3022 | 1547 | 2559 |

Stopping residual updates loses about 6.25 seconds of MI accuracy and 1.00
second of RTX accuracy on this set. Stopping base updates changes MI MAE by
only +232 ms and RTX by -12 ms. These effects are not additive: when both
updates stop, each model also sees a different residual input and trajectory.

MI's base-model contribution changes over time. During the first ten minutes
after the anchor, online versus base-frozen MAE is 11936 versus 11320 ms:
continuing base updates is worse by 616 ms. After ten minutes it is 4956
versus 5455 ms: continuing base updates helps by 499 ms. For RTX the same
early/late pairs are 1456/1471 and 1599/1577 ms, close in both directions.
This is consistent with an initial MI base overshoot followed by useful
adaptation, but the aggregate alone cannot establish a causal root cause.

An independent anchor immediately after the first >=20-second sample gives
the same broad ordering: MI online/freeze-both/freeze-base/freeze-residual
MAE 6445/19053/6844/12712 ms; RTX 1551/2475/1534/2630 ms. The two anchor
sets overlap heavily and are not independent replications. Further tests
must freeze at a verified low-pressure fitting state and evaluate later high
pressure; this replay does not yet answer low-to-high parameter transfer.

## Low-current-pressure anchor

An explicit A-window anchor was then chosen before event time
`1791377550000000`: for each node, use the latest fitting update whose
current sample has `running + waiting <= 20`. The selected counts were
17/20/20 for the three MI nodes and 17/1/1 for the three RTX nodes. Scoring
begins at each node's first >=20-second-latency sample after
`1791377720000000`. This separates anchor and scoring times and keeps the
same online request set within each comparison.

| Hardware | Matched samples | Online MAE | Freeze both MAE | Freeze base MAE | Freeze residual MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| MI308 | 3964 | 6468 | 18299 | 6485 | 12727 |
| RTX5000 | 2600 | 1582 | 3136 | 1572 | 2541 |

Even at these low-current-pressure anchors, residual updating accounts for
most of the online gain. Freezing base changes MI by +17 ms and RTX by -10
ms, while freezing residual changes them by +6259 and +959 ms respectively.
This does **not** establish a low-pressure-only model: the anchor's fitting
batch includes older high-pressure samples (per-node batch pressure P90
104-207, maximum 131-289), and its parameters already include all training
before the A window. To test low-only-to-high generalization, a separately
fitted low-only model or clean low-only initialization is still required.
The near-equality of online and base-frozen final MAE therefore does not
contradict `LatencyPhaseLeastSquaresReport.md`: that report fits a separate
low-only static model, finds weakly identified MI parameters and poor
low-to-high extrapolation. Here the anchor inherits high-pressure history and
the residual continues learning after the anchor.

Generated trace summary:
`/tmp/navi-freeze-low-sample-anchor-A-20261008.jsonl`. The explicit anchor
selection has a focused UT for time and pressure bounds.

## Independent B/C window check

The same freeze modes were replayed on B (short changing pressure) and C
(mostly steady), using their logged weight 4, base scale 0.7 and residual
rate 0.08. The anchor is immediately after the first >=20-second-latency
sample, not a verified pressure transition. Each row compares an identical
matched request set within that window; do not compare its absolute MAE with
A's residual-0.8 rows as an isolated pressure effect.

| Window | Hardware | Samples | Online MAE | Freeze base MAE | Freeze residual MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| B | MI308 | 2091 | 2077 | 2039 | 4642 |
| B | RTX5000 | 1229 | 1626 | 1626 | 1940 |
| C | MI308 | 10191 | 1995 | 1978 | 5517 |
| C | RTX5000 | 5497 | 1692 | 1685 | 1787 |

Residual updating again makes the larger contribution to the final
prediction, especially on MI. Freezing base improves the B/C final MAE by
at most 38 ms in these windows, but this does not imply that the base formula
or its features are unnecessary: the continuing residual can compensate for
a frozen base, and each counterfactual follows a different update path. In B,
only one MI node has an update before its first >=20-second sample in the
recorded window, so the table uses the post-sample anchor for all nodes.

Generated summaries: `/tmp/navi-freeze-{B,C}-20261008.jsonl`.

Generated trace summary: `/tmp/navi-freeze-A-20261008.jsonl`.
Tool: `CompareLatencyFrozenState.py`; its anchor-selection UT passes.

# Error-rate learning rate: current evidence

This is an offline counterfactual, not an online result. MAE is mean absolute
prediction error; P95 is the 95th percentile of absolute error. All values
below are milliseconds. Replay baseline predictions match the logged online
base and final predictions within 0.1 ms for every included request.

## Comparable windows

| Window | MI / RTX predictions | Fixed base scale / residual rate | Pressure coverage |
| --- | ---: | --- | --- |
| A | 4474 / 3110 | 16 / 0.8 | Changing traffic, including sparse high-error intervals |
| B | 2095 / 1229 | 0.7 / 0.8 | Short changing-traffic interval, actual ingress 19-403 QPS; online residual rate was 0.08 |
| C | 10195 / 5501 | 0.7 / 0.8 | Mostly steady high pressure; online residual rate was 0.08 |

All candidate comparisons hold the window's fixed residual rate at 0.8.
Changing residual from the logged 0.08 to 0.8 accounts for most of the MI
improvement in B and C; it is not an error-rate learning-rate effect.

| Window | Fixed MI / RTX MAE | Linear gain 256 MI / RTX MAE | Hinge gain 512, threshold 0.1, cap 32 MI / RTX MAE |
| --- | --- | --- | --- |
| A | 5931 / 1507 | 4066 / 1533 | 5502 / 1512 |
| B | 1296 / 1679 | 1227 / 1736 | 1276 / 1680 |
| C | 1130 / 1663 | 1100 / 1719 | 1130 / 1662 |

The linear controller uses
`clamp(256 * abs(actual - base) / max(actual, base, 500), minimum, maximum)`.
The hinge controller uses
`clamp(minimum + 512 * max(0, error_ratio - 0.1), minimum, 32)`. Window A uses
minimum 16; B and C use minimum 0.7. Lowering linear gain to 128/64/32 in C
reduced the RTX penalty but also removed the extra MI benefit. Raising a
common fixed scale to 32/64 in A or 16/32/64 in B and C worsened RTX more
than the bounded hinge candidate did.

Two-minute paired-block resampling of the hinge/cap-32 candidate against its
fixed-rate control gives MAE changes (candidate minus control): A MI -430 ms
(95% interval -838 to -107), A RTX +5 ms (-2 to +12); B MI -20 ms (-45 to +4),
B RTX +1 ms (+0.3 to +1.8); C MI 0 ms (-11 to +11), C RTX -1 ms (-4 to +2).
Intervals describe variation across these time blocks, not a new deployment.

## Bounded error-rate candidate (cap 96)

This comparison holds the residual rate fixed at 0.8 on **both** sides. The
fixed base scale is 16 in A and 0.7 in B/C. The candidate uses hinge threshold
10%, gain 512, maximum base scale 96, and error-ratio normalizer floor 500 ms.
The production-controller replay with this explicit floor has byte-identical
prediction rows to the independent candidate trace in A. It is not compared to B/C's logged
residual-0.08 predictions when attributing the error-rate effect.

| Window | Hardware | Samples | Fixed MAE (ms) | Candidate MAE (ms) | Change (ms) | Paired 2-minute node-block 95% interval for change (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | MI308 | 4474 | 5931 | 4479 | -1452 | -2411 to -549 |
| A | RTX5000 | 3110 | 1507 | 1524 | +18 | +1 to +38 |
| B | MI308 | 2095 | 1296 | 1277 | -19 | -46 to +6 |
| B | RTX5000 | 1229 | 1679 | 1680 | +1 | +0.3 to +1.8 |
| C | MI308 | 10195 | 1130 | 1128 | -3 | -14 to +9 |
| C | RTX5000 | 5501 | 1663 | 1662 | -1 | -4 to +2 |

In A, all three MI nodes improve (MAE changes -1514, -1864, and -981 ms).
All three RTX nodes worsen (+49, +7, and +1 ms). The intervals resample
node-by-time blocks from these same traces, so they describe temporal
variation, not uncertainty across independent deployments. One five-minute
RTX block on node 215.2.192.71 worsens by about 227 ms,
so the aggregate +18 ms masks a local regression. On matching zero-
point updates, the candidate's final zero-point absolute drift has mean/P95
3.57/8.50 ms for A MI and 2.08/7.02 ms for A RTX; B is 0.37/1.15 and
0.01/0.02 ms; C is 1.35/3.43 and 0.07/0.13 ms. This does not establish
long-term zero-point stability.

The same A-window samples and floor-500-ms normalizer show the following
cap sensitivity, relative to the fixed-scale-16 control. All rows use the
10% hinge threshold, gain 512, and residual rate 0.8; negative MAE change
means improvement. The local block is node 215.2.192.71 at 20:50-20:55
Asia/Shanghai on 2026-10-07 (82 predictions).

| Maximum base scale | MI MAE change (ms) | RTX MAE change (ms) | Local RTX block MAE change (ms) |
| ---: | ---: | ---: | ---: |
| 32 | -430 | +5 | +54 |
| 64 | -1126 | +14 | +150 |
| 96 | -1452 | +18 | +227 |

For the cap-96 production-controller trace, the local block starts with
base scale 16; several high-error updates reach 96. During the following
high-latency rise, predictions for three consecutive requests reach about
60-65 seconds against actual latencies of 54-56 seconds, adding 3.5-3.8
seconds of absolute error per request relative to fixed scale 16. The cap
comparison supports a genuine MI-response versus RTX-overshoot tradeoff on
this dataset; it does not establish that the same tradeoff persists online.

## Longer replay from the same activation point

For each of the six nodes, a replay input retained historical samples but
restored the last logged model and Adam state immediately before its first
A-window prediction. Fixed-rate and cap-96 production-controller predictions
then reproduce all 7584 corresponding A-window trace predictions (maximum
fixed-rate difference 0.00000018 ms; cap-96 prediction rows identical).
Starting the candidate even a few updates earlier changed its subsequent
trajectory on two MI nodes, so activation time is part of this comparison.

The same anchored inputs contain later samples. The table compares each
candidate to fixed scale 16 and residual rate 0.8, with threshold 10%, gain
512, and normalizer floor 500 ms held constant. Negative MAE change means
improvement. "First 30 min" is the first two 15-minute bins after
20:50 Asia/Shanghai on 2026-10-07; it contains the observed traffic change.
The later interval is mostly sustained high pressure, not another measured
multi-cycle transition.

| Base-scale cap | MI first 30 min: count / MAE change (ms) | RTX first 30 min: count / MAE change (ms) | MI later: count / MAE change (ms) | RTX later: count / MAE change (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 2244 / -754 | 1829 / +5 | 27057 / -19 | 16114 / +0.4 |
| 64 | 2244 / -2154 | 1829 / +19 | 27057 / -10 | 16114 / +3 |
| 96 | 2244 / -2903 | 1829 / +28 | 27057 / -5 | 16114 / +2 |

The fixed control's first-30-minute MAE is 9841 ms on MI and 1487 ms on RTX;
later it is 1245/1634 ms. Across all 29301 MI and 17943 RTX predictions,
cap 96 changes MAE from 1903 to 1676 ms and 1619 to 1624 ms respectively.
The MI benefit is concentrated early; more high-pressure steady samples dilute
it in the full-window average. After the verified A interval, fixed-rate
counterfactuals are **not** asserted to reproduce the online model trajectory,
and the recorded features do not account for scheduling feedback that a
different online model might create. This is not a replacement for fresh
online multi-cycle validation.

Small aggregate RTX changes also hide later local regressions. Among
five-minute RTX node blocks with at least 30 predictions, the worst MAE
increase is 137 ms at cap 32 (137 predictions on 215.2.206.22), 158 ms at
cap 64 (130 predictions on the same node), and 227 ms at cap 96 (82
predictions on 215.2.192.71). On the cap-32 worst block, fixed/candidate MAE
is 2011/2148 ms. Thus lowering the cap limits one early overshoot but does
not establish local stability across nodes or time.
With cap 96 and the threshold raised from 10% to 20%, the first-30-minute
MI/RTX MAE changes are -2710/+27 ms, and the worst RTX five-minute block
still worsens by 208 ms. A higher threshold alone does not remove the
early-node overshoot.

## Persistent signed-error gate

The hinge can optionally require N consecutive fitting samples with a base
error ratio above 10% **and the same error sign** before applying its excess
learning-rate gain. A small error or sign reversal resets the count. N=1 is
the old behavior and remains the default; this does not change the adaptive
mode or the residual rate. In A's fixed-rate update trace, 1247 of 1533 MI
above-threshold updates but only 22 of 166 RTX updates occur at streak >=4.
This motivated a counterfactual, not a hardware-specific rule.

All rows below use gain 512, cap 96, normalizer floor 500 ms, weight 4, and
residual rate 0.8. Each window is re-anchored independently to the last
logged model and Adam state before its first scored prediction. N=1 exactly
reproduces the previous candidate's prediction rows in A/B/C (7584/3324/
15696 predictions), so the streak comparisons have the same initial state
and samples. MAE changes are relative to each window's fixed-rate control;
negative values are improvements. The last column is the largest RTX
five-minute node-block MAE increase among blocks with at least 30 samples.

| Window | Minimum streak | MI MAE change (ms) | RTX MAE change (ms) | Worst RTX block change (ms) |
| --- | ---: | ---: | ---: | ---: |
| A, changing pressure | 1 | -1452 | +18 | +227 |
| A, changing pressure | 3 | -1324 | +5 | +107 |
| A, changing pressure | 4 | -1196 | +2.5 | +67 |
| B, short changing pressure | 1 | -19 | +1 | +2.7 |
| B, short changing pressure | 3 | -17 | 0 | +0.2 |
| B, short changing pressure | 4 | -18 | 0 | 0 |
| C, mostly steady | 1 | -3 | -0.6 | +21 |
| C, mostly steady | 3 | -8 | -0.6 | +18 |
| C, mostly steady | 4 | -10 | -0.4 | +8 |

In A's first 30 minutes, N=1/3/4 changes MI MAE by -2903/-2603/-2341 ms
and RTX MAE by +28/+8/+3 ms. N=3 retains 91% of N=1's A-window MI gain
while reducing aggregate RTX degradation, but it still leaves a 107-ms
local RTX regression. With the master's actual ingress-QPS metric and the
same causal five-second average for all candidates, A's 1414 labeled MI
rising predictions have fixed/N=1/N=3/N=4 MAE 11197/9346/9580/9825 ms;
829 RTX rising predictions have 1860/1893/1872/1869 ms. These phase labels
use admission time approximated from completion time minus observed latency;
they are not direct per-request ingress timestamps.

An additional A-window cap/streak grid uses the same anchored inputs, weight
4, residual rate 0.8, threshold 10%, gain 512, and floor 500 ms. All rows
below are evaluated on the **intersection** of prediction keys present in
every candidate and the fixed-scale-16 control. The first 30 minutes have
2244 MI and 1829 RTX predictions; the fixed MAE is 9841/1486 ms. Negative
change means improvement. The worst RTX block is the largest five-minute
per-node MAE increase among blocks with at least 30 predictions.

| Minimum streak | Base-scale cap | MI MAE change (ms) | RTX MAE change (ms) | Worst RTX five-minute block change (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 96 | -2903 | +28 | +227 |
| 3 | 32 | -745 | +4 | +28 |
| 3 | 64 | -1805 | +6 | +67 |
| 3 | 96 | -2603 | +8 | +107 |
| 4 | 32 | -705 | +3 | +33 |
| 4 | 64 | -1652 | +3 | +33 |
| 4 | 96 | -2341 | +3 | +33 |

For streak 4, lowering the cap from 96 to 64 loses 689 ms of MI improvement
without reducing the worst RTX block in this interval. Across the full
intersection (4562 MI / 3166 RTX predictions), streak 4 with cap 96 changes
MAE by -1172/+2 ms and has a worst RTX block increase of 33 ms. This is a
more balanced offline pilot than streak 3/cap 96 on these traces, not evidence
that the rule will generalize to a new multi-cycle online run. A fresh replay
with cap 96 reproduces the earlier streak-3 trace on all 47244 common
prediction keys to within 0.000005 ms; the new cap runs include a few
earlier predictions absent from the previous trace, excluded by the
intersection above. The full anchored-input intersection extends beyond the
original A comparison window, so its counts are not directly comparable to
the A-window counts in the preceding table.

For matched zero-point updates, N=3 final zero-point absolute drift P95
relative to fixed control is 3.35/2.65 ms for A MI/RTX in the first 30
minutes, 1.15/0 ms for B, and 3.21/0.08 ms for C. N=4 is 3.75/0.88,
0.75/0, and 3.24/0.04 ms respectively. Neither gate is established as an
online improvement: all three windows share the same deployments, B is
short, C is mostly steady, and no fresh multi-cycle candidate traffic has
been run. N=3 and N=4 remain offline pilot candidates, not new defaults.

For a separate phase check, use the master's actual ingress-QPS metric and
the causal five-second mean at approximate admission time (completion time
minus observed admission-to-prefill-complete latency). Relative to 30 seconds
earlier, a QPS change of at least +10 is rising and at most -10 is falling.
For A MI, rising MAE is 9980 to 7208 ms (2354 predictions), but falling MAE
is 1613 to 1627 ms (1778). For A RTX, rising is 1671 to 1693 ms (1663) and
falling is 1452 to 1467 ms (1152). The P95 of **positive** prediction error
(prediction minus actual, considering only positive errors) in A MI's rising
phase falls from 38692 to 17778 ms; RTX rises from 4480 to 4660 ms. In B,
MI rising/falling MAE is 1337/1262 to 1328/1235 ms, while RTX is 1668/1689
to 1669/1690 ms. C has only 21 downsampled ingress points and is not suitable
for this phase check. The approximate admission time and sparse C metric are
limitations; online multi-cycle traffic remains the decisive check.

Holding gain 512, cap 96, and normalizer floor 500 ms fixed, an A-window
threshold check gives:

| Error-rate threshold | MI MAE (ms) | RTX MAE (ms) | MI / RTX updates above base scale | MI / RTX final zero-point drift P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 5% | 4442 | 1539 | 76.7% / 39.5% | 12.5 / 7.5 |
| 10% | 4479 | 1524 | 57.5% / 8.7% | 8.5 / 7.0 |
| 20% | 4589 | 1527 | 27.9% / 1.0% | 10.6 / 0.5 |

The fixed control is 5931/1507 ms MI/RTX. A higher threshold therefore
does not monotonically protect RTX: with fewer accelerated updates, later
updates can still change its path. The 10% threshold is the best compromise
among these three at cap 96, not an established global optimum. Lowering the
threshold to 5% trades 37 ms of additional MI improvement against 15 ms of
additional RTX degradation relative to 10%, and increases MI zero-point drift.

This is a common error-rate rule, not a hardware-specific switch. In the
logged baseline, the per-sample **base** error ratio in A has median/P90
13.0%/32.8% on MI versus 3.8%/9.2% on RTX; 59.8% versus 7.6% exceed 10%.
In B the exceedance is 27.4%/1.3%, and in C it is 37.9%/1.3% (MI/RTX).
This explains why the rule activates more often on MI, but it does not by
itself establish why MI's base model has larger errors or why the rare RTX
accelerations cause its local regression.

## Squared-error-rate control without a persistence gate

The production controller also supports a direct squared error-rate rule:
`base_scale = clamp(gain * (abs(actual - base) /
max(actual, base, 500))^2, 16, 96)`. It leaves the residual rate at 0.8.
This is a separate counterfactual from the 10% hinge with a four-sample
same-sign gate. All rows below use the same 4073 prediction keys in A's first
30 minutes (2244 MI / 1829 RTX); the fixed-scale-16 control has MAE
9841.4/1486.5 ms. The worst RTX block is the largest five-minute per-node
MAE increase among blocks with at least 30 predictions.

| Rule | MI MAE (ms) | RTX MAE (ms) | Worst RTX block change (ms) |
| --- | ---: | ---: | ---: |
| Fixed scale 16 | 9841.4 | 1486.5 | 0 |
| Hinge, streak 4, gain 512 | 7500.4 | 1489.6 | +33.4 |
| Squared, gain 128 | 8353.9 | 1500.0 | +84.7 |
| Squared, gain 256 | 7645.6 | 1505.9 | +136.5 |
| Squared, gain 512 | 7219.5 | 1510.8 | +197.7 |
| Squared, gain 1024 | 6980.4 | 1513.3 | +215.6 |
| Squared, gain 2048 | 6951.3 | 1514.5 | +231.4 |

The squared rule responds more strongly to MI's large errors, but every tested
gain worsens RTX more than the gated hinge, including the low-gain variant
that improves MI less. This is a single historical transition with fixed
features and no scheduling feedback from changed predictions; it does not
establish the online tradeoff. The replay used
`production_error_rate:16,GAIN,96,0.8,500,2` on
`/tmp/navi-A-first-prediction-anchor-20261008`; output traces are
`/tmp/navi-A-error-power2-gain{128,256,512,1024,2048}-20261008.tsv`.

## Counterfactual against the master's current default controller

A read-only master config query on 2026-10-08 returned the `adaptive` mode,
fixed-rate fallback scale 6, residual fallback rate 0.03, sample weight 4,
and `latency_validation_enabled=false`. The previous fixed-16/residual-0.8
comparison is therefore **not** a comparison against the current master
setting. Agent effective configs were checked separately on MI node
`11.225.196.216` and RTX node `215.2.192.71`: both initialized with
`batch_timeout_ms=150` and `latency_validation_enabled=true`, unlike the
master's 10 and false. Replaying
the current controller settings from the same historical model/optimizer
anchors is a counterfactual; the current online model state may differ.

The table compares `production` with
`production_error_rate_hinge:0.25,512,192,0.8,0.1,500,4`, using the same
prediction keys per window. The candidate uses a four-update same-sign gate,
base scale 0.25-192, and fixed residual rate 0.8. A contains a changing-
pressure opening followed mostly by sustained high pressure; B is a short
changing-pressure window; C is mostly steady. MAE and worst five-minute
per-node changes are milliseconds; positive changes mean the candidate is
worse. The worst-block calculation requires at least 30 predictions.

| Window | MI / RTX predictions | Default MI / RTX MAE | Candidate MI / RTX MAE | Worst MI / RTX block change |
| --- | ---: | ---: | ---: | ---: |
| A | 29301 / 17943 | 2039.6 / 1575.6 | 1681.4 / 1565.0 | +3195 / +417 |
| B | 2873 / 2423 | 1616.7 / 1553.0 | 1239.2 / 1583.1 | +10 / +297 |
| C | 10195 / 5501 | 1367.4 / 1704.1 | 1123.5 / 1660.9 | +58 / +388 |

Paired two-minute node-time-block resampling of those exact prediction keys
gives the following candidate-minus-default MAE changes. A negative value
favors the candidate. The 95% intervals describe variation across blocks of
these same deployments, **not** independent hardware or online uncertainty.

| Window | MI change / 95% interval (ms) | RTX change / 95% interval (ms) | MI / RTX blocks |
| --- | ---: | ---: | ---: |
| A | -358 / [-422, -295] | -11 / [-42, +20] | 323 / 335 |
| B | -378 / [-522, -254] | +30 / [-51, +117] | 35 / 47 |
| C | -244 / [-310, -183] | -43 / [-97, +10] | 98 / 104 |

MI improves consistently in this paired historical comparison, while none
of the RTX intervals excludes zero. This aggregate uncertainty does not
remove the 297-417 ms worst RTX five-minute regressions in the table above.

In A's first 30 minutes, the default/candidate MI MAE is 6784.7/6958.3 ms,
and RTX is 1557.3/1451.0 ms. Raising the candidate's minimum base scale
to 8 changes those first-30-minute values to MI 6538.1 and RTX 1474.0 ms,
but its later A RTX MAE rises from the default's 1577.6 to 1613.1 ms.
Across full A/B/C, that minimum-8 variant changes RTX MAE relative to the
default by +23.3/+65.5/-7.9 ms. Choosing solely by the first transition
would therefore hide later or cross-window regressions.

The current default controller's A first-30-minute base-scale median/P90 is
76/161 on MI and 0.25/11 on RTX. Its residual-rate median is 1.0 on MI and
0.03 on RTX. This explains why a cap-96 error-rate candidate can lag the
current default on MI despite beating fixed scale 16, but does not identify
the causal contribution of base versus residual adaptation. Fixed residual
0.03 with the gate gives MI/RTX MAE 11369.8/2045.3 ms in the first 30
minutes, versus 7500.4/1489.6 ms at fixed residual 0.8 and base minimum 16.
The controller and residual rate must therefore be compared together.

All windows reuse historical model states and samples from the same
deployments. They do not include scheduling feedback from the altered
predictions and do not establish online benefit. The large local RTX
regressions rule out declaring this variant an acceptable default on these
traces. Traces: `/tmp/navi-{A,B,C}-production-default*-20261008.tsv` and
`/tmp/navi-{A,B,C}-hinge4-min025-cap192-g512-r08-20261008.tsv`.

The worst A RTX block for the fixed-residual candidate is node
`215.2.192.71` at 00:00-00:05 Asia/Shanghai on 2026-10-08 (127
predictions). Default/candidate base MAE is 1431.6/1439.3 ms and both base
scales average 0.25, but final MAE is 1420.4/1837.7 ms. The default residual
rate averages 0.03 there; the candidate forces 0.8. This identifies the
residual-rate intervention, rather than base learning, as the immediate
source of that block's extra final error.

### Fixed residual-rate sensitivity

The same four-sample signed-error hinge was replayed with fixed residual
rates 0.2, 0.5, and 0.8. All candidates use base minimum/gain/maximum
0.25/512/192, threshold 10%, and normalizer floor 500 ms. Each row compares
identical prediction keys with the current adaptive-controller counterfactual.
MAE is in ms; the worst five-minute value is the largest candidate-minus-
adaptive MAE change on a node block with at least 30 predictions. A positive
change means that block got worse.

| Window | Hardware | Adaptive MAE | Rate 0.2 MAE / worst block | Rate 0.5 MAE / worst block | Rate 0.8 MAE / worst block |
| --- | --- | ---: | ---: | ---: | ---: |
| A | MI308 | 2039.6 | 2145.3 / +7036 | 1807.7 / +5382 | 1681.4 / +3195 |
| A | RTX5000 | 1575.6 | 1566.8 / +685 | 1564.0 / +336 | 1565.0 / +417 |
| B | MI308 | 1616.7 | 1560.6 / +546 | 1313.8 / +83 | 1239.2 / +10 |
| B | RTX5000 | 1553.0 | 1563.2 / +174 | 1578.2 / +188 | 1583.1 / +297 |
| C | MI308 | 1367.4 | 1480.8 / +582 | 1186.8 / +202 | 1123.5 / +58 |
| C | RTX5000 | 1704.1 | 1692.1 / +243 | 1667.3 / +258 | 1660.9 / +388 |

The A/B/C comparisons contain 29301/2873/10195 MI predictions and
17943/2423/5501 RTX predictions respectively. A smaller fixed residual
rate does not produce a safe compromise: it weakens MI's aggregate gain and
still leaves large local regressions on RTX and, in A, MI. This is a
historical counterfactual, not a fresh online result. The new traces are
`/tmp/navi-{A,B,C}-hinge4-min025-cap192-g512-r{02,05}-20261008.tsv`.

A temporary offline variant kept the original adaptive residual decision and
used the error-rate hinge only for base learning. It was removed from the
production source after the negative cross-window replay below; the trace
files remain for audit. The table uses all common prediction keys per window.
The hybrid has base minimum 0.25, cap 192, 10% threshold, four consecutive
same-sign errors, and gain 512. The last column is the worst RTX five-minute
node-block MAE increase versus the default controller.

| Window | Default MI / RTX MAE (ms) | Hybrid MI / RTX MAE (ms) | Worst RTX block change (ms) |
| --- | ---: | ---: | ---: |
| A | 2039.6 / 1575.6 | 2041.8 / 1567.5 | +47 |
| B | 1616.7 / 1553.0 | 1660.6 / 1553.0 | 0 |
| C | 1367.4 / 1704.1 | 1649.3 / 1704.9 | +57 |

In A's first 30 minutes, the hybrid's MI/RTX MAE is 6681.0/1547.9 ms,
versus the default's 6784.7/1557.3 ms, but it loses this MI advantage later
and in B/C. Increasing its base gain to 1024 changes full-window MI MAE to
2018.4/1724.2/1650.0 ms in A/B/C and leaves RTX near the default; it does
not solve the cross-window MI regression. Raising the minimum base scale to
8 produces a first-30-minute MI benefit but over 590 ms of local RTX
regression. A separate direct final-error-rate residual rule with gains 4/8
also fails in A: first-30-minute RTX MAE is 1719.5/1674.9 ms and the worst
RTX block worsens by 896/531 ms. Neither simple replacement retains both the
MI improvement and local RTX stability. Temporary traces are under
`/tmp/navi-{A,B,C}-hybrid-min025-*20261008.tsv` and
`/tmp/navi-A-both-error-rates-rgain{4,8}-20261008.tsv`.

Using the **persistent base** error for both rates was also tested and
removed after a negative A replay. In the default-controller trace, base
errors above 10% with at least four consecutive same-sign samples occur on
21%/29% of MI B/C predictions but almost never on RTX; this looked like a
potential hardware-independent trigger. The tested dual rule used that
four-sample gate, base minimum/gain/cap 0.25/512/192, and residual rate
`clamp(0.03 + residual_gain * max(0, base_error_ratio - 0.1), 0.03, 1)`
after the gate. Yet A first-30-minute MI/RTX MAE is 9555.5/1796.8 ms at
residual gain 4 and 8766.4/1867.0 ms at gain 8, versus default
6784.7/1557.3 ms. The worst RTX five-minute blocks worsen by 2550/3380 ms.
Rare RTX activations can therefore still cause a large residual overshoot,
and the gated updates are too sparse to retain MI's benefit. Traces:
`/tmp/navi-A-dual-rgain{4,8}-20261008.tsv`.

The residual update's `max(100 ms, 0.1 * base_prediction)` error limit was
audited on model-state replays of the current default controller. Recomputed
residual corrections match all 5288 B and 15692 C state rows. Only 0.838%
of B MI and 0.834% of C MI updates exceed that limit (RTX: 0.703% and
1.164%). Mean absolute pre-update innovations are about 1321/1056 ms on
B/C MI, versus mean limits 6129/6265 ms. Thus the configured per-update
error clipping is not the main explanation for B/C's low adaptive residual
rates or their cross-window MI gap. This audit does not test a fresh online
trajectory or rule out clipping in an unmeasured pressure transition.
Traces: `/tmp/navi-{B,C}-production-default-modelstate-20261008.tsv`.

## Error-rate multiplier on the existing adaptive controller

An additional counterfactual kept the current adaptive base and residual
decisions, then multiplied only the base learning-rate scale by
`min(MAX, 1 + GAIN * max(0, base_error_ratio - 0.1)^2)`. This is different
from replacing the controller with a fixed residual rate. The ratio uses the
current fitting sample and denominator `max(actual, base_prediction, 100 ms)`.
All rows use the same A/B/C anchors and prediction keys as the default
comparison above. MAE and worst five-minute node-block changes are in ms;
positive block changes indicate regression relative to the default.

| Window | Hardware | Default MAE | Gain 16, max 4 MAE / worst block | Gain 64, max 8 MAE / worst block |
| --- | --- | ---: | ---: | ---: |
| A | MI308 | 2039.6 | 1964.7 / +509 | 1941.7 / +641 |
| A | RTX5000 | 1575.6 | 1576.3 / +166 | 1577.3 / +513 |
| B | MI308 | 1616.7 | 1617.7 / +19 | 1625.6 / +50 |
| B | RTX5000 | 1553.0 | 1553.0 / 0 | 1553.0 / 0 |
| C | MI308 | 1367.4 | 1393.1 / +273 | 1448.1 / +421 |
| C | RTX5000 | 1704.1 | 1704.1 / +0.1 | 1704.1 / +0.4 |

The multiplier yields modest MI benefit in A but loses it in B/C. The
stronger gain increases A's RTX local regression. This simple error-rate
boost is not a stable improvement across the measured pressure windows;
it was tested only in the offline replay and has not changed production
defaults. Traces: `/tmp/navi-{A,B,C}-production-errorboost-g{16,64}-20261008.tsv`.

## Signed-error counterexample

In A's rising phase, the fixed base model has mean signed prediction error
-4848 ms on MI (underprediction) but +1662 ms on RTX (overprediction). In
the worst RTX five-minute block on node 215.2.192.71, the absolute-error
candidate raises base MAE from 2177 to 2832 ms and final MAE from 1114 to
1340 ms. Its base overprediction bias rises from 1669 to 2380 ms. This
motivated a separate counterfactual where the hinge uses
`max(0, actual_ms - base_prediction_ms)` instead of absolute error, holding
the 10% threshold, gain 512, cap 96, residual rate 0.8, and **normalizer
floor 100 ms** unchanged within this table. This table is not directly paired
with the main floor-500-ms candidate above.

| Window | Hardware | Fixed MAE (ms) | Absolute-error hinge (ms) | Underprediction-only hinge (ms) |
| --- | --- | ---: | ---: | ---: |
| A | MI308 | 5931 | 4515 | 4859 |
| A | RTX5000 | 1507 | 1526 | 1527 |
| B | MI308 | 1296 | 1277 | 1299 |
| B | RTX5000 | 1679 | 1680 | 1679 |
| C | MI308 | 1130 | 1128 | 1140 |
| C | RTX5000 | 1663 | 1662 | 1662 |

The one-sided rule fails to retain MI's gain and does not fix A's RTX MAE.
Worse, A RTX final zero-point drift P95 against the fixed control rises to
61.6 ms, versus 23.1 ms with the floor-100-ms absolute-error hinge. The experimental
production-controller mode was removed after this negative replay result;
the generated counterfactual traces remain under
`/tmp/navi-error-rate-{A,B,C}-under-hinge10-g512-cap96-r08.tsv`.

## Current-sample weight interaction

The A window was replayed with the same base-scale minimum 16 and residual
rate 0.8 while changing only the weight of the current fitting sample.
The logged weight-4 fixed baseline reproduces online predictions within
0.000001 ms. Candidate comparisons use the same 4474 MI and 3110 RTX
predictions. The optional error-rate hinge keeps threshold 10%, gain 512,
cap 96, and normalizer floor 500 ms.

| Current-sample weight | Fixed MI / RTX MAE (ms) | Hinge MI / RTX MAE (ms) |
| ---: | ---: | ---: |
| 1 | 5886 / 1507 | 5702 / 1508 |
| 2 | 5882 / 1507 | 5348 / 1512 |
| 4 (logged) | 5931 / 1507 | 4479 / 1524 |
| 8 | 5923 / 1515 | 4357 / 1542 |

The individually better fixed-rate weight 2 loses most of the hinge's MI
benefit. Weight 8 gives another 123 ms MI improvement relative to hinge at
weight 4 but adds 18 ms RTX error. Its MI final zero-point drift P95 is 8.3
ms versus 8.5 ms at weight 4, so zero-point stability is not the reason to
reject it in this window. The interaction is large enough that a single-factor
weight ranking cannot determine the combined candidate. None of these
combinations has passed the requested online MI/RTX multi-cycle check.
Using the same actual-ingress phase rule as above, weight 4 versus 8 gives
MI rising MAE 7208 versus 7117 ms and falling MAE 1627 versus 1454 ms, but
RTX rising MAE 1693 versus 1718 ms and positive-error P95 4660 versus
4893 ms. Weight 8 therefore improves MI's phase response at the cost of
RTX's local overshoot; the aggregate MAE alone is not an adequate selector.

## Residual-rate interaction

At weight 4 and fixed base scale 16, the A-window residual-rate ablation is:

| Residual learning rate | Fixed MI / RTX MAE (ms) |
| ---: | ---: |
| 0.2 | 7153 / 1677 |
| 0.5 | 6336 / 1528 |
| 0.8 (logged) | 5931 / 1507 |
| 1.0 | 5710 / 1528 |

Increasing the residual rate from 0.8 to 1.0 improves MI by 221 ms but
raises RTX MAE by 21 ms. Combined with the 10%/512/cap-96/floor-500-ms hinge,
rate 0.8 gives 4479/1524 ms MI/RTX; rate 1.0 gives 4308/1548 ms. The
combined rate-1.0 counterfactual has final zero-point drift P95 7.0/3.5 ms
MI/RTX against the logged baseline, versus 8.5/7.0 ms at rate 0.8. The
tradeoff is RTX prediction error, not zero-point drift in this window. This
supports keeping 0.8 as the current offline comparison rate, not declaring
it globally optimal.

## Transition check

For A, a rising segment requires at least eight samples in each adjacent
30-second interval and an increase of at least four in mean waiting count.
The hinge with cap 192 improves overall MI MAE by 1830 ms, but 799 samples with
insufficient nearby observations account for almost all of that improvement.
On the 1703 samples labeled rising by this rule it worsens MAE from 1582 to
1612 ms. This label is biased toward batches completing frequently: lowering
the per-window minimum from eight to two labels 2184 MI samples rising and
reverses the measured effect. The excluded samples have median waiting count
around 190 and median observed latency around 63-70 seconds. This completion-
sample rule is not adequate evidence about high-pressure transitions.

An independent phase check uses the master's logged replay target-QPS updates.
For each prediction, admission time is approximated as completion event time
minus observed admission-to-prefill-complete latency. A 30-second QPS increase
of at least 10 labels rising; a decrease of at least 10 labels falling. The
master log contains 2433 QPS updates in A, spanning 20-400 QPS. This labels
2539 MI predictions rising, 1799 falling, and 136 steady, without excluding
slow completions. On the rising predictions, fixed / hinge-cap-192 /
hinge-cap-32 MAE is 9223 / 5954 / 8503 ms. The P95 of positive prediction
error is 36497 / 12846 / 25638 ms. For the 1780 RTX rising predictions,
MAE is 1599 / 1633 / 1603 ms. A 60-second QPS-change window gives the same
direction: MI rising MAE 8962 / 5915 / 8257 ms, RTX rising MAE
1560 / 1595 / 1564 ms. Thus the prior claim that high-cap hinge worsens the
whole rising phase was incorrect; it applied only to a density-selected
subset. Target QPS alone is not proof of actual ingress QPS; the metric check
below provides that cross-check, while an online candidate run remains open.

Actual master ingress QPS was checked separately against the A target-QPS log:
2393 paired one-second observations have correlation 0.999 and mean absolute
difference 3.8 QPS, including low, middle, and high pressure. Using a causal
five-second average of the actual ingress metric instead of target QPS gives
2367 MI rising predictions. Fixed / hinge-cap-192 / hinge-cap-32 MAE is
9926 / 6416 / 9127 ms; P95 positive prediction error is
38523 / 13157 / 31126 ms. For 1662 RTX rising predictions MAE is
1670 / 1708 / 1675 ms. This corroborates the direction of the target-QPS
phase result, but the high-cap candidate still worsens RTX and the MI falling
segment slightly. Nine five-minute blocks of A have usable samples; the
high-cap MI MAE decreases in eight and increases slightly in one. The worst
RTX block increases MAE by about 152 ms. B's actual ingress spans 19-403 QPS:
high-cap MI rising/falling MAE changes from 1342/1264 to 1332/1235 ms,
while RTX changes from 1656/1700 to 1657/1701 ms. C's metric query returned
only 21 downsampled points over 207 minutes, so it cannot establish dynamic
phase performance.

## Implementation and remaining evidence

`LatencyLearningRateController` now supports an optional `error_rate_hinge`
mode. Its default `adaptive` mode is unchanged. The production-controller
replay matches the independent hinge implementation exactly for all 7584 A,
3324 B, and 15696 C predictions. The previous linear mode also reproduces
all 15696 saved C predictions exactly. Controller and curl configuration tests
pass; `navi_sched_role_test` passes 482/482.

On 2026-10-08, a read-only check of the live MI node 11.225.196.216 found
7770 training-update records but zero `fit_inflight_before_requests` records
in its current log; the latest training update was at 05:30 local time. The master
reported an active fixed-QPS replay with 18 eligible of 20 connected clients;
this check did not change the active replay. A later read-only check found
the master replay stopped; no new training update occurred on the sampled
RTX node after its restart. The fit-time inflight diagnostic is emitted only
while validation is enabled and a fit update occurs, so the absence of recent
lines cannot establish that the running image lacks the diagnostic.
Consequently the true fit-time inflight bucket experiment and an online
multi-cycle traffic test remain unverified. No candidate here is established
as a new default or as an online precision improvement.

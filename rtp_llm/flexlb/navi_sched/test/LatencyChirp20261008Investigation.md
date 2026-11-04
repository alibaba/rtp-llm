# 2026-10-08 variable-frequency replay investigation

## Same-batch Master versus Agent prediction audit

- Paired `schedule_batch_id` in the Master admission log with Agent training samples from the pre-512 MI run (MI cap 1024, RTX cap 64): 3,711 batches on MI `11.225.195.118`, 1,830 on RTX `215.8.64.161`. Master `final_execution_ms` equals the prediction carried into the matching Agent sample exactly. Source excerpts: `/tmp/navi-chirp-20261008-master-mi118-admission.log.gz`, `/tmp/navi-chirp-20261008-master-rtx161-admission.log.gz`, and the corresponding Agent files in `/tmp/navi-chirp-20261008-45s-full-exact/`. This is a two-node audit, not yet an all-node conclusion.
- The dashboard's "Agent before update" prediction is calculated when a Batch **finishes**, before fitting that Batch, using its actual cache-hit sample and the model as of completion. It is not an Agent-at-admission prediction. Reconstructing Agent revision at Master schedule time from training-update event times shows Master had that revision on `99.6%` of MI and `99.9%` of RTX matched batches. The mean time from schedule to finish was about `56 s` on MI and `55 s` on RTX; the Agent model advanced another `11.1` and `5.6` revisions respectively during that interval. A one-revision timing ambiguity remains because the update log carries sample event time rather than the exact publication instant.

| Matched node | Master final MAE ms | Agent-at-finish pre-fit MAE ms | Master base MAE ms | Agent-at-finish base MAE ms | Mean absolute residual-correction change ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| MI `11.225.195.118` | 6759 | 1638 | 5234 | 4333 | 5610 |
| RTX `215.8.64.161` | 3187 | 1844 | 6495 | 6036 | 3792 |

- On MI, `90.1%` of the paired batches have identical main workload features on both ends (all count/token/square/product features; excluding elapsed-running time), and `97.8%` have identical predicted and actual cache-hit token totals. Within the 3,345 MI batches whose main features match exactly, Master/Agent final MAE is `6710/1629 ms`. Thus cache-hit and the main logged workload-feature mismatch do not explain most of this gap. The residual correction changes much more than the base prediction; this supports the user's model-update hypothesis, while the separately advancing elapsed-time feature and possible scheduling feedback still need an exact counterfactual decomposition.
- On that MI node, among 210 batches taking at least 80 seconds from Master schedule to Agent finish, Master underpredicts **every** batch: mean signed error is `-28,805 ms`, Master WAPE is `30.5%`, and Agent-at-finish pre-fit WAPE is `1.7%`. The mean final-prediction gap grows with batch lifetime and number of intervening model updates. These are selected long-lived batches, not an unbiased estimate of total traffic; the 512-slot run must test whether their frequency and error actually fall.
- vLLM's inspected scheduler handles `running` before `waiting`; under the default FCFS branch KV exhaustion preempts from the tail of running, while priority mode can choose a lower-priority running request. Later requests can still share token budget/execute steps with earlier work and change its completion time. Do not infer the deployed policy or the frequency of preemption from source inspection alone. A new 512-slot MI run is in progress with the same 0-400 QPS chirp; compare like-for-like pressure phases and logs after it completes before claiming the cap change helps.

## Interim 512-slot comparison (not a final cap ablation)

- The 512-slot run started near 14:41:01 CST. The replay driver was still in its frequency-up sweep when these logs were collected. Current logs are in `/tmp/navi-chirp-20261008-512-all-interim/`; the old 1024-slot logs are in `/tmp/navi-chirp-20261008-45s-full-exact/`. Both manifests list the same 20 node IPs and the same vLLM image revision `08127ab61e`. The comparison below uses only the first 15 minutes after each run's 15-minute warmup, assigning each sample by `event_time_us - target_ms * 1000`. This is an inferred Agent-ready time, not measured ingress time.

| Hardware/run | vLLM max sequences | Samples | Master MAE (ms) | Master WAPE | Mean true latency (ms) | Mean preceding requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MI, previous run | 1024 | 6825 | 6691 | 12.2% | 54726 | 1194 |
| MI, current run | 512 | 6345 | 5469 | 10.0% | 54828 | 667 |
| RTX, previous run | 64 | 3558 | 2733 | 4.8% | 56556 | 164 |
| RTX, current run | 64 | 3684 | 2344 | 4.0% | 57891 | 168 |

- MAE is mean absolute `master_prediction_ms - target_ms`. WAPE is the sum of absolute errors divided by the sum of true latencies. Seven of ten MI nodes improve, while three worsen by 7.7%, 10.2%, and 16.6%; RTX also improves without a sequence-cap change. Both replay drivers initially reported 17 eligible clients, but membership later in each run has not been fully compared. The MI models were restarted while RTX models were not (mean logged local revision in this window: MI 1319 -> 420, RTX 888 -> 1448). Thus the difference cannot be assigned solely to the MI cap. At inferred target QPS below 50, MI WAPE is 16.8% -> 13.2%; at target QPS 300-400, 11.2% -> 8.9%. These QPS bins use the driver's commanded wave at inferred ready time, not measured ingress QPS.
- Counting IDs in each completed sample's `request_ids` list gives 30,078 -> 27,013 MI requests in this window (about 10% fewer), versus 17,114 -> 17,307 RTX requests. This measures requests represented by the scored Agent batches, not Master ingress QPS; it still shows the MI cohorts did not receive equal scored workloads. The sample's `batch_size` is an aggregate latency-model feature and must not be interpreted as the arriving batch's request count. A controlled cap comparison needs matching delivered request distributions and model state, not just the same replay target waveform.
- The logged `backend_queue_ms` is measured from Agent batch-ready time to its first vLLM execution. Over this same window, MI mean target/queue time is 54,726/53,089 ms at 1024 and 54,828/53,319 ms at 512: about 97% of the observed target in both runs is backend waiting, while target minus queue time is only 1637/1509 ms. RTX queue fractions are 96.2%/96.4%. Thus slower shared GPU steps are a real mechanism but cannot alone explain several-second total-latency prediction errors; the queueing/admission dynamics and per-node request distribution require separate audit. For the 947 MI `11.225.195.118` batches admitted with all 512 slots full, mean target/first-execution wait/execution wall times are 55.1/53.6/1.5 s.
- In the current vLLM adapter, `_make_workload_contribution()` sets `remaining_total_tokens` equal to `remaining_prefill_tokens`; it does not estimate remaining decode work. The unified latency formula consumes total preceding request count, prefill running/waiting features, and running-prefill elapsed time, but no distinct decode progress or remaining-output feature. This is a plausible source of queue-time uncertainty under a saturated running-slot limit, not yet a demonstrated corrective feature.
- As a small negative-control experiment, three MI nodes' 512-slot samples were matched to vLLM execution records by local batch ID. A chronological ridge correction trained on sweep minutes 0-10 and evaluated on minutes 10-20 used the online Master prediction plus existing preceding/running-prefill/waiting-prefill counts. Its held-out MAE on `11.225.195.118`, `11.225.202.174`, and `11.225.195.187` was 4229/3069/3342 ms, versus the uncorrected 5196/4383/4060 ms. Adding the separately logged vLLM admission running/waiting counts changed held-out MAE only to 4154/3059/3294 ms (75/10/48 ms additional gain). This one-window diagnostic is not a production-model ablation, but it weakens the claim that missing running count alone explains the multi-second error; the existing total preceding count already proxies much of it.
- A separate 20-node ridge diagnostic compared a correction using the online prediction and existing count features with additional `preceding_count^2` and `preceding_count * waiting_prefill_count` terms. Training on sweep minutes 0-10 and testing minutes 10-20, the interaction reduced MI MAE by 34 ms in the 1024 run and 144 ms in the 512 run; reversing the time windows reduced MI MAE by 74/152 ms. RTX gains were small forward and reversed-window RTX MAE worsened. This is not an online formula ablation and does not explain the several-second MI errors; do not enable it from these results alone.
- On MI `11.225.195.118`, 2169 training samples were matched by local batch ID to vLLM execution logs. At admission, 947 had all 512 running slots occupied; these had mean 160 waiting vLLM requests, mean true latency 55.1 s, and Master MAE 6.0 s. In the 2169 matched prefill execution-start records, `start_preempted_requests` and `start_running_recompute_requests` were both zero. This is evidence against preemption as the main cause in this sampled interval, not proof that no decode-only step ever preempted.
- For single-step prefill batches on that node with 100-199 prefill tokens and 2000-9999 total prompt tokens, median `execute_model_sum_ms` rose from 146.3 ms (128-255 concurrent decode requests, 27 batches), to 212.9 ms (256-383, 35 batches), to 370.4 ms (384-511, 125 batches). Prompt-size matching is coarse and these are observational samples; they support concurrent-decode contention but do not identify whether each decode request arrived before or after the measured batch. The vLLM scheduler prioritizes running requests, then admits waiting requests subject to token/KV budgets and `max_num_seqs`; the 512 cap limits new admissions into this shared execution set. NaviSched's formula includes total preceding request count as a proxy for occupancy, but not the contemporaneous decode-running count as a separate feature.
- The same MI node's `/tmp/navi-chirp-20261008-512-interim/MI_11.225.195.118.step.log.gz` contains 5997 vLLM execution-step records: 5996 have both prefill and decode work, and 538 include multiple NaviSched prefill batches. One step includes batch IDs 2-108 (107 prefill requests). Batch IDs are assigned in admission order, so later prefill batches did share a step with earlier ones. The step log counts decode requests but does not list their IDs or arrival order; it cannot by itself prove that each concurrent decode request arrived later.
- The current Master `scheduler-config --get` reports base `batch_timeout_ms=10`, while two restarted MI Agents (`11.225.195.118` and `215.77.72.161`) logged startup `batch_timeout_ms=150` and no later `applied` record. Master base configuration must not be mistaken for the Agent's effective setting. Before a controlled repeat, explicitly curl the intended complete parameter set (including batch timeout), verify every target Agent's `applied` log, and hold parameters unchanged throughout the run.

## Full 20-node sweep: production-switch ablation

- The later diagnostic re-anchor removes the continuous replay drift on the two previously excluded MI nodes for this sweep; the underlying pre-sweep drift cause remains unresolved. A single anchor immediately before the sweep now reproduces all 41,050 online final predictions (20 nodes); the maximum per-prediction replay difference is below `0.000001 ms`. The older 18-node exclusions and scores below describe the earlier, unanchored replay and must not be used as the current comparison basis.
- Score window: `1791431462000000` to `1791435062000000` microseconds, assigned by `event_time_us - target_ms * 1000`. Warmup trains the model but is not scored. Source logs are `/tmp/navi-chirp-20261008-45s-full-exact/`; replay inputs are `/tmp/navi-chirp-20261008-45s-full-jsonl/`. Baseline/config-switch results are `/tmp/navi-chirp-20261008-production-switch-preserve-grid-all20.json` and `/tmp/navi-chirp-20261008-production-switch-preserve-weight3-all20.json`.
- Changing only the adaptive base/residual rate ceilings previously reset the learning-rate controller. A no-op reset at the sweep boundary raises MI MAE from `1522` to `1646 ms`; this confounds a rate-limit comparison. The current local `configure()` change retains learned statistics for ceiling-only updates and immediately clamps the effective rates. Other controller changes still reset. Its focused C++ test passes. This behavior is **not deployed**.

| Current-sample weight / base cap / residual cap | MI MAE ms | MI second-half MAE ms | MI positive-error P95 ms | RTX MAE ms | RTX second-half MAE ms | RTX positive-error P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 / 192 / 1.0 (online-equivalent baseline) | 1522 | 1641 | 4513 | 1969 | 1916 | 4991 |
| 2 / 96 / 1.0 | 1488 | 1528 | 4348 | 2012 | 1964 | 5141 |
| 3 / 96 / 1.0 | 1510 | 1588 | 4400 | 1992 | 1945 | 5090 |
| 6 / 96 / 1.0 | 1536 | 1662 | 4558 | 1928 | 1870 | 4866 |

- MAE is mean absolute final-prediction error; positive-error P95 is the 95th percentile among predictions above truth, so lower is better for overshoot. Weight `2` improves MI mean error by `34 ms` (`2.3%`) but worsens RTX by `43 ms` (`2.2%`); one RTX node worsens by `8.2%`. Weight `3` has a smaller MI gain and still worsens one RTX node by `4.4%`. Weight `6` improves RTX by `41 ms` (`2.1%`) but worsens MI by `14 ms` (`0.9%`). None meets the cross-hardware criterion on this sweep. These are offline counterfactuals on fixed logged traffic, not online scheduling outcomes.
- Fit-time live inflight is currently logged around `recordSample()` only. Production history buckets are keyed by completed latency; the offline pressure-bucket experiment alone uses fit-time request count to choose extra historical samples. On its earlier 18-node scoring subset, the two-extra-samples variant changed MI MAE `1561.8 -> 1574.0 ms` and RTX `1969.2 -> 1929.5 ms`, with one MI node worsening by about `107 ms`; it is not a demonstrated general improvement and has not been enabled online.
- A second 20-node grid also switched the residual update-error ratio at the sweep boundary (results: `/tmp/navi-chirp-20261008-production-switch-preserve-clip-grid-all20.json`). This ratio clips each residual *update error* to at most `max(100 ms, ratio * base prediction)`; it does not clip the final prediction. At weight/base/residual cap `4/192/1.0`, raising the ratio from `0.1` to `0.4` changes MI MAE `1522 -> 1513 ms`, MI positive-error P95 `4513 -> 4445 ms`, RTX MAE `1969 -> 1969 ms` (unrounded RTX improves by less than 1 ms), and RTX positive-error P95 `4991 -> 4983 ms`. The worst per-node MAE changes are `+0.7%` on MI and `+0.4%` on RTX. At weight `2`, ratio `0.4` improves MI MAE to `1467 ms` but still raises RTX to `2012 ms`, so clipping does not remove the cross-hardware tradeoff. This is a small offline improvement, not a solution to MI lag or an online validation.

## Pending: degradation rate during node loss

- Observation (reported by user): around 11:08-11:10 CST, one node dropped out and the instantaneous degradation rate also fell.
- Do not infer that losing capacity improved scheduling until the event is replayed from logs.
- Align the node disconnect/reconnect events, active scheduler node count, ingress and scheduled QPS, per-node inflight/waiting counts, and the degradation numerator and denominator at one-second resolution. Check whether the node was unusually inaccurate before dropping, whether load shifted to another deployment, and whether missing reports changed the metric denominator.
- Compare the same interval immediately before and after the drop, including request-level decisions and actual latencies. Account for the changing chirp target QPS and its phase before attributing the rate change to node loss.

## Pending: high error on newly joined node

- Observation (reported by user): `215.77.72.161` joined around 11:28 CST and initially had a very large prediction error.
- Compare its first request and training sample times, restored model parameters and optimizer state, revision, ramp/trusted state, warmup/low-pressure sample coverage, and workload features with peers in the same interval.
- Recalculate its request-level prediction and error before and after joining; distinguish a bad initial model from workload skew or stale metrics. Do not mark this resolved from a dashboard alone.

## Pending: error spike after replay restart

- Observation (reported by user): model error became very large after traffic was restarted.
- Test the residual-expiration hypothesis per node. Record the last residual update before the gap, the first post-restart sample and its `residual_decay_factor_before`, base/final predictions, and actual latency. The current implementation begins idle decay after 2 minutes and reaches zero after 5 minutes.
- Calculate counterfactual first-batch errors with the logged residual kept at full strength versus the actual decayed correction; compare with nodes whose residual did not expire. Also check changed client count, request mix, pressure, cold-start/ramp, and model revision before deciding whether the decay window is too short.
- Checked all 20 current nodes in the first five minutes of the replacement replay (data: `/tmp/navi-chirp-restart-audit-all/`). Each completed sample's ready time is reconstructed as `event_time_us - target_ms * 1000`, matching the unified target definition and `time_compression=1`; none were ready before the old replay stopped. The residual decay factor was below one for only 29 MI and 21 RTX samples, with none below 0.5; the no-decay counterfactual changed their errors by at most hundreds of milliseconds, not the multi-second MI spike. This rejects idle residual decay as the main cause of this restart spike, but does not prove the configured decay window is optimal in other idle scenarios.
- Grouped by inferred ready-time minute after the new replay start, MI final MAE was about 0.039 / 16.62 / 4.34 / 18.84 / 4.79 seconds; its signed prediction-minus-truth bias was about -0.011 / -14.63 / +3.01 / +17.55 / +1.39 seconds. RTX final MAE was about 0.195 / 2.05 / 2.06 / 1.98 / 2.11 seconds. MI waiting-batch-size mean reached about 275 in minute 2. The sign flip from underprediction during rising load to overprediction on falling load points to pressure-following lag/overshoot as the primary effect to investigate; compare parameter update traces before declaring a specific algorithmic cause.

## Current replay evidence

- Driver: 20-400 QPS, 15-minute 5-minute-period warmup from 10:32:10 CST, then a continuous 60-minute 5-minute -> 30-second -> 5-minute period sweep.
- The in-flight run above retains its 30-second minimum. Future runs use a 45-second minimum as requested; editing the script does not change the already-running process.
- Twenty node training logs and exact history append/fit logs were collected under `/tmp/navi-chirp-20261008-1038-exact/`; no collection failures. The pressure test was not stopped or changed during collection.
- Exact history selection exported to `/tmp/navi-chirp-20261008-1038-jsonl/`. The first export omitted pre-update Adam and residual state, causing the replay to skip the first fit and diverge by about 510 ms MAE. After adding those state fields, the 10:33-10:43 baseline matches 7,671 online predictions with 0.0105 ms MAE; local counterfactual ablation can use this baseline.
- The first run was interrupted around 11:34 CST for the requested replacement. Keep its 20-400 QPS / 30-second-minimum data separate from the new run.
- The replacement began around 11:36:02 CST with 17 eligible replay clients: 0-400 QPS, 15-minute warmup, 60-minute 5-minute -> 45-second -> 5-minute sweep, 60-second near-zero drain, then 100 QPS and 50 QPS for five minutes each. The deployed endpoint rejects exact zero, so the driver sends 0.000001 QPS during the target-zero drain without stopping the replay.

## Exact-replay residual ablation on the first run

- Source: all 20 agent logs from the first run, replayed from their exact online history selections. In the window whose inferred batch-ready time is 10:33:00-10:43:00 CST, there are 4,683 MI and 3,341 RTX predictions. Ready time is `event_time_us - target_ms * 1000`; this window is not the same as filtering by completion time. Baseline replay versus logged online final prediction has MAE 0.02082 ms on MI and below the displayed precision on RTX.
- MAE below is milliseconds, with prediction minus truth as the signed bias. `cap` changes the adaptive residual-rate ceiling from 1.0; `clip` changes the per-sample error clipping ratio from 0.1. Each variant replays the same history and is an offline counterfactual, not a deployed parameter setting.

| Variant | MI MAE | MI bias | RTX MAE | RTX bias |
| --- | ---: | ---: | ---: | ---: |
| Online-equivalent baseline | 3023.6 | +1088.1 | 2043.1 | +472.5 |
| Residual cap 0.3 | 4267.5 | +1094.4 | 2323.9 | +453.3 |
| Residual cap 0.6 | 3333.6 | +1100.5 | 2118.8 | +446.8 |
| Error clip 0.2 | 2886.3 | +1014.6 | 2036.9 | +477.5 |
| Error clip 0.4 | 2791.5 | +974.2 | 2039.4 | +480.9 |
| Error clip 0.8 | 2781.7 | +1004.3 | 2039.7 | +483.1 |

- Paired by host, sequence and event time, clip 0.4 reduces MAE on all 10 MI nodes; among 10 RTX nodes, it improves 6 and worsens 4. The largest RTX per-node worsening is about 45.6 ms; the largest MI improvement is about 990.3 ms. Thus aggregate RTX MAE alone hides a small but real cross-node regression risk.
- For a phase check, estimate the first-run target QPS as `210 - 190*cos(2*pi*t/300)` with `t` measured from 10:32:10 CST to the inferred ready time. This is the intended driver waveform, not measured ingress QPS. On MI, clip 0.4 changes high-QPS rising-phase MAE from 5022.4 to 4430.2 ms and high-QPS falling-phase MAE from 1801.8 to 1739.3 ms. On RTX, the same high-QPS rising phase worsens from 1852.6 to 1892.9 ms, while the falling phase improves from 2013.8 to 2002.5 ms. Low/mid/high here mean intended QPS below 100, 100-250, and at least 250 respectively.
- Define overshoot here as a positive final-prediction-minus-truth error, and take P95 only over those positive errors in the same ready-time window. MI's positive-error P95 is 20843.2 ms at baseline, 19486.8 at clip 0.2, 17176.2 at clip 0.4, and 17265.0 at clip 0.8. RTX's is 6793.1, 6692.5, 6747.3, and 6794.6 ms respectively. The positive-error sample counts vary slightly with the intervention. RTX's high-QPS rising-phase signed bias increases from +818.2 to +889.2 ms at clip 0.4, consistent with its MAE regression in that phase.
- The gain is time-local, not a uniform steady-state improvement. Splitting by inferred ready time, clip 0.4 changes MI MAE from 4629.6 to 4165.8 ms in 10:33-10:38, but only 1450.9 to 1445.6 ms in 10:38-10:43. RTX changes from 2254.4 to 2240.7 ms and from 1747.8 to 1757.8 ms respectively. In the separate 10:43-11:04 continuation, baseline versus clip 0.4 is MI 1296.0 versus 1292.5 ms (9703 predictions) and RTX 1971.8 versus 1972.1 ms (5599 predictions). Baseline replay still matches the online predictions there within 0.00479 ms MAE on MI and below displayed precision on RTX. This supports an early transient benefit, not a proven sustained gain during the changing-frequency sweep.
- This is evidence that the current clipping ratio can constrain MI updates, but not evidence that increasing it is universally beneficial. Recheck per-node and phase effects on the full replacement sweep and validate online before changing defaults. The old and replacement runs must not be mixed.

## Base-rate ceiling and combined candidate on the first run

- The 20-node exact replay uses current-sample weight 4 and the production adaptive controller. Re-running the default controller produces identical prediction (`P`) rows to the saved online-equivalent baseline. `production_cap` changes only the controller's base learning-rate ceiling (default 192); the combined candidate also changes residual error clip from 0.1 to 0.4. An identity parameter-rate override plus clip 0.1 produces prediction rows identical to the cap-96-only run, checking that the combined command adds no hidden intervention.
- All values are final-prediction MAE in ms on the same inferred-ready-time window, 10:33-11:04 CST. `first5` is 10:33-10:38, `last5` is 10:38-10:43, and `later21` is 10:43-11:04; these are time slices, not equal-pressure subsets.

| Candidate | MI first5 | MI last5 | MI later21 | RTX first5 | RTX last5 | RTX later21 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default base cap 192, clip 0.1 | 4629.6 | 1450.9 | 1296.0 | 2254.4 | 1747.8 | 1971.8 |
| Base cap 64, clip 0.1 | 3980.1 | 1636.1 | 1250.6 | 2221.7 | 1677.1 | 1889.6 |
| Base cap 96, clip 0.1 | 3644.4 | 1519.6 | 1211.8 | 2298.5 | 1677.6 | 1901.5 |
| Base cap 128, clip 0.1 | 3845.3 | 1463.5 | 1225.4 | 2260.9 | 1685.9 | 1909.1 |
| Base cap 384, clip 0.1 | 4868.8 | 1745.6 | 1697.5 | 2302.7 | 1736.0 | 1885.3 |
| Base cap 64, clip 0.4 | 3817.7 | 1624.2 | 1247.7 | 2216.5 | 1682.7 | 1888.9 |
| Base cap 96, clip 0.4 | 3465.1 | 1509.3 | 1211.2 | 2271.0 | 1685.7 | 1901.7 |

- The cap-96 plus clip-0.4 candidate has full-window MI MAE 1623.3 versus 1858.4 ms at baseline (14,386 predictions) and RTX 1948.5 versus 1998.5 ms (8,940 predictions). Positive-error P95 falls from 5842.5 to 5255.3 ms on MI and from 5629.7 to 5538.5 ms on RTX. However, one MI node worsens by 12.6 ms and four RTX nodes worsen, with the largest RTX regression 81.7 ms; much of RTX's aggregate gain comes from one node improving by 507.7 ms. This candidate is not yet a robust cross-node result.
- Raising the base cap to 384 worsens MI, while lowering it to 64/96/128 improves the aggregate on this first-run window. The next gate is the independent 0-400 QPS replacement sweep with a 45-second minimum period, followed by actual online forward/reverse testing. No default or live parameter was changed for these counterfactuals.
- With base cap 96 and clip 0.4 held constant, varying current-sample weight gives full-window MI/RTX MAE (ms): weight 2 = 1921.6/1937.9, weight 4 = 1623.3/1948.5, weight 6 = 1603.5/1925.4, weight 8 = 1638.2/1983.2. Weight 6 is best on this aggregate, but two MI nodes regress versus the online baseline by 33.6 and 68.7 ms, and three RTX nodes regress by up to 57.8 ms. Weight 8 improves MI's first five minutes to 3085.4 ms but worsens its later 21 minutes to 1329.5 ms, versus 3465.1 and 1211.2 ms at weight 4. These tradeoffs make weights 4 and 6 candidates for the independent sweep; selecting a final weight from this one window would overfit.

## Log-anchored event times

- Master logged `11.225.193.172:27344` offline with `agent_stopping` at 11:08:39.219 CST. The degradation-rate numerator/denominator and request decisions before and after that event remain unaligned.
- Master registered replay client `215.77.72.161` at 11:21:23 CST, marked its agent online at 11:21:48 CST, and scheduled request `16546004219183661127` at 11:21:48.868 CST with revision 1, session sample count 0, `trusted=0`, and final execution prediction 312.611 ms. The agent's first training sample for that request finished at 11:22:22.457 CST with target 33439.442 ms and pre-update final prediction 1578.982 ms. The dashboard appearance around 11:28 is not the first admission time. The reason that master and agent predictions and truth differ so much is still under investigation.

## Replacement sweep: complete run and replay integrity

- The container driver logged the near-zero drain, 100 QPS step, 50 QPS step, `steps_complete`, and `stopped`. During the steps, `navi_sched_replay query` reported 17 eligible clients and per-client QPS 5.88235294 and 2.94117647 respectively. After stop, the query returned `action=stop` and per-client QPS zero. This verifies accepted control state, not measured ingress QPS; correlate with the ingress metric separately.
- All 20 current inference nodes were collected after stop under `/tmp/navi-chirp-20261008-45s-full-exact/`, with zero collection failures. Each file includes training samples, training updates, history appends and exact history-fit selections through about 13:02:20 CST. Exact JSONL conversion completed for all nodes under `/tmp/navi-chirp-20261008-45s-full-jsonl/`. The model replay's default prediction (`P`) matches the online final prediction to displayed precision on 18 nodes, including all 10 RTX nodes and the new MI node `215.77.72.161`.
- A second read-only collection after 13:18 CST under `/tmp/navi-chirp-20261008-45s-drained-exact/` has zero failures and is byte-identical to the first after decompression for all 20 nodes. Thus the earlier snapshot was not missing late training samples after replay stop; the 50-QPS step scores below are not truncated by collection timing.
- Two MI nodes must be excluded from candidate scoring until replay integrity is repaired: `11.225.199.135` has mean/max prediction disagreement 160.4/7774.1 ms, and `11.225.203.0` has 297.0/11767.5 ms over the new run. Their last old-run five-minute windows matched to under 0.1 ms mean difference, then model/Adam state deviations grew during the new run. The agent config log shows the default adaptive config and no later config update on `11.225.203.0`; a config change is not established as the cause. The divergence may reflect a sensitive update trajectory, but its root is not yet proven. Candidate results below use only the 18 online-matched nodes; excluded nodes remain a required replay gap, not a silent pass.

## Replacement sweep: independent candidate checks

- MAE is mean absolute final prediction error, in milliseconds. The warmup is the first 15 minutes after 11:36:02 CST and the sweep is the next 60 minutes. Samples are assigned by inferred batch-ready time, `event_time_us - target_ms * 1000`; this is not a direct measurement of request ingress time. The MI cohort has 4760 warmup and 21360 sweep predictions; RTX has 4496 and 14342. All candidate variants use the same logged input samples and exact historical fit selections, so they do not include scheduling feedback from changed predictions.

| Variant | MI warmup MAE | MI sweep MAE | RTX warmup MAE | RTX sweep MAE |
| --- | ---: | ---: | ---: | ---: |
| Online-equivalent baseline | 3965.2 | 1561.8 | 1738.0 | 1969.2 |
| Base cap 96 only | 4341.9 | 1505.4 | 1705.0 | 1972.5 |
| Residual error clip 0.4 only | 3568.0 | 1543.8 | 1744.2 | 1968.5 |
| Base cap 96 + clip 0.4, weight 4 | 3787.2 | 1495.4 | 1708.8 | 1972.0 |
| Base cap 96 + clip 0.4, weight 6 | 3688.2 | 1510.3 | 1744.5 | 1944.5 |
| Base-score maximum fall 0.05, otherwise baseline | 4011.1 | 1512.4 | 1657.6 | 1936.1 |
| Base-score maximum fall 0.10, otherwise baseline | 4193.3 | 1492.5 | 1681.0 | 1948.8 |

- The cap-96-only candidate worsens MI warmup high-QPS rising MAE from 8553.6 to 9445.5 ms, while clip-0.4-only improves it to 7472.9 ms; combining them reaches 7939.1 ms. Thus the lower cap damps the later oscillating sweep but impairs initial high-pressure following, while a wider residual correction partly compensates. This is a real phase tradeoff, not a single overall percentage gain.
- Define overshoot as positive prediction-minus-truth error, with P95 computed only among positive errors in each phase. Baseline MI warmup/sweep P95 is 10563.6/4665.8 ms. Cap 96 plus clip 0.4 at weight 4 changes it to 11871.4/4443.7 ms; weight 6 changes it to 12032.7/4391.4 ms. Increasing the base-score fall limit to 0.05 changes it to 12896.7/4361.7 ms. The sweep tail improves, but warmup overshoot worsens substantially in every tested combined or faster-fall candidate.
- At weight 4, cap 96 plus clip 0.4 worsens 7 of 10 RTX nodes' sweep MAE, by up to 68.6 ms, despite only +2.8 ms aggregate RTX sweep MAE. Weight 6 improves aggregate RTX sweep MAE by 24.7 ms but still worsens 3 RTX nodes by up to 44.8 ms, and worsens MI's 100-QPS step MAE from 1671.3 to 1786.6 ms. These are screening results, not a deployment decision. Further phase, node and online forward/reverse checks are needed before changing defaults.

## Exact running-decode feature ablation

Definitions used in this section:

- P17 is the number of requests in `vllm.scheduler.running` that already have at least one output token at the admission snapshot.
- P18 is the sum of `num_output_tokens` over those requests. It is generated work, not remaining decode work. Unless every request has the same known output budget, P17 and P18 cannot uniquely determine the remaining decode work or the next slot-release time.
- Baseline masks both P17 and P18. P17-only, P18-only, and both replay the same samples, history selections, update order, model formula, residual path, and Adam state while exposing only the named inputs.
- MAE is mean absolute final-prediction error in milliseconds. Improvement is `(baseline MAE - candidate MAE) / baseline MAE`; positive is better. The ready-time window is `1791574428538055 <= t < 1791575328639018`, three intended five-minute `5 -> 100 -> 5 QPS` cycles.

The older decode-progress experiment and this exact experiment answer different questions. The older experiment reconstructed decode progress from execution lifecycle logs and fitted a separate held-out linear correction in a much higher-latency window. Its best pressure-interaction variant changed aggregate MAE only `1378.0 -> 1376.0 ms`; P17/P18 together changed it to `1422.6 ms`. The exact experiment reads P17/P18 synchronously from vLLM and puts them inside the production nonlinear formula and online Adam path. Calling both results simply "the decode ablation" hides the different feature accuracy, formula, optimizer, and pressure regime.

One MI node, `11.225.198.191`, is excluded from efficacy scoring. Its continuous replay diverges from the logged online trajectory after sub-picosecond JSON roundoff is repeatedly amplified by the high-rate nonlinear updates. Restoring the logged pre-update parameters and Adam state before every update reduces the maximum state difference on all 20 nodes to `5.68e-14`, proving the single-update formula, fit-batch order, and P17/P18 values are reproduced. It does not make the divergent continuous counterfactual reliable.

| Variant | MI samples | MI MAE baseline -> candidate | MI improvement | MI improved nodes | RTX samples | RTX MAE baseline -> candidate | RTX improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P17 only | 15,832 | 156.23 -> 132.20 | 15.4% | 6 / 9 | 6,474 | 30.29 -> 25.63 | 15.4% |
| P18 only | 15,832 | 156.23 -> 137.98 | 11.7% | 6 / 9 | 6,474 | 30.29 -> 27.53 | 9.1% |
| P17 + P18 | 15,832 | 156.23 -> 123.34 | 21.0% | 6 / 9 | 6,474 | 30.29 -> 25.56 | 15.6% |

The mean gain is real on this logged schedule but is not node-stable on MI. For P17+P18, the MI host-cluster bootstrap 95% interval for absolute MAE improvement crosses zero, while the 30-second time-block bootstrap interval is positive. The best MI node, `11.225.201.224`, improves by `59.9%`; `11.225.205.114` regresses by `51.0%`, `11.225.207.104` by `9.0%`, and `11.225.195.118` by `7.6%`. The first MI cycle regresses by `6.0%`, followed by improvements of `22.4%` and `28.2%` in cycles one and two. RTX has too few later-cycle samples for a balanced repeated-cycle conclusion: 6,239 of its 6,474 scored samples are in cycle zero.

The per-node conditional relationship is non-stationary. Among the nine MI nodes with at least two cycles containing 50 samples, correlation between P17 and the baseline residual changes sign across cycles on five nodes; P18 does the same on five nodes. Under pressure, P17/P18 also correlate strongly with existing running/waiting pressure, commonly `0.6-0.87`, so part of their mean gain is an additional pressure proxy rather than independently identified decode remaining work.

`11.225.205.114` is the clearest counterexample. It receives samples in early blocks and then has an approximately 390-second training gap before traffic returns. Its mean generated tokens per running decode request changes from roughly `88` in cycle zero to `42` in cycle two. Its P18 coefficient traverses `+7.04` to `-6.45`; over the full node window, P18 has only about `0.001` correlation with the baseline residual. This cohort change can make "few generated tokens" mean either young long-lived requests or simply short-output requests, which P17/P18 cannot distinguish.

The earlier strict lifecycle-pair audit is consistent with this explanation rather than contradictory to it. Across 12 old-feature-neighbor pairs, all slower samples have a younger decode cohort, longer retrospective remaining slot lifetime, and lower slot-release rate. Decode count is already near the 511/512 running cap in both members and differs little. P18 observes cohort age, but neither P17 nor P18 observes each request's output budget; the strongest discriminator in the retrospective audit is remaining lifetime/release rate, not raw decode count.

A 25-point P17/P18 learning-rate grid does not remove the MI counterexample. The lowest aggregate MI MAE uses multipliers `P17=1.0, P18=0.3` and improves MI by `21.9%`, but its worst MI node still regresses by `39.3%`. The widest MI coverage uses `P17=0.03, P18=0.1`, improves 8/9 MI nodes and all RTX nodes, but the remaining MI node regresses by `38.5%`. Therefore no tested P17/P18 rate is stable enough to select as a default from this window.

Current conclusion: exact P17/P18 are useful observability fields and can improve average dynamic tracking, but they are not a stable causal representation of remaining decode pressure. Keep their collection, protocol, replay, logs, and metrics. Do not select a new production default solely from the `21.0%` aggregate result. The decisive follow-up is a repeated high-pressure run with exact fields and a directly observable remaining-budget candidate such as `sum(max_tokens - num_output_tokens)` when the request contract makes that quantity meaningful; it must beat P17/P18 on held cycles and remove the per-node sign reversal before formula adoption.

### High-pressure repeat: conditional signal versus online ablation

This subsection separates three measurements that answer different questions:

- A **neighbor pair** contains two samples from the same host, separated by at least 60 seconds, whose 13 pre-existing workload features are close after per-feature normalization but whose true latencies differ by at least 2 seconds. It tests whether P17/P18 distinguish locally ambiguous old-feature samples; it does not test a trainable production model.
- A **continuous replay** starts from the same logged state and then lets each candidate follow its own online Adam, base-model, and residual trajectory. It measures the deployable counterfactual on the fixed logged schedule.
- A **fixed re-anchor replay** restores the logged parameters and Adam state before each update. It isolates approximately one update and the current forward contribution, but intentionally removes accumulated candidate-path effects.

The repeated run used a `20 -> 400 -> 20 QPS` sinusoid with a 300-second period for 1,800 seconds. Its score window is `1791577711000000 <= ready_time_us < 1791579511000000`; data is under `/tmp/navi-decode-high-20261010-2028-jsonl/`. MI `11.225.198.191` is excluded from efficacy scores for the same numerical-divergence reason described above. This leaves 14,446 MI and 12,543 RTX predictions.

The neighbor result is real but narrow. Among 45 MI pairs, the slower member has a smaller P18 value in `88.9%` of pairs, while P17 is nearly unchanged. MI P17 is already saturated (`p50=499`, `p90=508`, cap 512), so the pair result means: at the same saturated decode count, a younger cohort tends to retain slots longer. On 433 RTX pairs, the slower member has smaller P18 in `49.4%` of pairs, indistinguishable from random direction. Across all MI samples, correlation between P18 and baseline underprediction is only `-0.077`; it is `-0.025` while QPS rises and `-0.227` while QPS falls. Therefore a local MI falling-phase relation is not a stationary global coefficient.

| Continuous replay variant | MI MAE ms | MI change versus baseline | RTX MAE ms | RTX change versus baseline |
| --- | ---: | ---: | ---: | ---: |
| Baseline, P17/P18 masked | 1067.60 | reference | 669.44 | reference |
| P17 at default parameter rate | 1252.35 | -17.30% | 673.12 | -0.55% |
| P18 at default parameter rate | 1108.10 | -3.79% | 654.90 | +2.17% |
| P17 and P18 at default parameter rates | 1240.32 | -16.18% | 654.66 | +2.21% |
| P17 at `0.01x`, P18 at `1x` | 972.54 | +8.90% | 653.65 | +2.36% |

Here a positive change means lower MAE. P17 at `0.01x` alone improves MI by `2.05%` but worsens RTX by `1.59%`; P18 alone has the opposite hardware tradeoff shown in the table. Their joint candidate improves both. The pair is therefore not two independent strong predictors. P17 supplies cohort cardinality and P18 supplies aggregate cohort progress, allowing the model to distinguish a small old cohort from a large young cohort. P18 alone conflates those cases, while fast P17 learning destabilizes the decomposition.

The optimizer trace explains the default-rate failure. P17 enters the nonlinear workload input unscaled at values near 512; P18 is first divided by 1,024 and is around 33 in the MI window. Adam independently normalizes each parameter's gradient history, so their parameter coefficients receive similar cumulative update magnitudes even though one unit of P17 has a much larger forward response. Across 20 hosts, default P17 and P18 cumulative absolute coefficient updates have a mean ratio of `0.87`, while their cumulative absolute prediction shifts have a mean ratio of `9.99`. On representative MI `11.225.192.113`, slowing P17 changes its cumulative prediction shift from `93,327 ms` to `1,333 ms`, while P18 remains active (`8,501 -> 10,824 ms`). On representative RTX `215.8.64.161`, P17 changes from `13,313 ms` to `103 ms`, while P18 changes from `1,173 ms` to `870 ms`. This is direct evidence that default Adam over-responds through P17 rather than evidence that P17 has no information.

The `P17=0.01x, P18=1x` candidate is not selected only on one aggregate number:

- In the high-pressure window, MI improves in 7/9 hosts and 5/6 cycles; RTX improves in 6/10 hosts and 4/6 cycles. Host-cluster bootstrap gives MI a 95% improvement interval of `[22.95, 175.29] ms`; RTX crosses zero at `[-9.04, 45.86] ms`.
- In the earlier lower-pressure window, the same fixed candidate improves aggregate MI MAE by `15.13%` and RTX by `9.94%`, but MI improves in only 5/9 hosts and its host-cluster 95% interval crosses zero.
- Leave-one-cycle-out rate selection chooses a rate using the other five high-pressure cycles, then scores the unseen cycle. Aggregated held cycles improve MI by `8.00%` and RTX by `0.86%`, but combined MAE still regresses by `0.47%` on cycle 3 and `2.57%` on cycle 5.
- With parameters P0-P16 effectively frozen and only P17/P18 trainable, adding the new fields worsens the already weak frozen baseline. They do not independently replace the old workload model; their useful information is conditional on it.

The apparent contradiction is therefore resolved: P18 contains a repeatable local conditional signal on saturated MI cohorts, but the signal is phase-dependent and requires P17 to disambiguate cohort size. Default Adam gives the nearly saturated P17 an excessive forward response and corrupts the continuous parameter path, so the default ablation regresses. A slower P17 rate recovers most of the aggregate signal, but the remaining per-host and late-cycle regressions are why this rate is an online experiment rather than a new default.

### ABAB correction: replay the runtime configuration timeline

The earlier B1/B2 counterfactual values of approximately `0.096%` and `1.39%`
are invalid. Those replays applied one command-line rate to the whole input even
though the live run switched P17 between `1x` and `0.01x`. Re-exporting every
update with the effective scheduler configuration from the preceding
`[navi-scheduler-config-update]` record reproduces all 28,876 updates on 20
hosts. The maximum differences from the logged online state are `1.14e-13` for
parameters, `1.18e-10 ms` for residuals, `9.35e-13` for Adam first moments,
`4.37e-11` for Adam second moments, and zero for residual timestamps.

A second replay issue was exposed by `11.225.205.114`. Starting a replay in the
middle of a window from decimal log values initially agrees to about `1e-13`,
but the high-rate nonlinear updates amplify that rounding difference after
roughly 50 to 100 updates. The baseline then differs by as much as `1.64` in a
base parameter and `2297 ms` in a residual. This is numerical path sensitivity,
not a missing formula input: replaying from the beginning remains exact. The
corrected counterfactuals therefore replay the full history and change only the
P17 rate in the target interval, without an intermediate state restore.

In the following table, **slow improvement** is
`(fast-rate MAE - slow-rate MAE) / fast-rate MAE`. Positive means P17 at
`0.01x` is better. B1 and B2 ran slow online and are compared with a full-history
fast counterfactual; A ran fast online and is compared with a full-history slow
counterfactual.

| Interval | MI samples | MI MAE slow / fast (ms) | MI slow improvement | RTX MAE slow / fast (ms) | RTX slow improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 6,207 | 1127.15 / 1182.34 | 4.90% | 533.62 / 535.83 | 0.41% |
| A, reverse direction | 11,345 | 1220.97 / 1260.95 | 3.17% | 511.65 / 516.63 | 0.96% |
| B2 | 5,680 | 1042.47 / 1073.85 | 3.01% | 509.79 / 504.93 | -0.95% |
| Sample-weighted total | 23,232 | 1152.26 / 1194.20 | 3.51% | 516.50 / 518.45 | 0.38% |

The MI direction is stable: slow P17 wins on all 10 MI hosts after the three
intervals are combined. B1's host-cluster 95% difference interval is
`[34.57, 79.42] ms` in favor of slow P17; A's is `[16.34, 65.67] ms`; B2's is
`[2.54, 65.62] ms`. RTX improves on only 5 of 10 hosts in the combined result,
so its `0.38%` aggregate change is not a stable hardware-wide benefit.

The next table is a different experiment. Each candidate replays from the
beginning with the named model input held at zero. **Removal improvement** is
`(online MAE - removed-feature MAE) / online MAE`; positive means the current
feature path is harmful on this fixed logged schedule.

| Removed input | MI MAE online -> candidate (ms) | MI removal improvement | RTX MAE online -> candidate (ms) | RTX removal improvement |
| --- | ---: | ---: | ---: | ---: |
| P17 decode request count | 1171.79 -> 1023.59 | 12.65% | 519.08 -> 535.59 | -3.18% |
| P18 generated token sum | 1171.79 -> 1169.88 | 0.16% | 519.08 -> 548.28 | -5.62% |
| P17 and P18 | 1171.79 -> 1001.63 | 14.52% | 519.08 -> 538.19 | -3.68% |

This resolves the apparent contradiction without claiming that decode state is
useless. P17/P18 contain conditional information, as shown by the lifecycle
neighbor pairs and by RTX regressions when the fields are removed. The current
shared parameterization is nevertheless harmful on aggregate MI: P17 is almost
saturated, duplicates existing pressure inputs, and has about ten times P18's
accumulated forward response under default Adam. Slowing P17 removes part of
that optimizer instability (`3.51%`), while removing its current forward path
removes more (`12.65%`). Neither is a production choice yet: removing P17 still
worsens 2 of 10 MI hosts over the combined intervals, and removing both worsens
3 of 10. The next feature candidate must represent remaining decode work or
slot-release pressure directly and must pass the same full-history, per-host,
forward-and-reverse test.

### Remaining decode budget instrumentation

The weak global P18 ablation does not contradict the lifecycle-pair result.
P18 is completed work. In the 45 MI old-feature-neighbor pairs, a smaller P18
usually identifies a younger decode cohort, but it does not distinguish a
short request near completion from a long request near its beginning. Online
retraining can also move this phase-dependent signal into correlated
running/waiting features and the residual model. Consequently, conditional
pair separation and marginal online-ablation gain answer different questions.

The next observation is therefore named `running_decode_remaining_token_budget`
and is computed at admission as
`sum(max(0, request.max_tokens - request.num_output_tokens))` over
`scheduler.running` requests that have emitted at least one token. This is an
output upper-bound budget, not the unknown final output length. It remains zero
on PD-separated paths and is carried through the workload snapshot, unified
latency sample, agent/master summaries, schedule replay, logs, and KMonitor.
It is observational only until a repeated online run and full-history replay
show stable held-cycle gains across MI and RTX.

The instrumentation review also found that the Python workload publisher sent
the running-prefill count in the `running_request_count` field while sending
P17 for all running decode requests. That violates the C++ invariant
`running_decode_request_count <= running_request_count` under decode-heavy
load. The publisher now sends the actual `len(scheduler.running)`; the separate
`running_capacity_prefill_request_count` remains the population used to advance
running-prefill elapsed time on Master. This affects the live workload/control
path, not the already-exact admission training samples used by the preceding
P17/P18 replay.

As a backward-data check, the updated replay read the prior 20-host exact JSONL
and emitted all 635,263 baseline rows. With fixed per-update re-anchoring, every
field is identical to the previous reference (`max_abs_diff=0`), so the absent
budget field defaults to zero without changing the prior conclusion.

### Exact remaining-budget repeat on 2026-10-10

The repeated run used the fixed online configuration (`current sample weight=4`,
`base learning-rate scale=16`, `residual learning rate=0.8`, adaptive rates
disabled) and a `20 -> 400 -> 20 QPS` sinusoid with a 300-second period for
1,800 seconds. All 24 clients answered the history query, 19 participated, and
the replay stopped automatically without a client error. Client ingress,
Master ingress, and Master scheduling recorded approximately 377,747, 377,949,
and 377,927 requests respectively. Agent ingress recorded approximately
276,101 requests because overload handling prevented all scheduled requests
from reaching execution; the model analysis uses completed agent samples, not
the offered request count.

The score window is
`1791594797000000 <= ready_time_us < 1791596597000000`, where ready time is
`event_time_us - target_ms * 1000`. It contains 19,207 MI308 and 20,318 RTX PRO
5000 predictions. The online-equivalent P17+P18 replay reproduces every update
on all 20 hosts. Maximum differences from the logged states are `4.55e-13` for
base parameters, `7.28e-11 ms` for residual parameters, `1.74e-14` for Adam
first moments, `7.11e-14` for Adam second moments, and zero for residual
timestamps. No host is excluded from this comparison.

The following table reports final-prediction MAE. The online P17+P18 path is
the reference. `Budget` temporarily maps remaining budget into P18's parameter
slot, so it is a screening result rather than an independent production
parameter.

| Model inputs | MI308 MAE ms | Change versus online | RTX 5000 MAE ms | Change versus online |
| --- | ---: | ---: | ---: | ---: |
| Online P17 + P18 | 135.73 | reference | 169.49 | reference |
| No decode input | 143.37 | -5.63% | 159.61 | +5.83% |
| P17 only | 136.20 | -0.35% | 168.42 | +0.63% |
| P18 only | 141.96 | -4.59% | 164.63 | +2.87% |
| Budget only, in P18 slot | 141.80 | -4.48% | 162.79 | +3.95% |
| P17 + budget, budget in P18 slot | 136.01 | -0.21% | 166.20 | +1.94% |

The sign differs by hardware and pressure direction. Removing both decode
inputs worsens MI rising-phase MAE by `13.97%` but improves MI falling-phase
MAE by `5.17%`. On RTX it improves rising and falling MAE by `6.49%` and
`5.10%`. Therefore an aggregate near-zero result can be cancellation between
opposite dynamic phases; it is not evidence that the fields never affect the
model.

The raw correlation and ablation results are not contradictory. P17 correlates
with target latency by `0.549` on MI and `0.403` on RTX, but it is almost a
duplicate of existing pressure state. Its correlation with
`preceding_request_count` is `0.9990` on MI and `0.9977` on RTX, and linear
regression on the 13 pre-existing features explains `99.95%` and `99.71%` of
P17 variance. The same old features explain `91.72%/85.66%` of P18 variance and
`98.06%/85.52%` of remaining-budget variance on MI/RTX. A feature can identify
a slow state in a raw pair while adding almost no new dimension to a model that
already observes that state.

Remaining budget is also highly determined by P17 and P18. They explain
`98.18%` of its variance on MI and `82.78%` on RTX. After removing their linear
contribution, remaining budget's partial correlation with current prediction
error is only `0.021` on MI but `0.285` on RTX. This matches the replay:
replacing P18 with budget adds only `0.14%` over P17 on MI and `1.32%` on RTX.
In a separate leave-one-cycle-out correction with whole cycles held out, adding
budget beyond P17/P18 changes MI MAE by less than `0.01%`; RTX improves by
`1.80%`, but validation accepts the correction in only two of six cycles.

P18's learned coefficient is usually negative (window means `-3.54` on MI and
`-2.45` on RTX). This is semantically consistent: generated tokens primarily
measure cohort age, and an older cohort is often closer to releasing slots.
It is not a positive remaining-work feature. The configured remaining budget
is better named, but it is still an upper bound because requests may terminate
before `max_tokens`, and an aggregate sum loses the distribution of which slot
will be released first.

The production decision remains unchanged: keep P17, P18, and remaining budget
as exact observability fields, but do not add remaining budget to the default
formula from this run. The missing causal state is closer to slot-release time
or the distribution of per-request remaining service than another aggregate
request count. Any replacement must improve unseen whole cycles on both
hardware types before it changes the parameter layout.

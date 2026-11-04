# Same-input latency conflicts and predecessor audit, 2026-10-09

## Scope and method

The initial 17:15-17:45 investigation was read-only. The arrival-cutoff experiment
and local instrumentation added afterwards are described separately below.
Window: 17:15:00 through 17:45:00 CST, completion time. Twenty-two nodes:
10 MI308, 10 RTX PRO 5000 and 2 H20. All collections succeeded.
Compare samples only within the same node and process lifetime.

Input: all 12 active, nonconstant v21 non-PD features: running and waiting
request counts, hit/miss tokens, miss-square sums, hit-miss product sums,
preceding request count and running-prefill elapsed time. Batch count is disabled
in these models. Each feature is divided by its within-node window standard
deviation; zero-variance features use scale one. Compute ordinary Euclidean
distance by looping over samples. No latency is used to calculate distance.

Strict neighbor criterion: distance <= 0.2 AND every component differs by <=5%,
where relative difference is abs(a-b)/max(1,abs(a),abs(b)). Both completion times
and inferred admission times must be at least 60 seconds apart. A large latency
conflict means ratio >=1.5 AND absolute difference >=5 seconds. Admission time
is reconstructed from completion minus the logged unified target duration.

Pair counts are unordered. Displayed pairs never reuse a sample within a node.
Disjoint pairs are selected greedily by descending target gap, not by a maximum
matching algorithm. Nearest-neighbor counts count samples, not independent pairs.

## Cross-hardware results

All figures below use the same strict neighbor definition. Percentages are
conditional on finding a qualifying nearest neighbor; they are not the fraction
of all queries or an intrinsic hardware error probability.

| Hardware | Samples | Close pairs | Large-conflict pairs | Disjoint conflicts | Nodes with conflicts | Samples with nearest neighbor | Large-conflict nearest samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MI308 | 5797 | 194 | 19 | 10 | 4/10 | 208 | 21 (10.10%) |
| RTX PRO 5000 | 4812 | 930 | 1 | 1 | 1/10 | 688 | 2 (0.29%) |
| H20 | 154 | 0 | 0 | 0 | 0/2 | 0 | Not estimable |

With the per-component guard relaxed to 10%, retaining distance <=0.2:
MI308 has 38 conflicting pairs out of 383, and 28/327 nearest samples conflict;
RTX has 15/1707 pairs and 25/1083 nearest samples. Thus RTX is not immune, but
strict conflicts are more prominent in this MI cohort. H20 still has no matching
pairs, so zero conflicts is not evidence that H20 is unaffected.

Selected examples: times are CST completion times; latency and queue are seconds.
The vector distance is dimensionless. Maximum component difference is percent.

| Node | Batch pair | Completion times | Distance | Max component difference | Actual latency pair | Backend queue pair |
| --- | --- | --- | ---: | ---: | --- | --- |
| MI 11.225.195.118 | 4583 / 4736 | 17:20:02 / 17:25:48 | 0.1600 | 4.68% | 77.712 / 46.426 | 75.586 / 44.927 |
| MI 11.225.199.135 | 4483 / 4743 | 17:20:34 / 17:35:09 | 0.1404 | 3.72% | 42.186 / 76.077 | 41.467 / 75.693 |
| MI 11.225.199.135 | 4479 / 4836 | 17:20:30 / 17:40:04 | 0.0821 | 2.79% | 54.763 / 83.429 | 54.042 / 80.718 |
| MI 11.225.202.174 | 4474 / 4878 | 17:20:10 / 17:41:43 | 0.1293 | 3.35% | 108.911 / 57.944 | 107.381 / 56.763 |
| MI 11.225.202.175 | 4485 / 4776 | 17:30:03 / 17:41:02 | 0.0772 | 2.73% | 77.074 / 38.730 | 75.288 / 37.401 |
| RTX 33.248.51.230 | 2725 / 3215 | 17:17:54 / 17:43:09 | 0.0809 | 4.44% | 14.421 / 24.892 | 9.472 / 10.805 |

The RTX example differs primarily after first execution: batch prefill wall times
are 4.950 versus 14.088 seconds, rather than a large initial-queue difference.
These batches contain multiple requests; wall time is not one request's GPU time.
Start snapshots have 64 running slots and no preemption/resume. The MI examples
have 512 running slots and their differences are primarily initial queue time.

The existing v21 Python formula was checked against logged parameter revisions
for all 10,763 window samples. Maximum absolute base-prediction discrepancy was
5.53e-10 ms. Applying either member's parameter vector to BOTH members of a pair
does not produce the observed divergence. For MI 4483/4743, the two predictions
are 124.097/124.837 seconds under the first state and 76.999/77.496 under the
second state, versus actual 42.186/76.077. This rules out confusing different
online coefficient revisions with a same-model feature-distance comparison.
It does not mathematically prove that every possible formula on these features
must fail; near-identical is not identical, and latent state can include future
interference as well as omitted admission features.

## Detailed predecessor audit: MI 4483 and 4743

Both admission snapshots have 512 running requests and zero running-prefill
requests. Waiting-prefill counts before adding the target batch are 184 and 188.
The target batch sizes are four and one respectively.

| Event | Batch 4483 | Batch 4743 |
| --- | --- | --- |
| Inferred admission CST | 17:19:52.005 | 17:33:53.242 |
| Inferred first execution CST | 17:20:33.472 | 17:35:08.934 |
| Prefill completion CST | 17:20:34.191 | 17:35:09.319 |
| Unfinished preceding prefill batches | 4463 through 4482 (20) | 4720 through 4742 (23) |
| Requests in those complete batches | 194 | 203 |
| Total prefill request-step count of those batches | 194 | 203 |
| Largest preceding batch prefill wall time | 4463: 9.657s, 21 requests | 4721: 8.432s, 18 requests |

The 194/203 totals include requests in the first batch that had already completed
prefill at the target admission; they must not replace the snapshot's 184/188
remaining-prefill counts. Each request in these batches contributes exactly one
prefill request-step. There is no evidence of a long, repeatedly chunked individual
prefill in these preceding batches.

In the interior of the waiting intervals (exclude one second at each boundary
because stdout timestamps have one-second precision), the logs contain:

| Measurement | Before 4483 | Before 4743 |
| --- | ---: | ---: |
| Logged steps containing prefill | 100 | 129 |
| Prefill request-steps | 178 | 178 |
| Prefill tokens | 10010 | 10419 |
| Mean prefill requests per logged step | 1.780 | 1.380 |
| Mean decode requests per logged step | 510.220 | 510.620 |
| Mean execute-model duration, ms | 399.185 | 387.715 |
| Maximum execute-model duration, ms | 497.335 | 483.726 |

The slower waiting interval does NOT show slower individual logged mixed steps.
It processes a comparable prefill workload in more, more sparsely placed steps.
No later-admitted completed batch starts over 5ms ahead of either target. The
observed prefill-start snapshots show no preemption, resume or running recompute.
These are batch-level and sampled-state checks, not a proof of no per-request
reordering at any intervening step.

## Request-ID evidence of ongoing decode

Collected 14,461 KV `stored` events from the two exact waiting intervals in
`navi_sched_alog.log`. Join cause_request_id and cause_batch_id to training logs.
For requests whose ENTIRE batch had completed prefill before target admission,
subsequent KV writes demonstrate continuing token/KV progress (subject to the
usual scheduler's forward allocation, not precise GPU completion timestamps).
The trace does not contain terminal events, and no last-write time is treated as
a request completion or slot-release time.

| Observed subset | Before 4483 | Before 4743 |
| --- | ---: | ---: |
| Identified preexisting post-prefill requests with KV writes | 499 | 492 |
| Median lower bound on decode age at admission, s | 122.641 | 81.141 |
| Age below 60s | 38 | 170 |
| Age 60-100s | 59 | 227 |
| Age 100-150s | 401 | 33 |
| Age at least 150s | 1 | 62 |
| Median interval between observed KV store events, s | 4.398 | 4.141 |

Decode age here means target admission minus the whole batch's prefill completion.
It is a LOWER BOUND for each individual request and describes only the observed
subset, not all 512 requests. It is not remaining decode length.

Examples of long-lived preceding requests:

| Waiting target | Request ID | Source batch | Decode-age lower bound at admission, s | Still making KV progress at decode-age >=, s | Store events during target wait |
| --- | --- | ---: | ---: | ---: | ---: |
| 4483 | 10899060759381932205 | 4431 | 234.821 | 245.676 | 3 |
| 4743 | 10899060759382068656 | 4636 | 251.462 | 327.142 | 19 |
| 4743 | 10899060759382071781 | 4643 | 218.969 | 278.429 | 15 |
| 4743 | 10899060759382076960 | 4651 | 203.718 | 252.182 | 12 |

Thus there ARE long-lived preceding decode requests, and the cohort age/mix is
different despite similar input vectors. They continue progressing; these records
do not establish a frozen request or a single slow GPU invocation. One old request
occupies only one slot and cannot alone explain why 512 slots remain occupied.
Similar total output lengths would not imply similar REMAINING work at admission.
Actual individual output lengths have not been recovered, so their equality is
not assumed in the conclusion.

## Evidence limits and next instrumentation needed

- A complete lifecycle for all preceding requests cannot be recovered from this
  capture. Admission/pre-fill batch membership is logged; per-request decode
  completion, output-token count and slot release are not.
- Pure decode steps are skipped by `_complete_backend_execution` when there is
  no prefill progress. Gaps between logged prefill steps therefore cannot be
  labelled CPU/GPU stalls or assigned entirely to decode from these logs alone.
- `execute_model_sum_ms` is not a disjoint time partition. Async step spans can
  overlap, and negative `inter_step_gap_ms` occurs in these logs. Do not calculate
  idle time as wall minus that sum or sum overlapping batch durations.
- The missing causal audit would record arrival/order, first-prefill completion,
  output progress, terminal reason and slot release per request, plus schedule
  step IDs and pure-decode intervals, using a bounded rotating file rather than
  stdout. No such production change or deployment was made in this investigation.
- Evidence currently favors unrepresented decode cohort/progress state over
  unexpectedly slow individual prefills for this pair. Exact attribution of the
  additional 34.226 seconds of queue time remains unproven without release events.

## Artifacts and tests

- Analysis: `AnalyzeLatencyFeatureNeighbors.py`; ten focused tests in
  `AnalyzeLatencyFeatureNeighborsTest.py`, all passing under conda310 Python.
- Current captures and architecture metadata:
  `/tmp/navi-neighbors-all-20261009-1800/manifest.json` and
  `/tmp/navi-neighbors-h20-20261009-1800/manifest.json`.
- All thresholds, disjoint examples and component differences:
  `/tmp/navi-neighbors-all-20261009-1800/comparison.json`.
- Formula reconstruction and shared-state checks:
  `/tmp/navi-neighbors-all-20261009-1800/model-validation.json`.
- Full lists of the 43 predecessor batches and 991 observed decode-request
  records, including request IDs and store event timestamps:
  `/tmp/navi-neighbors-all-20261009-1800/135-predecessor-audit.json`.
- Original per-step evidence:
  `/tmp/navi-mi135-20261009-1730/execution-window.log` and
  `/tmp/navi-neighbors-all-20261009-1800/135-step-gaps.log`.

Reproduce one strict same-node comparison:

```bash
/home/zq103303/conda310/bin/python3.10 \
  aios/navi_sched/test/AnalyzeLatencyFeatureNeighbors.py \
  /tmp/navi-neighbors-all-20261009-1800/MI_11.225.199.135.log.gz \
  --focus-start 2026-10-09T17:15:00+08:00 \
  --focus-end 2026-10-09T17:45:00+08:00 \
  --history-start 2026-10-09T17:15:00+08:00 \
  --history-end 2026-10-09T17:45:00+08:00 \
  --max-relative-difference 0.05

/home/zq103303/conda310/bin/python3.10 \
  aios/navi_sched/test/AnalyzeLatencyFeatureNeighborsTest.py
```

## Arrival-only cutoff experiment, 18:16-18:35 CST

Master was 172.18.97.117. The old sine controller was terminated without invoking
its stop-on-exit handler. All traffic commands used the container's installed
`/usr/local/bin/navi_sched_replay`. At 18:16:28 the active replay was updated to
80 total QPS. At 18:28:26.523-26.647 it was updated to 0.000001 QPS, retaining
the same selection and outstanding requests. This is effectively an arrival
cutoff for this observation window, not an indefinite pause. `stop` was not used:
the current client stop callback aborts outstanding replay graphs.

Configuration generation 1791528130516202 was unchanged: adaptive learning off,
learning-rate scale 16, residual learning rate 1, residual update error ratio 0.3,
current-sample weight 4 and batch timeout 10 ms. Fifteen clients had history.
The entry rate was approximately 80 QPS; scheduled traffic was approximately
68-74 QPS, with 6-12 QPS rejected by overload admission. Thus the plateau was not
an unconstrained sub-capacity experiment. Master ingress became zero at
18:28:28. Collected samples contain no inferred admissions more than two seconds
after cutoff. Twenty-node training-log collection succeeded.

Steady controls were admitted at least 180 seconds after the plateau began and
at least 120 seconds before cutoff, completing at least 60 seconds before cutoff.
Tail samples were admitted in the final 60 seconds before cutoff and completed
more than two seconds afterwards. Admission is inferred as completion minus
logged target; it is Agent admission, not Master ingress. Samples are batch-level
training records, not individual decode completions.

In the following table, Actual and Master are sample-weighted mean latency in
seconds. Error = original Master prediction minus actual latency: positive means
overprediction, negative means underprediction. N is the number of samples.

| Hardware / cohort | N | Actual, s | Master, s | Error, s |
| --- | ---: | ---: | ---: | ---: |
| MI308, steady | 1771 | 29.433 | 29.468 | +0.035 |
| MI308, tail | 118 | 30.366 | 28.795 | -1.571 |
| RTX PRO 5000, steady | 1223 | 32.605 | 32.365 | -0.240 |
| RTX PRO 5000, tail | 86 | 29.838 | 30.427 | +0.590 |

Excluding unstable MI node 11.225.199.78 does not reverse the result: steady
error is -0.264 seconds (1610 samples), tail error -1.583 seconds (116 samples).
MI node effects are mixed. Tail error on 11.225.200.122 is -10.03 seconds
(15 samples), while 11.225.202.174 is +6.56 seconds (5 samples). On the previously
audited node 11.225.199.135, tail actual latency is 32.03 seconds and error
-1.39 seconds (13 samples).

There is no broad, large tail overprediction on MI in this experiment. This does
not support future-arrival interference as the primary explanation for the old
34-second queue discrepancy; it does not exclude all such interference. Strict
steady-to-tail matching on the entire feature vector found ZERO pairs. A relaxed
exploratory distance limit of 0.5 and component guard of 10% finds 16 pairs, only
six of which also satisfy the original 5% component guard. None is an original
strict match; they must not be presented as controlled counterfactual evidence.

Last collected prefill completions were between 18:28:35 and 18:29:00. At
18:34:15-20 all 20 nodes had zero Navi queued/running-prefill counters. Native
vLLM running/waiting counters on MI135 and RTX33.248.51.230 were zero at 18:35:39.
Those two nodes' request-success counters were also zero, so they cannot prove
normal decode completion. Old MI logs do not provide a complete cancellation
audit. The experiment establishes the command path and observed drain, not a
complete per-request proof of no aborts.

Artifacts: `/tmp/navi-steady-drain-20261009/`, including `setup.json`, `cut.json`,
`post-state.json`, `analysis.json`, `analysis-10pct.json`, `ingress-after-cut.json`
and per-node `drained/` logs. `AnalyzeLatencyDrain.py` performs the analysis;
its five tests cover error signs, full-vector matching, cutoff boundaries and
missing controls.

## Scheduling controls and the next causal capture

The current vLLM V1 scheduler first assigns tokens to running requests, then
admits waiting requests while both token budget and running slots permit it.
FCFS is not an unconditional execution-order guarantee: KV availability, remote
KV waits, grammar readiness and preemption can change progress order.

| Control | Effect in the inspected source | Verified live setting |
| --- | --- | --- |
| `--max-num-seqs` | Maximum running slots; decode retains slots until completion/preemption | MI135/MI78: 512; RTX33.248.51.230: 64 |
| `--max-num-batched-tokens` | Per-step token budget shared by running decode and new prefill | RTX33.248.51.230: 4096; MI CLI does not explicitly set it |
| `--enable-chunked-prefill` | Allows a waiting prefill to consume part of the remaining budget | Resolved runtime value still needs new config audit |
| `--long-prefill-token-threshold` | Caps a long request's scheduled chunk | Resolved runtime value still needs new config audit |
| `--scheduling-policy` | FCFS vs priority admission/preemption policy | Resolve from runtime audit before comparisons |
| `--gpu-memory-utilization` | KV capacity indirectly affects admission/preemption, not a direct slot count | 0.9 on the three inspected nodes |

`max_num_partial_prefills` and `max_long_partial_prefills` do not directly cap
concurrent prefills in this V1 schedule loop. Their config path can set the
long-prefill threshold, so they are not guaranteed to have no effect. Do not
assume these knobs reproduce the behavior documented for other schedulers.
No vLLM parameter was changed during this experiment. A future max-num-seqs
comparison must keep model learning parameters, prompt selection and offered
traffic fixed and measure both latency and throughput, not just prediction error.

Local instrumentation now records:

- Scheduler-side `admitted`, `slot_acquire`, `first_output`, `slot_release`,
  `terminal` and `kv_free`, with per-process session, scheduler sequence, wall and
  monotonic timestamps, request ID, batch linkage, counts, actual output tokens,
  max output tokens, preemption count and reason. Listener errors are counted.
- Slot release is distinct from terminal and KV free. Streaming input can release
  a slot without terminating; KV transfer can delay freeing after termination.
  Bulk release queue counts are post-bulk snapshots, not intermediate counts.
- Every backend execution callback, including pure decode, records its step ID,
  prefill/decode tokens and requests, start/end clock, workload and success.
  These spans may overlap and are not GPU-only time; never sum them as disjoint
  wall-clock intervals.
- Resolved scheduler configuration is recorded when the scheduler is created.

The bridge uses the existing asynchronous rolling `logs/navi_sched_alog.log`
sink (1 GiB per file, retained count 20, compression and 128 MiB async cache).
No new directory, worker thread, lock or retained-request map is introduced.
The existing logger's rotation/restart semantics apply; these are diagnostics,
not a crash-durable transaction log. A crash can lose buffered events; parsers
must reject incomplete lifecycles rather than infer releases from missing data.

Instrumentation is locally implemented and has NOT yet been deployed as of
this report update. Nine native tests cover synchronous/asynchronous finish,
next waiter, preempt/resume, waiting abort, delayed free, duplicate finish,
listener failure, streaming wait, bulk abort and decode token-budget competition.
The targeted Navi observer/direct-transport Bazel tests passed with `py310`.
Broader native scheduler tests are NOT all green in this offline environment:
57 passed, 46 failed, one skipped in the attempted run; unavailable multimodal
model/tokenizer fixtures blocked that validation. GPU performance and the next
instrumented live replay remain unverified.

The next capture must start before traffic so all occupied-slot lifecycles are
present, repeat the same-input pair search, then account for each predecessor's
slot holding/release and decode completion during the target's waiting interval.
This is still the missing evidence for the old MI135 pair, not a diagnosed fix.

## Distant-pair and full-lifecycle analysis support

The neighbor tool now accepts `--min-separation-s 600` and
`--opposite-pressure-phase`. The latter requires rising vs falling LOCAL pressure,
defined by preceding-request count relative to its median 15-30 seconds earlier
(three earlier admissions required; changes within 5% are flat). It does not
silently label commanded QPS phase as local pressure phase. Phase is an additional
filter, never a substitute for the full 12-component distance and relative guard.

On the old 17:15-17:45 capture, combining ten-minute separation, opposite local
pressure phases and the original strict feature criteria finds no MI pairs and
41 RTX pairs, none with large latency conflicts. This is an insufficient matched
MI cohort, not evidence that the previously observed MI conflicts disappeared.
Phase labels can be inspected without requiring opposite phases. The next run
needs more complete cycles, retaining both successful matches and unmatched cases.

`AnalyzeVllmRequestLifecycles.py` reads the new rotating-file events for one node
and process session. It preserves every captured request and execution callback,
reconstructs slot intervals, and separates queue-to-slot, slot-to-first-output,
decode-to-terminal and total lifetime. For each selected batch it lists ALL
unfinished predecessors, their occupied-slot status, eventual output length and
remaining lifetime at target admission. It also lists every slot release and
acquisition while the batch waits, including later-admitted requests that obtain
a slot first. Comparing cohorts exposes shared long-lived predecessors even when
the target batches are ten minutes apart.

The parser checks scheduler sequence gaps, conflicting duplicates, missing slot
acquisitions, writer error counters, process restarts and malformed JSON tails.
It understands AUTIL's trailing record fields. Input is bounded to 2 GiB
decompressed / one million audit events by default; it writes the report to
stdout for a caller-selected artifact, not into the running service. A sequence
can be contiguous but not fully drained: these are separate fields, and neither
missing terminals nor a process restart may be interpreted as request completion.

Tests: 13 neighbor tests and 8 lifecycle-analysis tests pass. In particular, a
test with targets 600 seconds apart and a 900-second shared decode request must
report overlap. Raw events are retained so stage counts and timings are auditable,
not only aggregate statistics. No independence claim is valid when integrity
checks fail; even disjoint captured cohorts do not exclude all shared-resource
interference.

## Full surrounding-request evidence and rollout follow-up

The lifecycle report now explicitly lists every later arrival while the target
batch still owns resources, not only overtakers during its initial queue. Each
successor links to its complete raw lifecycle, batch, slot acquisitions and final
output count. All captured requests remain in the report, including those outside
the selected target intervals. A request that has terminated but still retains KV
blocks remains a resource-owning predecessor until the logged free event.

Pair comparisons report both shared admission-time predecessors and shared
requests across the complete target lifetimes. For example, a later arrival during
the first target can still be decoding when the second target arrives ten minutes
later. Disjoint target intervals alone do not establish independence. Missing
admission, terminal, slot release or KV free, sequence gaps and writer failures
are reported separately from the observed time separation.

The strict next comparison retains all 12 feature components, distance <=0.2,
maximum component difference <=5%, and prefers >=600-second separation with
opposite local pressure phases. No matches is a valid outcome and does not cause
an implicit threshold relaxation. Warmup is retained for predecessor tracking but
excluded from the selected comparison targets. Capture continues after offered
traffic is reduced so that native decode can finish; cancellation must not be used
to manufacture a fully drained tail.

Deployment evidence, 2026-10-09 around 19:22 CST:

- Wheel pipeline 287644 run 78189207 succeeded for Alibaba `08d5e4ba369`.
- vLLM image pipeline 27142 run 78189859 succeeded for vLLM `63b66b1ca4`,
  using that pinned wheel. The local analysis changes do not require a new image.
- MI308 deployment published biz 273 and reports FINISHED, ten ready nodes.
  All ten nodes' scheduler configuration audits were retrieved, with
  `lifecycle_supported=true`, FCFS, async scheduling, max sequences 512 and
  max batched tokens 8192. One ASI query timed out and succeeded on retry.
- RTX deployment was submitted as biz 274 after a template readback mismatch
  interrupted the first attempt. A later exact content comparison found no
  difference; retry used the same validation and publication flow, without
  bypassing checks. At this checkpoint RTX was still RUNNING; sampled startup
  logs showed graph compilation, not completed readiness.
- Master remains biz 272. Its live learning configuration is unchanged:
  scale 16, residual rate 1, residual update ratio 0.3, sample weight 4,
  adaptive disabled, generation 1791528130516202. Agent applied-config logs
  confirm the same learning settings; their batching override is 150 ms,
  while the Master's own batch timeout is 10 ms. These are not identical roles.
- A real initial capture from MI135 contains the new scheduler-config event
  plus initialized/applied configuration records. It does not yet contain an
  instrumented pressure run or establish a request-level cause of the old gap.

New local tests cover successors after first execution, delayed KV release,
missing tails, a successor bridging two distant target lifetimes, and a contiguous
terminal-only trace without admission. These are analysis integrity tests, not
evidence that an online latency defect has been resolved.

### Live lifecycle acceptance and new sine run

The first capture after publication now verifies real request transitions, not
only startup configuration. In the initial sine capture around 19:34 CST:

| Hardware | Captured nodes | Admitted requests | Fully completed lifecycles | Integrity errors |
| --- | ---: | ---: | ---: | ---: |
| MI308 | 10 | 9782 | 5871 | 0 |
| RTX PRO 5000 | 9 | 3626 | 3010 | 0 |

Captured nodes are the ready nodes actually downloaded, not the desired replica
count. Admitted requests count distinct request IDs within each process session.
Fully completed requires admission, terminal, KV free and all acquired slots
released. Integrity errors count missing sequences, conflicting duplicates,
missing admissions/acquisitions, malformed records and reported audit failures.
Requests still decoding at capture time remain in the data and are not failures
or fabricated completions. This snapshot is not the finished experiment.

One MI135 example, request `10899060759382710785`, batch 2, has 28.622 ms
admission-to-slot, 4513.409 ms slot-to-first-output, 94759.621 ms
first-output-to-terminal, and 547 final output tokens. An RTX example has 1188
final output tokens. The configured output cap is 2048 for these requests;
the cap must not be mistaken for a fixed actual output length. These are startup
acceptance examples, not matched steady-state pairs or a causal attribution.

The new sine controller started on Master at 19:33:19.758 CST, through the
container's `/usr/local/bin/navi_sched_replay`, using the existing continuous
driver. Command range is 0.01-400 QPS, period 300 seconds, duration 3600 seconds;
the first 900 seconds are warmup. Learning settings were not changed. The driver
does not use stop-on-exit; a final update reduces QPS to 0.000001 to allow native
decode to finish. The replay session itself has a 90-minute safety duration,
started before the sine, so final collection must not assume requests surviving
past that safety deadline completed normally.

At the initial readiness check RTX had nine ready replicas out of ten and its
rollout was still RUNNING. Topology-change intervals and cold-start acceptance
traffic must be excluded from steady-state pair selection. Successful control
updates prove only commanded QPS; the actual ingress curve still needs checking.

Local tests after these additions: 41 `Analyze*Test.py` tests passed, including
13 lifecycle-analysis tests. Code and this report are on `navi_sched`; no Master
rebase or unrelated worktree edits were included.

### Pure-decode workload-field caveat

Live acceptance exposed a diagnostic-field issue in deployed wheel `08d5e4ba369`:
`_start_execution_tracking` populated running/waiting and preempt/resume counts
only for steps containing prefill. The new pure-decode callback records therefore
carried default zeros for these fields. Those zeros are NOT measured empty queues.
Per-request scheduler transition counts, slot intervals, scheduled decode counts
and completion timestamps are separate and remain usable.

The analyzer now marks uncollected callback counts as null and preserves their raw
values separately. A follow-up observer patch reads queue lengths and transition
list lengths in constant time on every step, without scanning decode requests or
adding locks, and writes an explicit `workload_counts_collected` flag. This patch
is locally verified but NOT part of the running sine image: the ongoing capture
is not restarted. A test uses length-only containers that fail on iteration to
guard the constant-time path. Both observer/direct-transport Bazel targets passed
with clang/new_abi/debug/py310; 42 analysis tests passed. This corrects diagnostic
interpretation, not the unexplained MI steady-state latency discrepancy.

A one-shot local collector is scheduled for 20:50 CST, after the sine's planned
20:33 cutoff and before the replay session's safety expiry. It will fetch bounded
rotating records through the same ASI path into
`/tmp/navi-lifecycle-sine-final-20261009`. Its execution and authentication success
are pending, not claimed here. Node-level captures include their actual cutoff
time; records lacking a structured event timestamp are retained and counted
separately, not falsely represented as exact-window records.

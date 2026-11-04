# Discrete refinement baseline reuse

## Scope and evidence

This change reuses immutable node baseline values within one invocation of
`CostSchedulerKernelInternal::refineDiscreteAssignment`. It does not establish
the root cause of the historical 45-second scheduling wait. Production latency
and throughput after this change have not been verified.

The corrected production profile in
`/tmp/navi-gpu-zero-20261027/master-current.corrected-summary.json` identifies
discrete forward evaluation as a major CPU consumer. The original perf 5.10
symbolization was displaced by 0x1000 for this ELF; the uncorrected function
names must not be used as evidence.

The pre-change hashes of the implementation, header and incident tests match
`/tmp/navi-master-live-20261026-1327/package/manifest.json`. Thus the baseline
comparison starts from the implementation recorded for the deployed Master,
rather than an unrelated Git revision.

## Repeated work and lifetime

Each trial moves one request between two nodes. Both affected queues must still
be evaluated in request order, including promotion from waiting to running
capacity. Previously, every candidate also reloaded capacity metadata and
recalculated the same unified floor and reported workload linear costs.

`DiscreteForwardWorkspace::NodeBaseline` now stores those scalar values per
node. The first full evaluation populates them lazily. Pair trials reuse them.
A full evaluation always resets the cache; a different context address or node
count also resets it. The refinement workspace is local to a single invocation,
and the node info and latency model are held through const shared pointers.
There is no global cache or additional lock. Storage is O(node count), with no
persistent state.

The final full evaluation refreshes the baseline and recomputes the objective.
The addition order of floor, reported, hit-token and miss-token terms is
preserved. The cost model, optimizer budgets and candidate traversal are
unchanged. Faster evaluation under a finite wall-clock budget may permit more
candidates; exact assignment equality below is measured with fixed work.

## Validation

Artifacts are under `/tmp/navi-discrete-baseline-opt-20261027`.

- `old-expanded.xml`: uncached implementation fails the zero-reload assertion.
- `fixed-final.xml`: the affected tests pass with caching enabled.
- `removed.xml`: removing the implementation makes the assertion fail again.
- `restored.xml`: restoring the implementation passes all 107 affected tests.
- The cache lifecycle test covers a new context with equal node count, pair
  reuse and full refresh after a floor feature change.
- Existing independent evaluator, capacity split, unified prediction, gradient
  and replay context tests are included in the restored run.

The benchmark evaluates 4096 candidates, with no wall-clock limit, nine runs
per case and the first run excluded. Values below are the upper medians of the
eight retained timings, in microseconds (sorted index 4, zero-based).
All six selected-node vectors and hexadecimal floating-point objectives are
identical across old, fixed, removed and restored runs.

| Nodes | Requests | Old | Fixed | Removed | Restored |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 128 | 3859 | 3142 | 4074 | 3091 |
| 24 | 512 | 4955 | 4276 | 5207 | 4208 |
| 24 | 2048 | 9366 | 8841 | 9679 | 8803 |
| 432 | 128 | 3108 | 2574 | 3232 | 2558 |
| 432 | 512 | 4937 | 4241 | 5203 | 4197 |
| 432 | 2048 | 8734 | 8100 | 9125 | 7987 |

These are synthetic fixed-work measurements, not an online scheduling latency
or end-to-end throughput claim. The subsequent full run passed 627 role tests
and 42 cluster tests (`full-role.xml`, `full-cluster.xml`, `full.log`).

## Package scope

The Master wheel preserves the prior Master Python members and replaces only
the core library and wheel RECORD. The current core sources also contain six
inherited Agent files changed since that Master was built: Pybindings.cpp,
AgentBatchR.cpp/.h, NaviSchedResourceTest.cpp and NaviSchedAgent.cpp/.h. They
match the Agent v18 source manifest. Comparing the entire recorded source set
against that manifest yields exactly the three files changed by this
optimization. Both comparisons are recorded in the new package manifest; the
whole core must not be described as a three-file delta from the prior Master.

The built core contains `.symtab`, `.debug_info` and `.debug_line`. Both GDB
and llvm-addr2line resolve its `forwardDiscreteImpl` address to source line
892. This does not replace live perf validation in the target container.
The packaged library hash was verified in a production Python 3.10 runtime;
initialization and shutdown of a local cost Master both passed
(`packaged-smoke.log`). This smoke test uses a local test configuration, not
the production deployment configuration or real traffic.

## Remaining verification

The original inference nodes were no longer discoverable during the ASI
recheck. The original Whale deployment lookup returned `data: null`; it did
not provide current runtime state. Online rollout and real traffic validation
remain outstanding. No service was restarted to perform these offline checks.

## Follow-up: value-only discrete evaluation

Artifacts: `/tmp/navi-discrete-value-only-20261027`.

`forwardDiscreteImpl` also calculated two derivatives that it discarded. A
scalar overload of `calculateLatencyFromRequestCostSum` now uses `AutoGrad<0>`
to retain the original value arithmetic without gradient dimensions. A plain
double rewrite would change divisions performed by AutoGrad as multiplication
by a reciprocal. The existing derivative evaluator and continuous optimizer
are unchanged.

The numeric test compares exact double values for 25,344 inputs: both model
sizes, initial and seeded random parameters, positive/negative costs, adjacent
values around the nonlinear branch boundary, prompt sizes and preceding
request counts. All 26 model tests and 107 affected role tests pass in the
initial fixed run. The restored full run also passes all 627 role, 42 cluster
and 26 model tests. All six assignments and objectives remain exactly equal
in this restored run. The final packaged library hash was verified in the
production Python runtime, and local cost Master initialization and shutdown
both passed. This package has not been deployed or verified with real traffic.

An external GDB probe counts calls to the derivative evaluator only when
`forwardDiscreteImpl` is an ancestor frame. The same single replay test passes
its normal assertions in all versions. The additional zero-call requirement
fails with 32 calls on the old implementation, passes with zero after the
change, fails with 32 when the call sites are reverted, and passes with zero
when restored. The probe verifies that its breakpoint is resolved and that
the inferior exits successfully; it adds no production counters.

Initial fixed-work timings show a small change; these measurements do not
establish a large production improvement. Values are upper medians in us of
eight runs after one warmup, each with 4096 candidate evaluations.

| Nodes | Requests | Before Value-Only | Fixed | Call Sites Reverted |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 128 | 3091 | 3013 | 3073 |
| 24 | 512 | 4208 | 4105 | 4226 |
| 24 | 2048 | 8803 | 8746 | 8953 |
| 432 | 128 | 2558 | 2515 | 2550 |
| 432 | 512 | 4197 | 4145 | 4208 |
| 432 | 2048 | 7987 | 7927 | 8024 |

All six selected-node vectors and objective hexfloat strings are identical
between these three runs. New profiling is needed to establish the remaining
cost distribution after baseline caching and value-only evaluation.

## Historical window flow cross-check

`historical-flow-audit.json` independently parses the archived
`historical45-stages.log`, SHA256
`44fd000dec41561f8da3a6e75ed79f23abee40c3b4928fcaa68b839c53a55659`.
For schedule 45788, the 45.018234-second interval contains 528 context creation
timestamps representing 15,558 requests, and 274 admission-prediction starts
representing 8,169 requests. All requests in those prediction-start cohorts
have failure records. Their context-to-prediction waits increase from
22,364.856 ms to 45,066.346 ms.

The context timestamp is the logged `schedule_time_us`, not frontend request
arrival. Prediction-start counts are not successful execution throughput or
completed cleanup. The previous 275-round phase audit also includes a round
that intersects the left boundary; this audit counts starts strictly within
the interval. These counts support continuing backlog investigation but do
not identify CPU ownership or establish the root cause of the 45-second wait.

## Profile representativeness and recorded replay check

The fixed-work incident fixture attributes 40.59% of refinement samples
inclusively to residual application after symbol correction. This is not a
representative production hotspot: that fixture enables residual learning at
0.8 and deliberately accepts historical models through its test helper.
The actual captured replay has residual learning disabled on all 24 nodes
and zero residual parameters. No residual optimization follows from this
synthetic profile.

The latest implementation replays `current-pending-replay-8384.json` with
`--require-recorded-match`: one replay, zero failures, one recorded match,
189 requests, three scheduled and 186 rejected, with no invalid target.
Measured schedule computation is 961 us. The source SHA256 is
`3ddc9036a73103a600f6c73020f74d714d751a4edc2ba05671c009e6eac4f33b`.
This verifies the recorded decision with real pending state; only three
requests reach cost optimization, so it does not reproduce high-load latency.

A separate controlled load input retains all 24 node models, configuration,
and pending state, cycles recorded requests and matches with unique request
IDs, bypasses admission audit, and uses uniform initial weights, ten optimizer
iterations and 4096 discrete evaluations. It uses the replay runner's legacy
runtime defaults rather than restoring audit-only node runtime. It is a load
experiment, not an exact historical reproduction. At 2048 requests its first
successful run schedules all requests in 33,535 us. Input provenance and
alterations are recorded in `controlled-load-manifest.json` under
`/tmp/navi-discrete-value-only-20261027/`. Performance runs omit context summary
and replay-output flags so they do not enable optimizer replay construction.
The offline runner executes worker shards sequentially; these measurements
cannot establish production graph parallelism, per-round wall time or QPS.

The controlled load completes 64 profiled runs with zero replay failures.
Schedule computation ranges from 32,950 to 34,805 us (median 33,723 us).
The perf capture contains 17,771 user CPU samples. Its executable DWARF
callchains are empty in perf 5.10, so no inclusive attribution is valid.
Top instruction pointers resolved independently with LLVM and the recorded
ELF identity include 464 samples in `forwardDiscreteImpl` and 308 in
`aggregateCapacitySplitCost`. These are whole-process self samples; parsing,
pending restoration and process startup dominate the capture and are excluded
from any claim about production scheduling latency.

## Reuse equal-capacity continuous aggregates

Continuous evaluation computes a request-cost aggregate for current running
capacity, then another for reservation-only capacity. The captured node
metadata has equal capacities on nine of 24 nodes. When equal, both sums
read the same immutable cost rows and weights in the same order. Reuse the
first scalar before adding existing workload costs. Unequal capacities and
non-unified prediction retain their existing paths; no shared state or
optimizer policy changes.

External GDB call-count gates on the 2048-request controlled input require at
most 429 calls to `aggregateCapacitySplitCost`. Old behavior makes 528 calls
and fails; fixed makes 429 and passes; removing the change fails at 528;
restoring passes at 429. Each inferior exits normally and its replay succeeds.
The 99 eliminated calls equal nine nodes times eleven objective evaluations.
An initial probe attempted to compare optimized-out parameter values and was
discarded; only the independent entry count is used in the final gate.

Full old/fixed replay outputs compare exactly after removing measured timing
fields: all 2048 decisions, objective values, gradients and weights match.
Artifacts are under `/tmp/navi-capacity-reuse-20261027/`, with source hashes
in `gates.json` and comparison scope in `numeric-comparison.json`. GDB elapsed
times are not used to estimate performance. This optimization has not been
deployed or verified with live traffic.

With identical sampling settings and 64 runs per version, schedule computation
has median 33,723 us before capacity reuse and 32,661 us after (3.15% lower).
This is a sequential offline experiment, not a production throughput result.
The fixed full test run passes all 42 cluster tests and 626 of 627 role tests,
including all 19 incident tests and 11 context replay tests. The failing role
test is `testConcurrentStopFinishAndAbortReleaseEveryInflightOwnerExactlyOnce`:
its missing-remove count changes from zero to one. A separate 500-repeat run
also fails with capacity reuse removed, confirming this failure exists before
the change. Repetition with the change passes 500 iterations; that does not
establish a race fix. Overall validation is not fully green, and no deployment
or new package publication is claimed.

A separate 20,000-request input with the same controlled-work settings takes
335,987 us, schedules 5,107 requests and rejects 14,893 with zero invalid
targets. It does not reproduce a 45-second wait and does not exercise online
dispatch or failure cleanup.

## Pending exclusion trigger experiment

Reuse the existing `master_pending_cache_benchmark --exclusion-replay` rather
than adding another benchmark. The old scan oracle was checked against
`648100e5:aios/navi_sched/resource/PendingCacheUtil.cpp`: after a positive
pending reference lookup, both exclusion methods iterate all reservations.
That revision's dispatch failure copies the batch, marks one request failed,
and refreshes remaining requests' cache hits before committing the candidate.
This code path establishes an algorithmic amplification mechanism, not full
CPU attribution of the historical incident.

Starting with the recorded pending state, clone node zero's scheduled and
agent ownership records with unique IDs. Retain all hashes and recompute
checksums, checking the checksum implementation against the unmodified
capture first. Do not amplify Master reservation payloads or workload; this
input is only valid for the pending-index benchmark. Each input is bounded
below its existing 256 MiB input limit. Provenance and alterations are in
`pending-amplification-manifest.json`.

| Entries Per Source | Isolated Scan ms | Indexed ms | Full KV Scan ms | Full KV Indexed ms |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 6.951309 | 0.141105 | 0.039590 | 0.042252 |
| 4096 | 64.617505 | 0.146473 | 0.039095 | 0.042576 |

The full KV negative control matters: more batches alone do not trigger the
slow scan when physical readiness short-circuits the pending checks.

Next remove 96 physical cache hashes on node zero that remain covered by
pending ownership. This deliberately altered readiness is not a historical
capture. Both full KV variants still return exactly 12,873 matched blocks
over 189 requests. All native-index correctness checks also pass.

| Entries Per Source | Full KV Scan ms | Full KV Indexed ms |
| ---: | ---: | ---: |
| 512 | 11.929970 | 0.077054 |
| 4096 | 100.275599 | 0.078526 |

These are benchmark helper CPU times (median of five interleaved trials),
not production RPC latency. The expensive condition is a physical miss with
a positive pending reference, even if that reference is ultimately excluded.
Larger node, request or prompt counts in other tests do not ensure that
condition. Current indexing removes batch-count scaling in this experiment.

An independent extraction of archived decision-audit samples within the
logged historical wait window, 09:27:34.277809 through 09:28:19.296043 on
2026-10-21, finds 28 selected samples, of which 22 have positive pending-hit
deltas (822 blocks in total). `historical-pending-window-audit.json` records
source hashes, timestamps and rows. These selected samples establish that
pending-only coverage existed in the window; they do not measure scan calls,
represent the whole traffic population or prove that this mechanism accounts
for all 45 seconds. Historical binary identity and same-window CPU ownership
remain necessary before claiming that full root cause.

## Historical report age and result wait are independent, 2026-10-29 audit

Re-read 9,449 archived rounds from `historical45-rounds.log` and the exact
two prediction log records for schedule 45788 in `historical45-selected.log`.
All times in this section are UTC on 2026-10-21 (add eight hours for CST).
The target context was created at 09:27:34.277809; its first prediction was
logged at 09:28:19.296043, a 45,018.234 ms interval.

| Selected Agent | Workload revision | Workload receive time | Report age in target context |
| --- | ---: | --- | ---: |
| 33.6.167.237:25954 | 930 | 09:25:59.572869 | 94,704.940 ms |
| 33.243.17.139:27414 | 455 | 09:25:52.611456 | 101,666.353 ms |

For both raw records, `(schedule_time_us - workload_report_receive_time_us)
/ 1000` exactly equals the reported age. This also matches source revision
`648100e5e7`, `NaviSchedScheduleNode::getWorkloadReportAgeMs`: it uses the
context schedule time, not the later prediction logging time. The report age
does not include the target's subsequent 45-second result wait. Source-to-loaded
binary identity for the historical interval is still not established by this
arithmetic agreement.

The archived population contains six selected batches with result wait above
40 seconds but report age below 100 ms. Schedule 46512, Agent
215.78.190.162:28034, has 110,159.322 ms result wait and 54.192 ms report age.
Conversely, 324 selected batches have result wait below 100 ms and report age
above 60 seconds: schedule 39535, Agent 215.91.131.155:31604, has 26.961 ms
result wait and 414,887.449 ms report age. These are selected batches, not unique
rounds or frontend requests. Stale reports alone are therefore not a sufficient
explanation for the slow-result population; long result wait also occurs with
fresh input reports.

Assuming time compression one, reconstructing receive time from each archived
context and its report age gives the same millisecond-rounded receive timestamp
across 1,506 selected samples for 33.6.167.237:25954, spanning 197,458.497 ms
of context creation times. For 33.243.17.139:27414 there are 2,560 samples
spanning 205,377.279 ms. The exact target raw records independently anchor
these two inferred receive times. These observations do not distinguish Agent
emission stopping, update transport/collector delay, or reusing an older
snapshot; they are not proof of continuous Agent inactivity or of a lock owner.

All 1,506 selected snapshots of the first plateau also retain model revision
612; all 2,560 of the second retain revision 47. In the overlapping interval,
the fresh-report negative control (schedule 46512) was created at
09:28:38.053981 and contains report age 54.192 ms for another Agent. This
contradicts a complete, global cessation of fresh workload snapshots throughout
that interval. Selective per-Agent/stream delay, Agent-side inactivity, and
per-node contention remain possible. The revision cross-check is retained in
`model-during-workload-plateau.json` alongside the other audit outputs.

Evidence and explicit limitations are in
`/tmp/navi-historical-queue-age-20261029/flow-and-age.json`,
`agent-report-plateaus.json`, and `negative-controls.json`. No production code,
configuration, or traffic was changed for this audit. Same-window update
emission/consumption evidence and CPU ownership remain missing; the historical
incident remains open.

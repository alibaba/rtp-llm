# Master reservation locking redesign

Status: immutable/COW reservation storage, owned replay capture and ordered
discrete-queue evaluation are implemented. The short-commit transaction protocol
below is still a proposal, not an implemented concurrency contract.

The first implementation steps share immutable prefix topology and request lookup
indexes between transaction copies. Reference counts use copy-on-write pages,
with explicit write access and shared read-only access. This does not yet
implement the short-commit protocol or change lock acquisition order.

## Current measured limits, 2026-10-23 afternoon

The historical approximately 45 second incident remains unresolved. Current
measurements identify expensive paths; they do not establish the complete causal
chain for that older time window. The real pending replay runs in tens of
milliseconds offline and does not reproduce the historical queue/lock delay.

Master 34 actually loaded core SHA
`6ec87390517ee448157b37a0446e567fbbc48142dc26c49f28460ce61b9ac486` at PID
879743 on 33.242.7.107. Its graph was running and the platform reported Master
Ready. Its deployment remains publishing because another inference allocation
has no IP. The existing inference process on 215.80.7.184 retained its startup
time and restart count. Do not report the entire deployment complete.

Fresh Web dashboard and actual `/api/query` points were read for 13:43:58 to
13:48:58, app `whale_prod_omega_llm_976371_5613449_DeepSeek-VL`, Master
Deployment All. Row 161 is `Master - 调度耗时与排队时点 / V53 Master`.
Panel 180 is `Master 准备阶段总览（ms）`; panel 181 is
`Master Snapshot 与 Replay 准备（ms）`; panel 183 is
`Master Snapshot 节点锁等待与持有（ms）`; panel 184 is
`Master 整轮与结果处理耗时（ms）`; panel 185 is `Master 结果处理分段（ms）`.

| Curve / phase | Mean of returned time buckets | Maximum returned time bucket |
| --- | --- | --- |
| Preparation, `schedule_prepare_ms` | 168.765 ms | 1577.458 ms |
| Replay preparation, `schedule_replay_prepare_ms` | 128.522 ms | 1510.922 ms |
| Cost preparation, `cost_prepare_ms` | 23.629 ms | 43.292 ms |
| Snapshot, `schedule_snapshot_ms` | 38.264 ms | 155.257 ms |
| `Snapshot 等待节点状态锁` | 36.225 ms | 178.846 ms |
| `Snapshot 持有节点状态锁` | 1.105 ms | 4.074 ms |
| `结果处理耗时` | 48.021 ms | 94.788 ms |
| `整轮处理耗时` | 278.388 ms | 1883.841 ms |
| `Reservation 创建与审计` | 27.387 ms | 44.282 ms |
| `失败投递回收` | 15.084 ms | 55.710 ms |

Queries use aggregator sum and downsample avg. These are not per-round maxima or
percentiles. Sample counts differ, so independently averaged curves cannot be
added as exact round totals. Node count and traffic changed between releases;
these windows do not measure an isolated online speedup. Raw Web text, requests
and responses are in `/tmp/navi-master-chain-34-20261023/`.

A 15 second Master-34 CPU profile contains 483 samples and zero lost samples.
`getMatchedBlockCountLocked` has 66 self samples. Its inclusive call paths include
28 replay, 31 cost-preparation and 8 failure-cleanup samples; inclusive counts
must not be added to self percentages. Thirty self samples in vector assignment
come from copying static hash sets in `capturePendingReplaySnapshot`; forward
iterator assignment traverses the hash set once for distance and again to copy.
The implementation now reserves the known size and traverses once.
Compression samples belong to `navi_gzip_log` / `FileAppender::compressHook`, not
to the cost optimizer or an unknown kernel loop. CPU work from log compression
competes for resources even though it runs in a separate thread.

## Replay ownership and lock boundary, implemented in Master 34

`NodeState::PendingReplaySnapshot` owns static hash vectors, COW reservation
copies and a typed `MasterPendingCacheR::ReplaySnapshot`. Capture takes the
existing runtime -> cache -> pending lock order and copies one coherent node
cut. JSON conversion happens after these locks are released. The captured
encoder contains no live node handle and acquires no state locks. No mutex,
worker queue or Agent protocol was added. Static/pending copies and JSON CPU
still exist; moving encoding does not make replay free.

The real 512-request fixture has 232 waiting requests and 33327 pending hashes.
An instrumented concurrency test retires all waiting requests during encoding,
checks exact drain to zero, restores the captured batch and pending state, and
checks that the captured state remains unchanged. Encoding also runs while the
test owns runtime, cache and pending write locks. Initial Release examples show
66-202 us capture and 14.2-17.8 ms encoding with empty static cache sets. These
figures must not be extrapolated to large production static caches.

## Ordered discrete queues and shared cost request view

Discrete best-move search previously reset six arrays for all N nodes and scanned
all R requests for every proposed move. It now builds ordered request membership
once per refinement call. A proposal virtually removes the request from its
source queue and merges it into the destination queue in original request order.
Only those two nodes are reset and evaluated. After accepting a move, the two
membership lists are updated with lower_bound/erase/insert; no sort is introduced.
The original-order aggregation is necessary for capacity promotion and identical
floating-point sum order. Full evaluation still validates every assignment and
resets all nodes. The final full objective remains the acceptance guard.

Candidate work changes from O(N + R) to O(R_source + R_target), plus constant
two-node model evaluation. Full initialization and final validation remain
O(N + R). A concentrated source still requires scanning its ordered requests.
The optimizer's loop count, learning rates, residual settings, time budget and
4096-evaluation cap are unchanged. Faster execution under a wall-time cap can
evaluate more proposals; replay with a fixed evaluation count stays deterministic.

Same clang / old ABI / Release build, fixed 4096 proposals, eight measured
repeats after warmup, synthetic mixed queues (half the requests on one node):

| Nodes / requests | Original median | Optimized median, first run | Restored median |
| --- | --- | --- | --- |
| 24 / 128 | 6596 us | 5831 us | 6033 us |
| 24 / 512 | 14817 us | 13066 us | 13726 us |
| 432 / 128 | 9958 us | 4853 us | 4995 us |
| 432 / 512 | 17688 us | 11893 us | 12407 us |

The regression requires an unrelated node's six workspace values to survive a
pair evaluation. It fails before the change, passes after the change, fails after
temporarily removing it, and passes after restoration. Independent full-node
objective tests enumerate 512 assignments with mixed capacity, two/three nodes,
13/4096 evaluation limits, and exact accepted-move and proposal counts.

KV cost preparation also used to copy the selected request array separately for
each node. An identity view now passes the original array. A filtered/permuted
view owns one array built at activation, shared read-only by all node workers.
Clear occurs after worker completion and releases the extra request references.
Invalid views do not publish a partially prepared array. No new lock was added.

Isolation benchmark, 432 mock nodes, including view activation/clear and all bulk
calls, eight measured repeats; this excludes real KV scans and parallel workers:

| Requests / view | Previous median | Shared view median |
| --- | --- | --- |
| 128 / identity | 1607 us | 57 us |
| 128 / half selected | 807 us | 32 us |
| 512 / identity | 6715 us | 219 us |
| 512 / half selected | 3363 us | 118 us |

These are component CPU measurements, not an end-to-end throughput multiplier.
The identity-array regression fails on the old implementation, passes after the
change, fails after removing it, and passes after restoration. A ten-thread test
checks exact reordered node/request mappings, shared-vector identity, query
counts, duplicate rejection and reference release. Its first TSAN run caught a
test-fixture race: expected node pointers were initialized while already-started
threads could access reordered nodes. Initialization now completes before any
thread starts; the original diagnostic remains in the evidence directory.

Artifacts are in `/tmp/navi-discrete-queues-20261023/` and
`/tmp/navi-shared-cost-view-20261023/`. The combined Release run passes 105
selected cases. Later validation and deployment must be recorded separately;
local implementation is not evidence that a production image contains it.

## Evidence and scope

The current node `_runtimeLock` protects runtime reports, the reservation deque,
request transitions and dispatch waiters. Its critical sections also perform
reservation copies, prefix rebuilding, cache queries and pending-index updates.
The lock is per Agent, not global across Agents.

On Master version 31, schedule 6600 at 2026-10-23 12:26:52 recorded snapshot
148059 us, cumulative runtime-lock wait 147882 us and cumulative hold 93 us.
A later 0.5 second futex sample recorded 252 paired waits in
`registerDispatchWaiter` (maximum 62961 us) and 11 in `reportAgent` (maximum
53505 us). Registration, cancellation, abandonment and Agent updates used the
same contended mutex address. These observations establish contention in this
window, not the cause of the historical 45 second incident.

Raw evidence is in `/tmp/navi-snapshot-lock-20261023/current-pressure.log` and
`futex-current-paired.json`. Source anchors are `NodeState::getScheduleSnapshot`,
`failRequestsLocked`, `cancelDispatchWaiter`, `abandonUnownedDispatch` and
`MasterStateR::reportAgent` in `resource/MasterStateR.cpp`.

## Required invariants

1. A committed revision contains consistent request states, reservation block
   ownership and inflight/workload counters. Readers must not combine revisions.
2. Agent cache updates precede the corresponding completion update. Preserve
   the order guaranteed by Navi edges; do not independently enqueue cache and
   completion work in a way that reverses application order.
3. Cancellation and admission have one serialized decision for each request.
   An acknowledged request cannot be reclaimed as unowned. An expired waiter is
   distinct from a request whose waiter has not yet been registered.
4. Agent reset changes a generation. Work prepared for the old generation cannot
   commit to the replacement Agent, even if its address and request ID repeat.
5. A failed update leaves the previous committed state intact. Notify pipes only
   after a successful commit and after releasing every state lock.
6. Pending counts reaching zero must correspond to actual removal of ownership
   entries; aggregate counters alone are not proof of reclamation.

## Ownership and locking

| Object | Ownership / synchronization | Work allowed while locked |
| --- | --- | --- |
| Agent registry | Existing registry lock; return a shared node handle and generation | Find/add/remove handles; no cache, batch, formatting or callback work |
| Batch definition | Immutable shared object: ordered requests, request lookup and prefix topology | No lock on reads; no whole-object copy per event |
| Batch transaction | Per-batch serialized writer; mutable request states and prefix reference deltas | Only writers for this batch serialize; acquisition must not block a Navi worker behind long computation |
| Node commit metadata | Short node commit mutex | Check generation/revisions, validate prepared delta, install pointers and aggregate deltas |
| Waiter entry | Request state and waiter ownership participate in the same short commit decision | Register/detach/claim one waiter; never rebuild pending or query KV |
| Published scheduling view | Atomically acquired immutable revision | Read without the node commit mutex; no reservation scan |
| Cache/pending lookup shards | Versioned read views; update only affected owners | Lookup / bounded shard delta; no full snapshot construction under a write lock |

The immutable batch definition does not include reference counts or request
states. Sharing those mutable fields would make rollback and independent
transactions incorrect. Old objects are released outside locks; swapping a
shared pointer inside a lock must not destroy its last large object there.

The registry lock must not be held while acquiring batch or cache locks. A node
handle's generation replaces the current long registry-lock lifetime around
cache application. A batch writer must never acquire the registry lock.

If the commit needs pending shard locks, take them in a fixed shard-ID order
under the short commit lock. A cache reader never takes the node commit lock.
No shard lock may be held across KV scans, serialization, allocation of a full
replacement index, pipe notification or waiting for a batch writer.

## Transaction protocol

1. Resolve node and batch handles; release the registry lock. Capture generation
   and the relevant committed batch/cache revisions.
2. Serialize mutation of this batch. Use Navi continuation / AsyncPipe when
   another batch transaction is running; do not occupy worker threads with long
   mutex waits or spin retries. Keep actual registration/admission ownership
   decisions at the short commit point, not in a separate unsynchronized map.
3. Prepare request transitions, block-owner deltas, workload deltas and any
   required predictions outside the node commit lock. Allocate and validate all
   potentially failing data before publication.
4. Take the node commit lock and affected shard locks. Recheck generation,
   lifecycle preconditions and every input revision on which the prepared
   result depends. Publish the batch, pending deltas and scheduling aggregates
   as one revision. The commit must be all-or-nothing; merely moving the existing
   `updateScheduled` call outside `_runtimeLock` is incorrect.
5. Release locks, destroy replaced data and notify detached pipes.

A revision conflict cannot discard a completion event or publish an old result.
Suspend/recompute the transaction against the next input view. Count conflicts
and test sustained cache churn. Avoid a tight unlimited retry loop. The writer
continuation and its queued event ownership must be explicitly bounded by live
requests / batches with backpressure; an unbounded replacement queue is not an
acceptable implementation. This liveness protocol must be implemented and
tested before replacing the current synchronous update contract.

## Reduce work rather than only move it

- Use batch-ID and request-ID lookup indexes. Preserve duplicate-ID semantics
  across batches and generation boundaries; do not assume global uniqueness.
- Request terminal transitions walk the affected request's prefix path and
  update reference counts. A block-owner delta is emitted only on an ownership
  boundary change. No whole-prefix-tree clone, sort or scan for every failure.
- Maintain reservation and inflight aggregate deltas at commit. Snapshot reads
  acquire one published view instead of visiting every retained reservation.
- Capacity-split features require original request order and promotion from
  waiting to running. Simple subtraction of aggregate sums is insufficient.
  Use an ordered aggregate index, with a scalar reference implementation in
  tests; do not change the model or running-slot boundary to simplify locking.
- Coalesce terminal events already pending for the same batch into one
  transaction. Never wait an extra timer to accumulate them, merge across a
  cache/ack ordering barrier, or infer failure for an unregistered live request.
- Cache-hit predictions carry the versions they were computed from. Recompute
  only when a dependency changes. In particular, an excluding-own-batch query
  must distinguish own-batch removal from changes to other cache owners.
- Build full cache replacements and replay payloads outside state locks. Capture
  coherent immutable handles/revisions first. A full snapshot must not copy the
  entire index on every request update as a substitute for concurrency control.

## Validation and rollout gates

First retain the existing synchronous implementation as the behavioral oracle.
Reproduce the affected failure/timeout cleanup using the real 512-request,
73544-prefix-node fixture, including mixed failed/waiting requests.

Required deterministic interleavings: cancel versus ack; cache update before
completion; duplicate and late events; reset during preparation; revision conflict
at commit; pending-index validation failure; last shared-owner deletion; expired
waiter versus not-yet-registered waiter; replay capture during updates.

Compare all request states, block-owner sets, inflight sums, capacity-split
features and pipe notification counts. Prove old behavior fails the intended
performance/operation-count regression, fixed behavior passes, removing the fix
fails again, and restoration passes affected tests. Timing assertions alone are
not stable correctness gates.

Benchmark healthy, simultaneous-timeout and stalled-Agent traffic separately.
Record per-operation lock wait/hold, copied bytes, scanned prefixes, KV lookups,
commit conflicts, queued events, oldest unpublished revision age and memory.
Report distributions and wall-clock drain rate, not only summed thread waits.
Evaluate several retained-batch sizes with identical compiler/ABI/optimization
flags. Verify final drain to zero and exact cache ownership against the oracle.

Roll out in stages: immutable definition/index and observability; incremental
terminal deltas; atomic publication / short commit; removal of old long locking.
Each stage needs its own tested artifact and exact rollback digest. The historical
45 second incident remains open until same-window logs, state and replay support
its full causal chain; a successful lock microbenchmark does not close it.

## First-step validation, 2026-10-23

Local clang / old ABI / Release fixture measurements (not end-to-end throughput):

| Measurement | Before | Shared topology |
| --- | --- | --- |
| 128 reservation copies, including destruction | 24546 us; reverse check 25110 us | 7077 us |
| Per-copy mutable prefix array | 1765056 bytes | 588352 bytes |
| Retirement scan visits for 232 waiting requests | 3946577 | 3946577 |

Retirement scan visits stay unchanged: this step removes topology copies and does
not claim to remove prefix rebuilding. The immutable topology allocation is
shared, not eliminated. The new reference-count storage cannot be shared across
independent transaction copies.

The regression fails on the old implementation, passes after the change, fails
after temporarily removing the change, and passes after restoration. Independent
copies retire the real fixture concurrently in four threads with different
request orders; the original keeps 232 waiting requests and 33327 pending hashes,
while each copy ends with 512 failed requests and zero pending/inflight requests.

Validation artifacts in `/tmp/navi-prefix-topology-20261023/`:

- `before-test.log`, `fixed-valid-test.log`, `reverse-test.log`: regression and
  timings; 38 batch cases pass in the initial full Release run.
- `restored-role-test.log`: 315 selected role/cache/lifecycle cases pass after
  restoration, including admission/cancellation racing and atomic snapshot tests.
- `copies-run_*_of_5.log`: concurrent independent-copy case passes five runs.
- `concurrent-copies-tsan-test.log`, `batch-all-tsan-test.log`: instrumented
  concurrency case and full 39-case batch suite pass without a reported TSAN
  race. The test binary contains `__tsan_init` and TSAN memory-access hooks.
- `role-live-gdb*.log`: two full-thread captures during the broader test run;
  the process progressed between Agent shutdown cases and ultimately passed.

These are bounded test results, not a proof that all production interleavings are
deadlock-free. No new lock or new lock acquisition order is introduced in this
step. The later lock-boundary change still requires its own interleaving,
backpressure, conflict and production validation.

## Reference pages and result-stage visibility

The next implementation stores reference counts in 256-node pages. A transaction
copies the page directory and shares the pages. `writable(index)` detaches only a
shared page before mutation. Read access never detaches a page. Existing node
serialization remains necessary: concurrent mutation of one reservation object
is not supported. Separate transaction copies can mutate independently.

The real fixture has 73544 nodes. Retiring its first waiting request detaches
768 nodes (6144 bytes of reference counts); 72776 nodes remain shared. This is
not the complete transaction allocation count: request entries, pending hash
vectors, the live scan index and the page directory still have copy costs.

Same compiler / old ABI / Release examples, with frame-pointer settings held
constant within each comparison:

| Operation | Contiguous references, reverse check | Paged references, restored |
| --- | --- | --- |
| 128 reservation copies | 7122 us | 3585 us |
| 232 copy/fail/commit transactions, real fixture | 24930 us | 19888 us |
| Shared nodes after one failure | 0 | 72776 |

These are single-run microbenchmarks, not percentile or production speedup
claims. The failure transaction benchmark omits the NodeState cache-match
callback, pending-index publication and runtime-lock competition. It still
performs the existing pending-vector rebuild and waiting-feature scan.

Artifacts: `/tmp/navi-reference-pages-20261023/`. The new sharing regression
fails before the change, passes after it, fails after temporarily restoring
contiguous storage, and passes after restoration. The batch suite passes 41
cases with TSAN and no reported TSAN warning. The final affected role suite
passes 316 cases, including result timing on response delivery failure and
on result preparation failure. These tests do not prove universal liveness.

The existing `schedule_compute_ms` timestamp ends when the result kernel starts.
The round lease remains held through `finishContext`, so dividing 512 by this
metric alone overestimates service capacity. The implementation now measures:

| Result phase | Included work | Metric suffix under `navi_sched.master` |
| --- | --- | --- |
| Prepare | Selected-node batch predictions and decision construction | `schedule_result_prepare_ms` |
| Audit | Admission prediction logging, globals attachment, request tracking release | `schedule_result_audit_ms` |
| Reserve | Cache-decision audits, reservation construction and publication | `schedule_result_reserve_ms` |
| Report / replay | Prediction metrics and replay audit metadata | `schedule_result_report_replay_ms` |
| Delivery | Response pipe publication via `context.finish()` | `schedule_result_delivery_ms` |
| Replay log | Optional replay JSON construction and logging | `schedule_result_replay_log_ms` |
| Failure cleanup | Group failed response deliveries and reclaim reservations | `schedule_result_failure_cleanup_ms` |
| Finish body | Result entry through cleanup, including local destruction | `schedule_result_finish_ms` |
| Round body | Activation through result cleanup | `schedule_round_total_ms` |

The last two stop before the timing metrics/log itself and final lease release.
They do not include queue residence before activation or Agent execution. An
unentered phase stays -1 and produces no sample; completed failure paths still
report their measured phases. The timing log reports the first nine rounds,
every hundredth round, and result handlers taking at least 100 ms. Dashboard
publication and actual Web visibility are separate runtime checks.

## Scheduling chain and concurrency audit

| Stage | Current execution | Next candidate and correctness dependency |
| --- | --- | --- |
| Dequeue / admission | Serial round owner; expired requests are retired first | Preserve FIFO, cancellation and one round lease; overlapping rounds require a coherent reservation view |
| Node snapshots | Serial visits; each reads one node runtime under its lock | Publish immutable aggregate views; bounded node tasks are possible, but cannot fix long writers or mix revisions |
| Optional replay capture | Serial full pending serialization and KV matching | Capture immutable handles, then serialize outside locks; retain collection gating and bounded capture size |
| Cost preparation | Node ranges across worker kernels; each node scans its requests in one read view | Use immutable cache views to allow request tiles without extending cache writer starvation; count contention before increasing workers |
| Continuous optimization | Node ranges for forward/backward, request ranges for projection | Retain barriers between gradient, projection and the next iteration |
| Discrete refinement | Serial best-move search; each candidate rescans two ordered queues | Incremental ordered aggregates first; parallel proposals need isolated assignments/workspaces, deterministic tie breaking and a common baseline |
| Result preparation | Serial across selected nodes; each batch is independent before publication | Parallel node predictions with a barrier before decisions and reservations |
| Reservation creation | Serial across selected nodes; batch initialization occurs before runtime lock | Parallel node transactions after registry/generation checks; preserve request/pending atomic publication |
| Failure / cancellation | Per-node runtime lock; per-request callbacks can repeat batch work | Explicit cancellation intent plus bounded per-batch processing; preserve ack ordering and terminal ownership |

## Full-path review and remaining work

Notation: Q is queued requests, R the current round's requests, N candidate
nodes, B retained reservations on one node, K retained prefix nodes, H request
hashes examined. The following are source-confirmed work patterns, not measured
attribution to the historical incident. Items without an experiment remain
candidates.

| Path | Current avoidable or scaling work | Proposed experiment / required invariant |
| --- | --- | --- |
| `MasterScheduleR::submit` | Resource read lock spans AsyncPipe setData and wake; one wake per submit | Measure lock hold and wake rate; releasing the lock needs a pipe-generation/stop linearization contract, not just a copied pipe pointer |
| `MasterScheduleContextKernel::drainRequests` | Drains until empty without a per-activation bound | Test sustained producers and event-loop fairness; a bounded drain must rearm remaining data, preserve EOF/stop draining, and not strand requests |
| Pending retirement | Full O(Q) remove_if on every activation, including a busy round | Add explicit cancellation notification/deadline indexing or a bounded cursor; preserve prompt cancellation, exact counters and expired-before-KV rejection |
| Schedule timer | A second O(Q) minimum-deadline scan | Maintain an exact earliest deadline with lazy removal; enqueue times may arrive out of order, and request timeout is distinct from batch wait |
| Snapshot | Visits N nodes serially, locks each runtime, and aggregates B reservations | Publish revision-consistent aggregate state or use bounded node tasks; do not read atomics for individually related fields and call the mixture coherent |
| Snapshot capacity split | Can scan waiting requests in retained batches to assign running/waiting slots | An ordered aggregate index must retain promotion order; simple subtraction is incorrect |
| Cold-start/admission analysis | Routing plan is built once for queue admission and again for cost routing | Cache only immutable analysis with request-set identity; admission can compact/reject requests between the calls |
| Unsafe marginal check | Negative waiting coefficients can cause serial per-request KV queries inside routing analysis | Measure model/traffic frequency; bulk/tiled queries need the same coherent cache view and must retain rejection semantics |
| Cost view | Previously O(NR) request copies | Implemented: identity reuse or one O(R) filtered array, shared after publication and cleared after the worker barrier |
| Replay capture | Serial node captures, static/pending copies, per-request/node KV matches and JSON allocation | Explicit collection only; encoding outside live locks and single traversal of static sets. The parallel-capture experiment is deferred; normal scheduling does not create trace objects or arrays |
| Cost matrix | Two float matrices plus weights and worker gradients allocated per round | Evaluate bounded reusable buffers per graph owner; an old generation cannot return a still-used buffer to a new round |
| KV matching | Node ranges parallel; requests within one node share its read view | Shared request-prefix traversal could avoid duplicate probes; build immutable topology once, validate hybrid tail/excluded-own-batch behavior and count unique versus repeated prefixes before adopting it |
| Continuous forward/backward | O(NR) per iteration, capacity-split ordered scans, barriers between phases | Reuse immutable per-node model terms; retain nonlinear objective, workload decay, gradients, model revisions, and the full configured iteration opportunity |
| Projection | Request tiles parallel, with node columns accessed across a row-major matrix | Measure cache misses before adding a transpose; account for transpose bandwidth/allocation and preserve simplex sums and deterministic tie breaking |
| Discrete refinement | Was O(N+R) for every proposal | Implemented ordered two-queue evaluation. Parallel proposals require independent workspaces, a frozen assignment baseline, deterministic best-move reduction and a commit barrier |
| Result preparation | Serial node batch estimates and repeated selected-request KV queries | Parallel selected-node estimates can join before decisions; reusing a match requires an explicit cache-version contract |
| Reservation creation | Registry read lock spans cache audits, a second batch estimate, prefix/pending construction and publication | Prepare outside registry lock with node-generation validation at commit. The second estimate is discarded in unified mode, but the live mode can change; do not blindly reuse a snapshot-mode value |
| Admission globals | Selected request loop repeatedly looks up shared model globals for the same collector | Freeze one valid model-global view per collector; test missing globals and concurrent model publication |
| Dispatch lookup | `findNodeState` copies all node handles and scans them for each register/cancel/failure | Add an exact current-generation target index; handle address/biz changes, duplicate addresses and replacement, with bounded ownership and no reverse lock edge |
| Reservation lookup | Linear B search under runtime lock for most events | Stable batch index must survive deque erasure/rollback; request IDs and batch IDs cannot be treated as globally unique |
| Failure mutation | Copy batch, retire prefixes, scan K live nodes, query surviving waiting requests, publish full pending vectors while runtime is held | Incremental block-owner deltas and cache-versioned waiting features; own-batch removal must not falsely invalidate an excluding-own-batch match, while changes to other owners must invalidate it |
| Agent pending full snapshot | Builds replacement under a write lock and copies/sorts current and candidate states even for a different revision | Build outside with revision validation; exact state comparison is only needed at the same revision and cannot be replaced by checksum equality alone |
| Agent cache/completion | Serialized Navi message application with rollback and waiter notification | Preserve cache-before-completion order. Any off-thread prepare must join/commit before the next dependent event can overtake it |
| Metrics/summaries | Reservation/waiter scans and JSON/string formatting can acquire the same node locks as scheduling | Capture small coherent summaries first, format outside; bound summary frequency and prove counters equal actual ownership |
| Audit/log compression | Extra KV probes for cache audits, large INFO records, background gzip CPU | Measure bytes/s and audit cost separately. Sample successful diagnostics only with explicit counters and retained failure evidence; do not silently remove incident evidence |

The immediately safe concurrency boundary is independent node computation on
owned snapshots, followed by an explicit join before publication. The unsafe
boundary is the middle of a node's cache/reservation/ACK transaction. Adding
threads there without a generation/revision/ownership protocol can create stale
commits or reverse the Agent's guaranteed cache/completion order.

Two plausible feedback mechanisms still need a controlled whole-path test:
expensive replay on the round owner delays all queued requests; their client
timeouts then trigger per-request failure recomputation under node locks, which
delays subsequent snapshots. Separately, O(Q) work on every request wake can grow
with queue length while reducing the round owner's service rate. The existing
measurements show the constituent costs, not a completed causal proof or a
reproduction of 45 seconds. Do not close the incident on this basis.

Current validation after restoring both tested optimizations: 233 selected
Master/cost/reservation/cache/dispatch tests pass in Release. Fifteen selected
TSAN cases pass without a reported warning after correcting the fixture race;
the binary contains `__tsan_init` and TSAN memory-access hooks. These are bounded
tests, not universal deadlock or liveness proofs. Production rollout and
same-window drain measurements remain separate validation gates.

Two repeated-query candidates remain unchanged: `prepareResult` predicts a
selected batch, then `reserveScheduleBatches` predicts it again; cache-decision
INFO diagnostics also query KV independently. Reuse requires a defined revision
and freshness contract. The current unified reservation path resets its additive
prediction to zero, so the second prediction is particularly suspect there.
Measure result reserve time before claiming its end-to-end contribution.

An expired-waiter-only coalescing proposal was rejected as a solution for the
observed destructor contention. `MasterDispatchGuardKernel` retains its strong
ack pipe while synchronously cancelling; its weak waiter normally remains live
while blocked on the runtime lock. Expired weak pointers cannot identify all
those pending cancellations. No such coalescing change has been implemented.

At 12:56:31, Master 32 schedule 1200 records 89.511 ms snapshot time, 88.579 ms
runtime-lock wait and 0.450 ms hold. The 15-second CPU sample beginning at epoch
1792731674 contains 465 samples with zero loss; KV matching has 49 self samples
(10.54%). Inclusive KV call chains include 20 replay-capture samples, 20 cost
preparation samples, 7 failure-recalculation samples, 6 result-preparation samples
and 1 reservation sample. These inclusive counts are not independent time totals
and cannot be summed with self percentages.

The 0.5-second futex capture lost three chunks and is not a reliable complete
latency distribution. A second 0.2-second capture used a larger ring and had no
reported loss. Preserve both artifacts and do not silently discard the loss.
Changed node counts and ongoing scaling prevent an online before/after speedup
claim for these releases. The historical 45-second attribution remains open.

## Master 35 runtime and follow-up changes

Master 35 was published at 14:07. The container process PID 887198 loaded core
SHA-256 `9a3981aaf0f6c21ea328458819c0b95f602f065b8abdfba76ed7d8ac112ef5e8`;
the platform subsequently reported Ready / HT_ALIVE / WT_READY / SVT_AVAILABLE.
The inference target comparison was empty and the existing Python worker's
restart count (3) and start time (1792729352000000) were unchanged. One separate
inference allocation remained without an IP, so the deployment as a whole was
still publishing. The guarded publish initially rejected an externally changed
deployment. A fresh read showed only operation metadata and inference autoscale
minimum changing from 100 to 20; the current value 20 was preserved.

Fresh Web and actual query responses for 14:10:21--14:15:21 are archived in
`/tmp/navi-master-chain-35-20261023`. The full Web page confirms the Row
`Master - 调度耗时与排队时点 / V53 Master`; panel 183 is
`Master Snapshot 节点锁等待与持有（ms）`. Curves
`Snapshot 等待节点状态锁` and `Snapshot 持有节点状态锁` have time-bucket
means of 38.816 and 0.724 ms respectively. Panel 181,
`Master Snapshot 与 Replay 准备（ms）`, curve `Replay 准备耗时` has mean
18.347 ms and maximum returned bucket 654.091 ms. Panel 185,
`Master 结果处理分段（ms）`, curves `Reservation 创建与审计` and
`失败投递回收` have means 31.312 and 14.263 ms. Queries use Hippo App
`whale_prod_omega_llm_976371_5613449_DeepSeek-VL`, Master Deployment All,
aggregator sum and downsample avg. These are not per-round p95s or maxima.
The node population changed from over 600 to about 232, so these values do not
establish an isolated release speedup.

The 15-second Master 35 CPU capture has 275 samples and no reported loss. KV
matching has 45 self samples (16.36%); reservation pending rebuilding has 19
(6.91%). Of 53 inclusive KV stacks, 41 contain `failRequestsLocked`, 11 contain
cost preparation, and one contains result finishing/reservation work. Inclusive
and self counts have different meanings. At 14:16:15, schedule 1904 reports
512 requests, 236 failed deliveries, 85.800 ms failure cleanup and 209.219 ms
total round time, with replay collection disabled. This is direct evidence of
remaining expensive cleanup; it does not reproduce or explain all 45 seconds.

Pending full snapshots compare exact definitions only at the same accepted
Agent revision, using the existing batch index instead of copying and sorting
both full snapshots. Generated changes and forced checksum collisions are
covered by an independent sorted-snapshot oracle. Eight comparisons measured
9.530 ms for the reference versus 1.948 ms for the indexed version.

The optional ten-way Replay capture experiment was removed from the release
scope following the user's direction. It is not present in the current source;
the unshipped intermediate wheel must not be deployed. Normal scheduling does
not construct ReplayInfo, trace arrays, random-decision records or scheduler
metadata maps. Capturing pending/static state and extra KV queries still require
an explicit collectReplay request. Branch checks for collection remain.

## Pending exclusion owner lookup (2026-10-23, in validation)

The old exclusion path visited every child batch belonging to the excluded
schedule and binary-searched its hash vector for each queried block. The new
path visits the already indexed owners of that block. A single owner needs one
reservation lookup. Shared owners use a cardinality shortcut when they outnumber
excluded batches; otherwise only actual owners are checked. No new index,
publication path or lock is introduced, and both full/tail ownership keep the
same schedule exclusion semantics.

For 16,384 full/tail queries, thread CPU median after warmup:

| Layout | Old | Fixed | Component speedup |
|---|---:|---:|---:|
| One child batch, 128 hashes | 0.365 ms | 0.108 ms | 3.4x |
| 128 child batches of one schedule, 128 hashes each | 44.859 ms | 0.132 ms | 340.6x |

Restoring the old implementation measured 44.871 ms for the split case and
failed the performance regression again; its independent reservation oracle
passed. The fixed pending suite passed before that reversal. Final affected
validation and rollout results are recorded separately. This synthetic split
case does not establish online speedup or the cause of historical 45 seconds.
Evidence: /tmp/navi-pending-owner-20261023/{old,fixed,reverse}-test.xml.

## Incremental failure accounting (2026-10-23, validation and rollout in progress)

Dispatch failure used to re-query every surviving waiting request under the
node runtime lock and static-cache read lock. A cache change between scheduling
and failure replaced stored cost features during cleanup. A regression with
three requests reproduced stored survivor hit/miss totals changing from 16/48
to 64/0 solely because unrelated live cache was populated before failure.
This demonstrates accounting revision mixing; it does not establish that an
online negative aggregate was observed.

The current fix removes the CacheMatch argument from bulk markFailed entirely.
Request hit/miss inputs already reside in each request entry; their feature
contributions are subtracted through the existing validated subtract operation.
Single and bulk failure share this operation. The last waiting request clears
its batch contribution. Bulk IDs use the existing immutable request index,
validate unknown IDs before mutation, deduplicate failures, and rebuild pending
ownership once. Master dispatch cancellation and Agent batch-abort handling no
longer re-query surviving requests. The outer NodeState candidate transaction
still publishes only after prediction and pending publication succeed; no new
lock or lock-order edge is added.

Initial affected validation: 263 Master cases plus 40 reservation cases passed.
Restoring the live KV refresh made the survivor-cost regression fail again.
Final restored validation and release evidence are kept under
/tmp/navi-failure-delta-20261023.

Thirty sequential production-API failures (same generated input) measured:

| Agent pending batches | Before incremental accounting | Fixed | Speedup |
|---|---:|---:|---:|
| 8 | 1.312 ms thread CPU | 0.325 ms | 4.0x |
| 2450 | 1.484 ms thread CPU | 0.314 ms | 4.7x |

These are single fixed-work samples from the test's existing CPU timer, not
percentiles or a controlled online throughput experiment. Remaining full
pending-vector rebuilds and batch candidate copies are included in both.
The true 512-request/232-waiter incident fixture completed its failure
transactions in 20.772 ms in the fixed standalone test; no comparable old value
has yet been collected for that measurement.

Additional production path findings from the current audit:

- reserveScheduleBatches queried static/pending cache again only to emit an INFO
  cache-decision audit per selected request. This scan is now explicitly gated
  by collection; required reservation publication remains unchanged.
- getScheduleSnapshot scans retained reservations and, in capacity-split mode,
  visits waiting request entries twice for the workload and reservation views.
  Their running-slot counts differ, so blindly reusing either aggregate would
  be wrong. An ordered contribution index is a candidate, not implemented.
- markAdmitted and successful completion still rebuild waiting features, even
  though these paths can potentially debit the entries changing state. Their
  promotion/capacity semantics and model refresh rules need separate tests.
- NodeState still linearly locates batch reservations; failure candidate copies
  still copy request entries. Prefix topology is shared and reference pages
  use copy-on-write, but a full pending vector is materialized and upserted.
  The next experiment must count removed/added block owners rather than assume
  the owner-query fix makes pending publication incremental.
- Result preparation queries chosen-node matches after cost preparation has
  already queried the corresponding node/request matrix. Reuse needs a named
  immutable scheduling input owned by the round, not an unconditional Replay.
- Queue admission and cold-start routing separately analyze the same node set;
  requests may compact between them. Reuse must preserve the request-set and
  cache/model revision assumptions rather than cache a mutable routing result.

Master36 is actually live with core SHA 6d14972e193ee1d6284b23252e75e435305e4b0fae44d4ec48321588c3e4db86.
Fresh status returned Ready/HT_ALIVE/WT_READY/SVT_AVAILABLE. Its 15-second perf
at epoch 1792738376 still measured 17.95% self time for KV matching, including
27 failure-cleanup stacks. It does not yet contain incremental failure accounting.
This is evidence that faster exclusion queries alone do not eliminate repeated
cleanup work. Agent targets compare equal and the existing Python worker start
time/restart count are unchanged. The deployment still has an unrelated
unallocated inference slot; full deployment completion is not claimed.

## 2026-10-27: full Agent snapshot preparation outside the node write lock

`MasterPendingCacheR::applyAgentUpdate` previously built the incoming full
snapshot's reservation and block-owner indexes while holding `NodeCache::lock`
for writing. The input-only preparation now runs before acquiring that write
lock. A preliminary read checks for an already stale snapshot; the authoritative
staleness, metadata, checksum and same-revision checks remain under the write
lock. If a concurrent reset makes the preliminary stale snapshot eligible, the
old locked preparation path remains as a fallback. Delta updates do not allocate
a prepared snapshot. On successful full replacement, swapping the candidate with
the old Agent state defers destruction of the old indexes until after unlock.
Overlap recomputation and state publication remain locked and atomic to readers.

Evidence is under `/tmp/navi-pending-snapshot-lock-20261027`. An external GDB
probe follows pthread write-lock/unlock calls in `applyAgentUpdate` and counts
`PendingCacheState::replace` calls in that stack. For the existing stale-snapshot
regression test, the four-stage results are:

| Source state | Preparation calls | Calls under write lock | Gate |
|---|---:|---:|---|
| Old (`old2.log`) | 2 | 2 | FAIL |
| Fixed (`fixed-test.log`) | 2 | 0 | PASS |
| Fix removed (`removed.log`) | 2 | 2 | FAIL |
| Fix restored (`restored-test.log`) | 2 | 0 | PASS |

The functional test passes in all four stages; the old-stage Bazel failures are
the probe's explicit lock-ownership assertion. The initial `old.log` probe did
not identify the ancestor frame and is invalid evidence. These observations
establish lock placement for the exercised ordinary snapshot path, not a
throughput improvement, a bound on all write-lock holds, or attribution of the
historical 45-second incident. This change has not been published or verified
online. Same-revision comparison, overlap recomputation, invalid-state clearing,
and the concurrent-reset fallback can still perform work under the write lock.

Restored-source affected validation (clang, old ABI, release, Python 3.10;
test cache disabled) completed with Bazel exit 0: role 626/626, reservation
55/55 and cluster 42/42. The role run explicitly excluded
`NaviSchedResourceTest.testConcurrentStopFinishAndAbortReleaseEveryInflightOwnerExactlyOnce`.
That test already failed on the pre-change source during the prior 500-repeat
comparison (`/tmp/navi-capacity-reuse-20261027/stop-race500-comparison.json`);
its Agent stop/finish race remains unresolved. Consequently these results do
not establish that the entire unfiltered role suite is green. Raw XML/logs and
`affected-results.json` are preserved beside the probe evidence.

### Snapshot/KV contention measurements

`master_pending_cache_benchmark --snapshot-contention-replay FILE` now measures
the first replay node with one full-snapshot writer and one continuous KV batch
reader. Each trial applies eight identical-content snapshots with monotonically
increasing revisions. Every 189-request query must reproduce its entire initial
match vector. Each update checks revision, checksum, batch/block/tail statistics,
snapshot readiness and rejection flags. Reader wait comes from the existing
`BatchMatchTiming::lockWaitUs`; it excludes node lookup and matching execution.
Reader shutdown is joined on normal and exceptional exits. Wait samples are
bounded to 100,000; maximum and sum still include every completed query, and
truncation is explicitly reported. None of the accepted trials was truncated.

Evidence: `/tmp/navi-snapshot-contention-20261027/results.json`, `analyze.py`,
individual run logs and Bazel profiles. All runs use clang / old ABI / release /
Python 3.10 and CPU affinity `0,1`. Each phase contains three large-state trials
and one unmodified-state control. The large input clones node 0 of the real
replay to 4,096 scheduled and 4,096 Agent batches. It retains 268 unique full
hashes per source but has 1,097,728 full block references per source. It is a
controlled shared-owner stress state, not the historical incident snapshot.

Three-trial medians for the large input:

| Phase | Maximum query lock wait | Summed query lock wait | Completed KV batches | Writer thread CPU / 8 updates |
|---|---:|---:|---:|---:|
| Before lock change | 277.174 ms | 2143.962 ms | 10 | 2136.175 ms |
| Fixed | 2.254 ms | 7.527 ms | 73,038 | 2053.273 ms |
| Change removed | 256.731 ms | 2002.556 ms | 10 | 1996.930 ms |
| Change restored | 2.331 ms | 9.432 ms | 81,627 | 2290.480 ms |

The final reverse comparison reduces the median of the per-trial maximum wait
by 99.092%. Writer CPU does not show a repeatable reduction: it increases 14.7%
in that last comparison while the reader performs substantially more work.
This removes reader serialization during preparation/retirement, not the cost
of rebuilding the indexes. Completed-query counts are an isolated continuously
querying reader's work, not Master scheduling throughput. Percentiles have very
different sample counts across phases; do not compare them as equal traffic
distributions. The local 10-ms maximum-wait criterion switches fail/pass/fail/pass
in all measured large trials, but is not introduced as a portable CI timing
assertion. The earlier GDB gate independently verifies lock ownership.

The unmodified node has only one Agent and one scheduled pending batch. These
updates are too short to ensure overlap: controls return only 1--10 reader
samples, so their microsecond maxima do not establish latency distributions or
a small-state speedup. This negative control and the amplified reference count
are both necessary limits on interpreting the large-state result. No live
snapshot frequency, historical function CPU attribution, or online benefit is
established by this benchmark. The production source is restored byte-for-byte
to the tested fixed version after the reverse experiment.


## Empty waiting-capacity snapshot fast path, 2026-10-27

`NodeState::getScheduleSnapshot` constructs two capacity views while holding
`_runtimeLock`. Both called `appendWaitingCapacityFeatures` for every retained
reservation even when every request was already dispatched, completed or
failed. The append function now returns after capacity-argument validation when
its authoritative waiting count is zero. Invalid slot arguments still fail;
valid calls leave both features and running slots unchanged. Request state and
waiting count already change under the same node lock. This adds no new cache,
lock, owner or lifecycle transition.

The regression test covers 128 requests in each of three states (dispatched,
completed and failed), preexisting ordinary features, empty features, occupied
slots, zero capacity and invalid slot counts. The old code passes all functional
assertions, but an external GDB work gate detects 3,072 unnecessary request
visits. Across old/fixed/removed/restored phases the 36 function calls remain
constant and request visits are 3,072 / 0 / 3,072 / 0. Functional assertions pass
in all four phases. Evidence is in
`/tmp/navi-empty-waiting-20261027/{old.log,fixed-test.log,removed.log,restored-test.log,results.json}`.

The benchmark's `--empty-waiting-replay FILE` mode restores existing reservation
records and applies both capacity views. It uses the existing bounded replay
reader and retains the workload only in memory; no production replay collection
is enabled. The real 2026-10-24 capture contains 24 nodes, 1,623 reservations and
9,573 requests, independently checked to be all DISPATCHED. There are 3,246
append calls per measured round. Seven trials of 1,000 repeats give median
thread CPU 75,478 ns before and 7,419 ns after (90.17% reduction for this isolated
operation; absolute saving 68,059 ns). State restoration is outside measurement.
The test checks unchanged slots and zero feature counts on every round.

This is not a complete scheduling round, does not include lock acquisition,
and uses a later capture rather than the historical 45-second incident state.
Its small absolute saving cannot explain that incident. No online improvement
or new image publication is established by these measurements.


Affected tests completed with clang / old ABI / release / Python 3.10:
`master_batch_reservation_test` 56/56, `navi_sched_role_test` 627/627 and
`navi_sched_cluster_test` 42/42. Bazel exit 0. The role invocation explicitly
excludes the previously reproduced, unresolved
`NaviSchedResourceTest.testConcurrentStopFinishAndAbortReleaseEveryInflightOwnerExactlyOnce`
Agent stop/finish race; these results do not establish that race is fixed.
XML, logs and the separate profile are retained in the same evidence directory.
The production file is byte-identical to the restored fixed snapshot.

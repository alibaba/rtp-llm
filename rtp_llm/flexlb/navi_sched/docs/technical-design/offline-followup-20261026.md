# Master offline follow-up, 2026-10-26

Current target override (user instruction, 2026-10-27):
`omega_llm_976371_5721316_DeepSeek-VL`. Use this application for subsequent
runtime inspection and authorized rollout. Historical application/service IDs
below identify evidence sources, not the current deployment target. The new
application was read back from Whale as service `6ae032f9f76e3e3b1ddc2041`,
business version 0, template `VLLM_TEST.1713`
(`6ae01ec3537bcf6e640e29a4`). Six deployments are present: L20, L40S, RTX3090,
RTX4090, RTX5090 and RTX PRO 5000. Carbon identifies the Master in the RTX PRO
5000 deployment at `33.52.173.227`; its ready/health/service flags are
true/HT_ALIVE/SVT_AVAILABLE. This proves control-plane readiness, not scheduling
progress or successful requests. Fresh Carbon responses are saved under
`/tmp/navi-empty-waiting-20261027/cluster-*.json`.

The template-specific `/versions?sort_by=created_time@DESC&page_size=3` GET
returned 1713, 1712 and 1711, confirming 1713 is the latest listed version at
this inspection. Container access succeeded at 14:17--14:18 CST. The resolved
Master image is
`hub.docker.alibaba-inc.com/isearch/vllm_navi_sched_master@sha256:ec2914565e2ad6848e413ab49a73e92b9052627abd1a7c2a45e7653810a8bda8`.
Its image build-info declares core SHA `978ba5c8d762bbc636c93b8f540e9524262b630d4c16e1e486dd7d9e7094d7cf`,
whereas the installed core path hashes to
`137d8c51046ffc9b0f55d63152241bffa7bfa571454f01ea03dc1e6675445646`.
The process maps name that path with the same inode, but direct map_files reads
return EPERM; the installed file hash must not be presented as a verified hash
of all loaded executable pages. The build-info alone does not identify the
currently installed core, and the origin of the difference remains unconfirmed.

The process environment and live snapshot both show a 512-request limit and
100-ms wait; environment sets gRPC IO to 10 and the actual client/server IO
thread counts are each 10. These differ from the user's prior requested 20,000
and 20 settings. The 14:18:50 snapshot has 146 collectors, 45 Master reservation
batches and 4,110 inflight requests. Sampled rounds 41100--41400 show total
82.298--83.943 ms and result finish 2.269--3.906 ms. They do not reproduce the
historical 45-second delay or establish that all requests succeed. Evidence:
`new-master-runtime.log`, `new-master-window.log`, `new-master-mapped-core.log`
and `new-master-target.json` in the same local evidence directory. No new
template, publication or runtime configuration write was performed here.

The user deferred further rollout to the next day. This work does not authorize
new production traffic or another deployment today. All measurements below are
offline evidence; none establishes that the historical 45-second incident or
the network delay has been fixed.

## Changes under validation

### Paired request cost evaluation

`fillCostsForNodeRange` used to build separate running and waiting latency
features for every request/node pair. `calculateRequestLinearCosts` constructs
the five single-request input terms once and evaluates both costs. Model
parameters, feature definitions, scaling, and the order of floating-point
summation remain unchanged. Zero terms are retained, including their behavior
with non-finite coefficients. The independent feature-based functions remain
the numerical oracle and are not changed by this optimization.

The model test covers parameter vector lengths 0 through 22, positive and
negative coefficients, zero/partial/full hits, and prompt sizes up to 1,048,576
tokens. The replay benchmark checks every entry of both float cost matrices
against the independent evaluator outside the measured region.

First local comparison, identical 20,000-request/223-node replay, 10 workers,
9 measured repetitions after warmup:

| Measurement | Before | Paired evaluation |
| --- | ---: | ---: |
| Preparation median | 90,309 us | 73,647 us |
| Cost evaluation, sequential median of 3 trials | 188,040 us | 46,388 us |

This scaled replay restores actual captured cache/pending state, but repeats
requests to create the larger request set. It does not reproduce production
cache-update contention, client delivery, Agent execution, or the historical
45-second stall. A reverse experiment and final test results are recorded in
the evidence directory rather than inferred from these initial measurements.

### KV matching operation counts

The existing slow log separates physical-cache lock wait, pending-cache lock
wait, matching time and cache sizes. It lacked work counts. Batch matching now
also records:

- `prefix_comparisons`: element comparisons against the previous request's
  confirmed prefix; shared vector identity requires no element comparisons.
- `reused_full_prefix_blocks`: confirmed full-prefix blocks skipped by reuse.
- `full_prefix_probes`: prefix positions actually tested for readiness,
  including the first miss. This counts block positions, not individual hash
  table lookups within the physical/scheduled/Agent readiness checks.
- `tail_probes`: hybrid tail positions tested, including the successful tail.

Counts are computed from traversal boundaries without a second scan. They are
local to each read-locked batch query and reset on every call. Logging remains
outside cache locks and uses the existing slow-log threshold/rate limit.

The deterministic test includes shared vector identity, equal prefixes in
different vectors, a full-prefix miss, hybrid tail fallback, and empty-query
reset. Expected counts are respectively 3 comparisons, 5 reused blocks,
6 full-prefix probes and 5 tail probes.

### Cross-node reuse of request prefix comparisons

Previously, each node compared adjacent requests' hash vectors independently
to discover their common prefix. `prepareCacheMatchPrefixes` now computes
adjacent common-prefix lengths once, before node workers are dispatched.
Node queries use the minimum of that length, the previous request's confirmed
full match on this node, and the current request's matchable length. This does
not reuse another node's cache-hit result. Full readiness still comes from
the current node's read-locked state; hybrid tail readiness is checked again
for every request.

The O(R) array belongs to the scheduling context. Activating or clearing a
filtered request view invalidates it; construction uses that view's actual
request order. Workers only read it. The serial cost path prepares the same
array. Queries without prepared metadata retain the previous implementation.
No lock ordering or cache publication protocol changes.

Tests cover reordered/filtered views, shared vector identity, null vectors,
short prompts, policy mismatch, randomized full/hybrid matching against the
scalar oracle, scheduled/Agent cache changes, and concurrent cache writers.
The deterministic work test requires zero per-node element comparisons when
prepared metadata is provided, with exactly unchanged block/tail probe counts.

Controlled comparison with paired cost evaluation enabled on both sides,
10 workers, 9 measured repetitions:

| Replay | Per-node prefix comparisons | Prepared common-prefix lengths |
| --- | ---: | ---: |
| Scaled 20,000 requests / 223 nodes | 108,534 us median | 82,880 us median |
| Historical 512 requests / 44 nodes | 1,222 us median | 945 us median |

The large-case wall times vary across measurement windows, so these gains
must not be added to the earlier paired-cost comparison. Preparation includes
constructing the common-prefix array. Each complete cost matrix is compared
against the previous matching path before accepting a benchmark result.

Both work-counter and cross-node-reuse gates record old FAIL, fixed PASS,
removed FAIL, restored PASS. The cross-node gate restores and runs the full
pending-cache and scheduling-context suites, including concurrent writers.
The failure criterion is an exact operation count, not a timing threshold.

## Offline lifecycle audit

The retained historical capture contains 9,573 reservation requests, all in
`DISPATCHED` state. Their mean total prompt length is 3,909.42 tokens; mean
stored cache-hit length is 3,851.89 tokens. Across 24 nodes, the sum of per-node
pending full-block unions is 38,448. Scheduled and Agent copies overlap, so
adding their two totals would double-count this capture's blocks.

For every retained reservation batch, the union of its live requests' block
hashes exactly equals its scheduled pending full-block vector. No scheduled
pending batch lacks a reservation; every Agent pending batch maps back to a
reservation by schedule batch ID. Serialized unique-block counters equal the
recomputed unions on every node. This is evidence against orphaned pending
entries in this one capture; it does not prove that all failure sequences are
leak-free or that any individual dispatched request still makes progress.

The retained 2026-10-26 load samples show natural draining after reducing
replay to 0.01 QPS. Across approximately 240 seconds, inflight decreases from
6,012 to 383 and pending blocks from 25,456 to 1,647. Neither reaches zero in
that observation window. Stopping replay later is cancellation cleanup and
must not be reported as natural completion.

## Remaining work and release gates

| Request | Evidence/status | Still required |
| --- | --- | --- |
| Historical 45-second stall | Historical timing and failure cohort retained | Reproduce the complete interval or identify the blocked owner from equivalent trace evidence; the old Pod is gone |
| Large-round preparation | Paired evaluator, prefix reuse and work-count instrumentation; local reverse gates passed | Final affected tests, then actual production phase timings after tomorrow's rollout |
| Reservation/result-finish tails | Earlier per-node workers, COW state, exact saved contributions and incremental removal exist | Validate remaining long lock transactions; the short-commit design is still a proposal, not delivered behavior |
| Client-to-Agent arrival span | Packet captures show slow body delivery; Agent post-deserialize work is small in the sampled cohort | Confirm kernel/path cause with valid probes and a controlled intervention |
| Adaptive learning removal | Controller source/build entries removed in inherited changes; retired configuration is rejected | Final package/source audit and role rollout verification; historical export rejection is not an active controller |
| Normal replay overhead | Context creation and replay-prepare metric are guarded by collection state in current code | Preserve no-collection regression tests; verify deployed requests and current Web data after rollout |
| First-arrival batch deadline | Current Agent code assigns the deadline only when creating a batch | Preserve late-request/timeout/cancellation tests; correlate remaining aggregation failures with transport captures |
| Dashboard completeness | No fresh Web dashboard read during offline work | Read the current rendered dashboard and query real data tomorrow before asserting panel names or reporting health |

Network counterproof: `tcp_small_queue_check.isra.0` received a scalar second
argument in all 1,208 captured calls. Dereferencing that argument as an skb in
the earlier probe was invalid. Entry probes also do not prove that the check
returned a throttle decision. The observed 8 ms gaps alone do not establish
TSQ as the cause, and preheated transfer controls do not isolate a pacing-option
effect. No kernel setting is changed on the strength of that evidence.

Local evidence: `/tmp/navi-offline-20261026/`. Raw load evidence remains under
`/tmp/navi-workload-ownership-20261024/`; historical capture remains under
`/tmp/navi-timeout-lifecycle-20261024/cohort/`; packet/probe evidence remains
under `/tmp/navi-arrival-span-20261024/`.

## Final local validation

- Release/clang/old-ABI/Python-3.10 build, frame pointers retained, no stripping.
- Full role suite: 623 tests, zero failures/errors, 298.142 seconds.
- Model suite: 25 tests, zero failures/errors, including non-finite parameter
  equivalence with the independent evaluator.
- Local wheel built and RECORD hashes/sizes verified. Only the core DSO and
  RECORD differ from the preceding Master package; Python/Observer payloads
  are unchanged. This wheel has not been deployed.
- Extracted core SHA256:
  `f53cc519eaa472f39fed2c3d9c97cdde17550e0041fe173ad4ae8851b4f7f545`.
- `.symtab`, `.debug_info`, `.debug_line` and both new function symbols verified
  in the packaged DSO. GDB resolves the prefix-preparation function to
  `ScheduleContext.cpp:471`.
- The exact packaged DSO was loaded by SHA and a local cost Master initialized
  and stopped successfully. This smoke test does not measure live traffic.
- All 15 changed source files were rehashed against the tested package source
  manifest after the smoke test. Review delta, benchmarks, reverse gates,
  source archive and wheel manifest are in the local evidence directory.

## 2026-10-27: independently audited onset before dispatch failures

Evidence and executable integer-time assertions are under
`/tmp/navi-initial-trigger-audit-20261027/audit.py` and `audit.json`. The manifest
hashes the retained source excerpts and collector output. The original remote
gzip is not present locally; no claim is made that missing historical records
have been recovered. All times in this section are China Standard Time.

The actual accepted configuration record at 2026-10-21 16:24:23.384889 contains
both previous and effective configuration. Exactly four fields changed:

| Field | Before | After |
| --- | --- | --- |
| `latency_adaptive_learning_rate.enabled` | false | true |
| `latency_learning_rate` | 0.05 | 0.011 |
| `latency_online_learning_rate_scale` | 1 | 6 |
| `residual_learning_rate` | 0 | 0.03 |

The remaining adaptive-controller fields are identical. All eight Agent ACKs
carry effective generation 1792570507599149; the last ACK is 53,290 us after
acceptance. The first retained dispatch-failure event follows 58.016611 seconds
after the accepted change. This is a temporal association, not proof of a
learning-induced regression: pending retention already grew for minutes before
this change, and the historical pre/post model states required for a controlled
counterfactual have not been recovered.

An independent parse of the raw `AUDIT` log lines covers all 18 complete rounds
7476 through 7493 before that first failure. Each round contains exactly 30
requests: summed per-node prediction membership agrees with the count of unique
per-request cache audits. All result intervals are non-overlapping.

| Quantity | Measured value |
| --- | ---: |
| Requests across the 18 rounds | 540 |
| First-to-last context creation span | 978,373 us |
| Arrival rate, later 17 cohorts over that span | 521.274 request members/s |
| Total prediction-to-last-reservation wall time | 1,821,295 us |
| Members / summed result-stage wall time | 296.492 request members/s |
| First round context-to-prediction wait | 28,760 us |
| Last round context-to-prediction wait | 808,881 us |

The wait increase is accounted for exactly, without converting wall time into
CPU time: 1,669,578 us of preceding result service minus 978,373 us of arrivals,
plus 88,916 us between consecutive result intervals, equals the observed
780,121 us of additional waiting. These rates describe this selected cohort and
stage, not global ingress QPS or successful execution QPS. This trace recurrence
is not a replay of historical production code.

The first failure joins directly by request and batch ID to round 7493:
request 14633499906513424175, batch 3253912271161154180. The context is 962,675 us
old at failure, and its last reservation log precedes failure by 2,077 us. The
individual Client's remaining deadline at context creation is missing, so these
values do not by themselves establish the transport failure reason.

Of the 540 audited requests, 489 have positive pending hits and source IDs. The
historical source at commit 648100e5 performs two full reservation-source scans
for each such source query, then obtains overlap-based Stats per request. Later
failure cleanup adds another independently reproduced amplification path. The
logs prove that the initial pre-failure serialization path is active and cannot
keep up in this cohort; they still do not partition its wall time into source
scanning, Stats, logging, lock contention and preemption.

A counting error was found in the old `master_trigger_growth.py` collector:
`round_requests` retains only the first node batch's `request_count` after
deduplicating schedule IDs. The historical logging call uses
`decision.expectedBatchSize`, not the entire round size. Thus that collector's
request sum must not be used as round traffic. The 540-member calculation above
sums every distinct node batch and independently checks request audit IDs.

The old minute pending totals are also explicitly asynchronous: each node's
last observation is summed, rather than an atomic cluster snapshot. They grow
from 3,455 batches / 3,962,176 blocks at the end of 16:23 to 5,517 / 4,791,066 at
the end of 16:24. Completed-event membership falls from 8,056/613 = 13.142 to
7,645/1,811 = 4.221 members/event. These observations establish growing state
and finer completion grouping; they do not prove why grouping changed or count
successful HTTP responses. Existing retained-source joins also show previously
completed batches supplying pending hits on all eight nodes before the first
failure, with oldest observed completion-to-hit ages from 194 to 469 seconds.
Retention alone does not identify the hash mismatch for every historical batch.

The new `testMasterReservationCacheAuditRequiresReplayCollection` exercises
30 requests that hit a pre-existing pending reservation, with collection both
off and on. It checks zero versus 30 audit records, exact pending source/hit
fields, identical reservation/block accounting, and complete failure cleanup.
Temporarily replacing the existing collection guard with an unconditional
block makes the test fail (30 records when zero are expected); restoring the
guard passes. Removing it again fails, and final restoration passes all 46
selected pending/lifecycle tests under clang / old ABI / release / Python 3.10,
with test caching disabled. XMLs and `gates.json` record the four stages.
`MasterStateR.cpp` is byte-identical to its pre-experiment version; this step
adds a regression for the existing optimization, not another production-path
change. None of this constitutes online validation or exclusive historical
CPU attribution.

## 2026-10-27: new-cluster release and iteration audit

Evidence is under `/tmp/navi-master-release-20261027/` and
`/tmp/navi-newcluster-replay-20261027/`. The current service is
`6ae032f9f76e3e3b1ddc2041`, app
`omega_llm_976371_5721316_DeepSeek-VL`.

The pre-release template 1713 already specified IO20, round20000 and 64 CPUs,
but the deployment's complete `NAVI_SCHED_MASTER_CONFIG_JSON` replaced the
template JSON and omitted IO/count settings. Carbon target and runtime both
confirmed the resulting IO10/count512 defaults. Template 1714 carries the
complete merged configuration, preserving the new app subscription, and the
deployment JSON override has been removed. Business version 1 was submitted
in one request for all six deployments; all six publish records were returned
without failed deployments. Effective inference target environment differences
are exactly removal of MODEL_CATEGORY, MM_CONCURRENCY,
DEVICE_RESERVE_MEMORY_BYTES and RESERVER_RUNTIME_MEM_MB. The previously retired
VLLM_VIDEO_LOADER_BACKEND was already absent. GPU images and other effective
inference settings are unchanged.

Master image is `hub.docker.alibaba-inc.com/isearch/vllm_navi_sched_master@sha256:80f1ea1272a3dcdf6b785b08f5c2daa149c2f80c6596815794a0a91ee4857fea`;
rollback is the preceding ec291456 image recorded in `package/published.json`.
Runtime PID 32042 reports core SHA256
`536729251fda6b8f1366c891623a0cf06428d865f2e15e10d820fc388d69ee07`,
matching the package, with 20 client IO threads, 20 server IO threads, count20000
and wait100ms. Build mode is clang/old ABI/release/py310; symtab, debug_info and
debug_line were checked. Packaging changes only the native core and wheel
RECORD relative to the preceding Master wheel. The image also refreshes the
formerly stale build-info.env. Existing affected tests passed: reservation56,
role627, cluster42; role explicitly excludes the previously documented,
unresolved Agent concurrent-stop race. Packaged cost-Master init/stop passed.
These gates do not prove the historical 45-second root cause resolved.

During convergence, six direct text probes succeeded; five image probes
succeeded and one returned NaviSched429. Thus end-to-end execution is verified
for both input types, but full fleet availability and stable-load performance
remain unproven. Readiness snapshots and request IDs are retained, including
unallocated L40S slots that predate this release.

Replay45094 has max_loop_count10, budget80000us, parallelism10; it completed
one gradient iteration and zero discrete refinement evaluations/moves.
Elapsed548386us includes preparation/activation time and is not pure gradient
or discrete-search CPU. The first step records forward56us, backward15us,
projection107us and elapsed548149us. Same-window logs for the neighboring
non-collect round45100 report round85699us, result5672us, reserve3759us and
replay_setup_us=-1. They are useful negative evidence against attributing
all rounds to capture overhead, but do not locate the missing time in45094.

Controlled offline replay variants retain the original requests, pending
state, nodes, admission random draws and initial weights. With discrete local
search disabled, 1, 10 and 50 gradient iterations produce identical decisions:
253 selected requests on one node, predicted batch execution67972.964ms.
Iteration count alone does not improve this case. Offline replay executes
worker shards serially and excludes input restoration; its wall time cannot
be advertised as production latency or an online speedup.

The downloaded native perf contains7266 cpu-clock samples over15.0865s, with
2047 nonempty call chains. Core addresses were symbolized from the exact ELF
and recorded mmap; executable PT_LOAD virtual addresses differ from file
offsets by4096bytes. Current matching-host kallsyms resolves784 observed
addresses. Kernel paths include futex_wake/try_to_wake_up, TCP receive and
epoll, with BPF receive probes visible. Sparse user call chains and CPU-clock
sampling do not measure off-CPU lock wait or establish historical causality.

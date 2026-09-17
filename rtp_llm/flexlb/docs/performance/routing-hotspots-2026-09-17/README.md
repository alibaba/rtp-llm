# Master routing hotspots: implementation and local measurements

## Commits and validation

Baseline: `1de3c4b0af297179defa1aac0ac4a109483ffa7b` (already includes incremental formula prediction).

| Commit | Change | Reactor tests before committing |
|---|---|---:|
| `05d5e17c51` | Maintain the probe's live queue position; remove repeated identity scans | 1,206 passed |
| `839a31469d` | Ordered identity index, immutable membership captures, shared version materialization | 1,212 passed |
| `de1b92f6aa` | Two-prefix prediction cache; primitive planner loop state | 1,214 passed |

All three were committed separately, in this order, after their own successful full reactor run.
Command (from `rtp_llm/flexlb`):

```sh
JAVA_HOME=/Users/hena/Library/Java/JavaVirtualMachines/ms-21.0.9/Contents/Home ./mvnw -B -P 'opensource,!internal' -pl flexlb-sync -am test
```

The separate before/after differential compares 2,000 full selections, 2,000 readiness plans,
and 4,000 complete candidates. Both revisions produce the same SHA-256; see the
[before](before-verification.txt) and [after](after-verification.txt) outputs. Tests also
cover expired holes on both sides of the probe, consumed groups, overshoot, disabled
prediction, concurrent snapshot construction, stale captures, retry after construction
failure, exact identity removal, and committed-work/ACTIVE component independence.

## Design

- `ProjectedQueue` owns the count of live members before the probe. Consumption and
  expiry update that count. Each plan reads the position once; its prediction adapter
  only clamps the requested prefix to that position. No callback-local position state.
- `PrefillActiveIndex` remains the canonical ACTIVE index. A TreeSet provides ordered
  iteration and an identity map locates the exact node for removal. Equal ordering keys
  preserve distinct identities. Membership changes invalidate only the membership capture.
- Each index entry lazily creates its immutable `GroupPlanner.Item` once. Captures reuse
  those values and build their immutable ordered projection once, outside the owner lock.
- `WorkerBatcher` captures ownership, constraints, and version under the existing lock,
  publishes that source, then materializes outside the lock. Readers of the same version
  share the source. Old builds only fill their own source, so cannot overwrite a new one.
  Successful materialization releases the source's references to request contexts.
- The append cursor retains only the latest and previous prediction. Older prefix queries
  explicitly fall back to full evaluation. Selection constructs Shape/OptionalDouble at
  the result boundary, preserving saturated arithmetic and strict/inclusive capacities.

## Measurement setup

Apple M5 Pro, 15 logical CPUs, 48 GiB RAM; Microsoft OpenJDK 21.0.9; 1 GiB JVM heap.
This is a local algorithm benchmark, not a production Master CPU-utilization forecast.

[Runner and fixture](../../../tools/routing-hotspots/README.md) use identical source on
both revisions, real request/runtime objects, and the supplied DeepSeek formula. No
Mockito instrumentation is active. An initial instrumented pilot was discarded because
it changed allocation behavior in identity/equality paths.

26 cases cover 10/100 endpoints, 32/1024 members, 1/64 planner threads, and a 1,024-request
batch cap with a 700 ms prediction budget. Capture tests include `captureRouteProjectionInputs`.
Status cases invalidate scheduling inputs. Membership cases remove/reinsert one request
per endpoint through PrefillState under its real lock. All mutations occur before each
concurrent wave, so every scan has identical members and deterministic checksums.

Three JVM forks per revision alternate order. Each case has two warm-up rounds and three
measurement rounds of at least 200 ms (index cases: 100,000 remove/add pairs). Table values
are medians of nine samples. Capture scans repeat eight times per invalidation; full
projection scans repeat twice. Numbers include task coordination and mutation work,
amortized over those scans; they are not timings for a single cold snapshot build.

One operation is a full scan of the specified endpoint fleet. With 64 planners, wall/op is
aggregate throughput, not per-request latency. CPU/op sums planner and coordinator thread
CPU, excluding GC/JIT threads. Allocation includes planners and coordinator. Scheduler
threads, RPCs, network waits, decode selection, and the full Master process are not measured.

## Results and limits

- 100 endpoints × depth 1024 × 64 planners, capture + projection: CPU/op **12.124 → 10.283 ms**
  (**15.2% lower**), allocation **14.394 → 5.926 MiB** (**58.8% lower**).
- 10 endpoints × depth 1024 × 64 planners: CPU/op **1.130 → 0.868 ms** (**23.2% lower**).
- 100 endpoints × depth 1024 × one planner: full-path CPU/op **10.585 → 6.756 ms**
  (**36.2% lower**). Scheduling-input-only invalidation strongly benefits from membership reuse.
- **Regression:** 100 endpoints × depth 1024 × one planner, membership-changing capture-only
  workload: wall/op **0.785 → 1.058 ms** (**34.7% higher**), CPU/op **0.783 → 1.032 ms**.
  This scenario still copies O(N) membership after every invalidation; the new representation
  does not make arbitrary membership changes O(1). Attribution between ordered traversal,
  extra snapshot arrays, and memory locality needs a dedicated profile; the current data
  establishes the regression, not its exact cause. Allocation still falls about 65%.
- 100 endpoints × depth 32 × 64 planners, membership capture also regresses modestly
  (**0.84 → 0.92 µs/op** wall). See all cases below, including regressions.
- An isolated remove/add pair adds **72 bytes** in the new index versus zero previously.
  At depth 32 its time rises roughly **61 → 70 ns**; at depth 1024 it is roughly flat
  (**209 → 202 ns**). Additional retained index-node memory was not separately measured.
- Thread-count, memory locality, GC, and local CPU scheduling produce variation across
  forks. Reported medians do not imply the online CPU gauge will fall by these percentages.
  The implementation is ready for review; the membership-heavy serial regression is a
  reason to evaluate the actual deployment's mutation/read ratio before rollout.

## Complete results

Values are **before → after**. For index rows one operation is one remove/add pair;
for fleet rows it is one full-fleet scan.

| Case | Wall µs/op | Thread CPU µs/op | KiB/op |
|---|---:|---:|---:|
| index_remove_add_depth32 | 0.061 → 0.070 | 0.060 → 0.070 | 0.000 → 0.070 |
| fleet10_depth32_p1_status_capture | 3.562 → 1.237 | 2.786 → 0.737 | 4.753 → 0.319 |
| fleet10_depth32_p1_membership_capture | 3.060 → 2.482 | 2.336 → 1.937 | 5.212 → 2.556 |
| fleet10_depth32_p1_status_project | 33.778 → 27.765 | 30.575 → 23.749 | 69.207 → 21.590 |
| fleet10_depth32_p64_status_capture | 0.230 → 0.211 | 0.369 → 0.270 | 0.120 → 0.023 |
| fleet10_depth32_p64_membership_capture | 0.224 → 0.207 | 0.361 → 0.312 | 0.127 → 0.058 |
| fleet10_depth32_p64_status_project | 3.664 → 2.980 | 24.446 → 18.720 | 50.760 → 20.641 |
| fleet100_depth32_p1_status_capture | 14.486 → 3.268 | 13.652 → 2.595 | 46.914 → 2.868 |
| fleet100_depth32_p1_membership_capture | 15.683 → 14.198 | 14.975 → 13.499 | 51.797 → 25.231 |
| fleet100_depth32_p1_status_project | 271.008 → 191.693 | 266.320 → 188.769 | 690.781 → 216.941 |
| fleet100_depth32_p64_status_capture | 0.679 → 0.456 | 3.349 → 1.243 | 1.349 → 0.064 |
| fleet100_depth32_p64_membership_capture | 0.837 → 0.923 | 3.995 → 4.632 | 1.428 → 0.417 |
| fleet100_depth32_p64_status_project | 47.551 → 28.531 | 348.451 → 260.233 | 508.048 → 205.722 |
| index_remove_add_depth1024 | 0.209 → 0.202 | 0.205 → 0.202 | 0.000 → 0.070 |
| fleet10_depth1024_p1_status_capture | 50.000 → 1.064 | 48.917 → 0.615 | 117.344 → 0.319 |
| fleet10_depth1024_p1_membership_capture | 50.101 → 35.314 | 49.169 → 34.320 | 117.949 → 41.423 |
| fleet10_depth1024_p1_status_project | 915.712 → 594.937 | 903.264 → 590.885 | 1930.250 → 609.324 |
| fleet10_depth1024_p64_status_capture | 1.740 → 0.179 | 9.856 → 0.249 | 12.483 → 0.023 |
| fleet10_depth1024_p64_membership_capture | 2.171 → 1.136 | 12.244 → 4.242 | 12.581 → 0.671 |
| fleet10_depth1024_p64_status_project | 139.555 → 107.582 | 1129.955 → 868.348 | 1490.737 → 608.140 |
| fleet100_depth1024_p1_status_capture | 759.856 → 3.492 | 757.102 → 2.713 | 1173.086 → 2.868 |
| fleet100_depth1024_p1_membership_capture | 785.362 → 1058.026 | 782.621 → 1031.932 | 1179.141 → 413.903 |
| fleet100_depth1024_p1_status_project | 10688.944 → 6951.518 | 10585.300 → 6755.714 | 19289.953 → 6079.277 |
| fleet100_depth1024_p64_status_capture | 23.126 → 0.383 | 191.787 → 1.147 | 153.546 → 0.064 |
| fleet100_depth1024_p64_membership_capture | 27.541 → 22.641 | 213.009 → 103.121 | 153.574 → 6.682 |
| fleet100_depth1024_p64_status_project | 2197.188 → 1140.611 | 12123.828 → 10283.273 | 14739.580 → 6068.059 |

Raw nine-sample results: `before-0.txt` through `before-2.txt`, `after-0.txt` through
`after-2.txt`; machine-readable medians: [summary.json](summary.json).

The runtime commits do not modify deployment configuration or online services.

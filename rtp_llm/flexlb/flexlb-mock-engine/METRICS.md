# Mock metric contract and cleanup

Audited against framework commit `7caf62e452` and its C++ real-engine source.
This change covers the mock control server's Prometheus exposition and the
stress, scenario, report and Whale no-Fetch consumers. It does not remove
state used by `/snapshot`, request lifecycle checks or scheduling.

## Names shared with real

| Removed name | Supported name | Source and meaning |
|---|---|---|
| `mock_engine_running` | `rtp_llm_running_stream_size` | Executing prefill + decode request counters, matching the existing Whale scheduler gauge. The old `runningTasks.size()` also included queued/allocated lifecycle entries; it cannot be relabelled unchanged. |
| `mock_engine_waiting` | `rtp_llm_wait_stream_size` | Scheduler queue, excluding decode ALLOCATE reservations waiting for P/Fetch. |
| `mock_engine_cache_blocks` | `rtp_llm_kv_cache_pool_total_blocks` | Simulated device block-pool capacity. |
| `mock_engine_available_blocks` | `rtp_llm_kv_cache_pool_available_blocks` | Free + evictable cached blocks; excludes held and request-referenced blocks. |

C++ definitions: `rtp_llm/cpp/metrics/RtpLLMMetrics.cc`,
`engine_base/schedulers/FIFOScheduler.cc` and `cache/BlockPool.cc`.
Mock sources: `JavaMockEngineCluster.getSnapshot/whaleMetrics` and
`MockLruBlockCache`. The snapshot retains `running` as lifecycle inventory
and adds `scheduler_running` for the executing-stream gauge.

Per-engine labels remain `engine_name`, `role`, `grpc_port`, `engine_ip`;
default `/metrics` sums gauges/counters by role. These labels describe mock
engines, not real pool/group topology. Comparing capacities across backends
still requires matching block size and topology. No GPU allocation is measured.

## Deliberately retained mock names

| Family | Actual consumer | Why the proposed real name is incorrect |
|---|---|---|
| `mock_engine_held_blocks`, `mock_engine_referenced_blocks` | `aggregate_canvas_run.py` → KV block-pool panels | Held means keyless allocations; referenced means indexed cache-key blocks in use. Real request-ref includes both classes; free means unallocated. Neither mock split can be renamed to free/request-ref. In this mock pool, request-held total = held + referenced = total − available. |
| `mock_engine_cache_key_hits_total`, `mock_engine_cache_keys_requested_total` | Cache-hit aggregation/panels and scenario balance samples | Cumulative admission-time prefix-matched **keys** vs requested keys. Current real recent-cache-key hit/total are per-request **token gauges**, based on `RecentCacheKeyWindow`, not cumulative residency counters. |
| `mock_engine_prefill_ms_avg`, `mock_engine_decode_ms_avg` | Per-engine execution curves; decode balance scoring | Means over bounded recent simulated execution samples in milliseconds. Real `rtp_llm_model_forward_us` reports individual forward execution in microseconds; decode request duration can span multiple steps. Multiplying by 1000 does not fix the sample/granularity difference. |
| `mock_engine_accepted_total`, `mock_engine_completed_total` | Elastic transient/balance checks; Whale `observe.py` | Cumulative mock lifecycle counters, not the real engine's QPS gauges. |
| `mock_engine_cache_evictions_total`, `mock_engine_kv_admission_fails_total`, `mock_engine_lack_mem_rejects_total`, `mock_engine_decode_reuse_blocks_total` | KV aggregation, admission/reuse panels | Mock block eviction and admission events remain useful; no verified identical real contract was found. |

The real cache-hit evidence is `PrefillCacheHitMetricsReporter.cc`:
`fillPrefillRecentCacheKeyMetricsCollector` assigns `hit_token_count` and
`input_token_count`; `record` matches a recent-key window. The mock already
has a separate `WhalePrefillMatchMetrics` implementation for that real
contract. Reusing its name for the lifetime key counters would collide with
an existing, different measurement. `compare_twin.py` no longer treats
recent-cache-key gauge names as cumulative key-counter aliases.

## Removed Prometheus families

Ten families are removed from HELP/TYPE, per-engine and role output:

- `mock_engine_up`: health and per-engine stopped state remain available.
- `mock_engine_cache_keys`: no analysis consumer. Cache inventory is **not**
  replaced by hit/request counters; it remains in `/snapshot.cache_keys`.
- `mock_engine_active_kv_tokens`, `mock_engine_available_kv_tokens`: no
  analysis consumer of these series; block-pool analysis is already used.
  Snapshot token fields remain needed for capacity/injection checks.
- `mock_engine_rpc_total`: the initial zero-consumer finding was incorrect.
  `whale_mock/observe.py` consumed Fetch counts. It now sums the existing
  `/snapshot.engines[].rpc_counts.fetch_response` by role, rejecting absent
  counters/roles instead of assuming zero. RPC instrumentation stays intact.
- `mock_engine_prefill_ms_p99`, `mock_engine_decode_ms_p99`,
  `mock_engine_prefill_ms_count`, `mock_engine_decode_ms_count`: only collected
  for a future planned score; no implemented analysis used them. Counts are
  bounded queue sizes, not monotonic sample counters. Snapshot diagnostics
  remain, and the role avg still uses queue counts as weights.
- `mock_engine_cancelled_total`: sampled but never used by a case or report.
  Cancellation analysis uses snapshot request IDs/lifecycle and event logs.

## Migration boundary and verification

Deploy producer, collectors, scenario framework and offline analyzer from
the same revision. Old names are not dual-emitted. Generated report JSON
keys (`total_blocks`, queue columns, etc.) remain stable; archived raw
Prometheus captures with the old names should use their original analyzer.
In particular old running values cannot be converted to scheduler-running
values by renaming.

Tests cover both HTTP exposition modes (including absence of removed
HELP/TYPE/sample lines), block-pool values, reserved-vs-running decode
state, collection through consolidated time series into queue/KV aggregation,
scenario consumers, and fail-closed no-Fetch verification.

Validation on 2026-09-21:

- Python: 129 elastic scenario tests; 48 collector/shared-runtime/cache-hit/
  migration/twin tests; 4 no-Fetch observer tests. All 181 passed.
- Java on 111, isolated run `20260921_064254.`: 35 tests across
  `PythonCompatControlApiTest`, `MetricsValidationTest`,
  `BlockPoolMetricsObservabilityTest`, `MockRemoteDecodeEngineTest`,
  `CacheKeyHitMetricsTest`, `KvAllocatedReportOptInTest`, and
  `WhalePrefillMatchMetricsTest`; zero failures/errors/skips.
  After the final cancelled-counter removal, the 17 control/metrics tests
  were rerun successfully (`metrics-final`).
- Maven: `./mvnw -B -P'opensource,!internal' -pl flexlb-mock-engine -am test
  -Dtest=<classes above> -Dsurefire.failIfNoSpecifiedTests=false`.
- No live Whale deployment or workload benchmark was changed or run.

## Prefill TPS alignment (`execution_us_v1`)

The previous HTTP context pair counted tokens per scrape without dividing by
elapsed time. Whale used successful-request totals divided by modeled formula
milliseconds. Neither was the real execution TPS contract.

The source of truth is `rtp_llm/cpp/metrics/RtpLLMMetrics.h`:
`RtpLLMTokenPSMetricsCollector::addTokenSize` gives each positive numerator
its own batch execution-time sum. For completed batches i, let C be executed
context tokens (input minus reuse), I be context input including reuse, and
E be measured execution microseconds:

- `context_tps = 1e6 × Σ(C_i where C_i>0 and E_i>0) / Σ(E_i for those batches)`.
- `context_tps_with_cache` applies the same rule independently to I.
- The wall pair divides the same numerators by actual elapsed report time.
  `wall_tps_report_interval_us` records that time per engine.

For a fully cached batch, C can be zero while I is positive. Thus execution
TPS denominators can differ; subtracting the two rates is not a cache-hit
calculation. Window hit ratios now use the wall pair. A sum across engines
is a wall-rate-weighted reuse ratio when their report intervals differ;
it is not an exact pooled token ratio in that case. Successful-request
snapshot totals remain a separate run-level view.

`RtpLLMMetrics.cc` and the two loop reporters in the header define missing
versus idle samples: no completed sample while an execution is active is
silent; a fully idle interval reports zero. Each mock HTTP/Whale reader has
an independent cursor and retains its wall origin through silent intervals.
The first completed sample is included. Crash resets start a new generation
and old completed callbacks cannot add tokens to that generation.

`NormalExecutor.cc` times execution from scheduler handoff through dispatch
(with process-start fallback); `MtpExecutor.cc` also uses the scheduler
interval. It is not simply `model_forward_us`. Mock records actual elapsed
time from batch execution start through simulated completion dispatch,
including modeled delay and runtime overhead. Batch token membership is
frozen at start: cancellation before start excludes work; cancellation after
start does not erase executed work. Both numerator and duration publish
atomically. Successful-request counters retain their original meaning.

This alignment covers the mock's one-phase prefill batches and engine totals.
The real `StreamGroups.h` supports execution slices and batch multiplicity;
mock does not model chunked prefill/beam execution or emit per-priority TPS.
Compare real engine totals (sum priority buckets where present). Simulated
execution time is not evidence of absolute GPU throughput. Decode TPS was
not changed or certified by this prefill migration.

Consumers preserve separate execution and wall curves. Elastic throughput
checks use wall TPS. The existing `compare_ab.py` command automatically
rejects mixed prefill contracts; no extra TPS option is needed. Capture
both versions with aligned producers/collectors and matching workload, and
use full output fetching; old raw captures cannot be repaired by renaming.
The aggregator infers the contract from the new measured-window series;
deploy a single producer revision per experiment rather than mixed fleets.

Verification on 2026-09-21:

- 191 Python tests across collector/runtime/telemetry/migration/cache/twin,
  elastic scenarios and the prefill contract. These cover deliberately
  unequal execution denominators, wall hit ratios, missing samples and both
  accepting and rejecting gate paths at that revision. The optional TPS gate
  and its three tests were subsequently removed; comparison retains the
  automatic mixed-contract check.
- 49 remote Java tests passed on host 111, run `20260921_070951.`, job
  `tps-final`; after adjusting Whale's wall endpoint to follow its atomic
  snapshot, the 30 TPS/Whale tests passed again (`tps-clock-final`).
- Deterministic ledger tests cover ratio-of-sums, zero-compute batches,
  long-step silence, first sample, independent observers and crash reset.
  Integration tests cover HTTP role sums, cancellation and Whale parity.
- No live GPU comparison or production TPS benchmark was run.

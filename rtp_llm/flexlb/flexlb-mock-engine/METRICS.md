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

# FlexLB Mock Engine

A Java-based mock engine for FlexLB load balancing testing. Simulates real GPU inference timing with configurable performance formulas, fault injection, and monitoring.

## Features

- **Realistic timing simulation**: Uses `ScheduledExecutorService.schedule()` to wait for formula-computed prefill/decode durations
- **Performance formula**: Supports `PrefillTimeFormula` AST evaluation with batch/input/hit-cache/compute token variables
- **Fault injection**: 10 fault types (enqueue_error, generate_error, no_respond, kv_pressure, queue_depth, crash_after, enqueue_delay, generate_delay, and more)
- **HTTP control**: 11 endpoints for runtime control (/snapshot, /inject, /clear_inject, /health, /requests, /set_perf, /set_kv_pressure, /set_queue_depth, /stop_engine, /start_engine, /metrics)
- **Inflight leak detection**: 30s periodic check with 60s grace period
- **KV cache modeling**: LRU cache with prefix matching, pressure simulation
- **Concurrency modeling**: Prefill batch-level wait queue (inflight capped by `max_prefill_concurrency`, default 1 per DP rank), decode wait queue + hard concurrency gate (`decode_max_concurrency`, default 132) with backpressure rejection when the pending queue is full

## Quick Start

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@21
export JAVA_MOCK_ENGINE_HEAP_SIZE=2g
export MOCK_BASE_GRPC_PORT=62000
export N_PREFILL=2 N_DECODE=4 DURATION_S=30 REPLAY_SPEED=10
cd tools/online_eval
bash run_online_eval.sh
```

## Configuration

### Performance JSON
| Key | Default | Description |
|-----|---------|-------------|
| sleep_scale | 1.0 | Global timing multiplier (0.1=fast, 1.0=realistic) |
| prefill.fixed_ms | null | Fixed prefill latency (bypasses formula) |
| prefill.min_ms | null | Floor for the final (post-scale) prefill sleep in ms; guards against sleep_scale making prefill unrealistically fast |
| prefill.scale | 1.0 | Prefill-specific multiplier |
| decode.scale | 1.0 | Decode-specific multiplier |
| decode.step_ms_by_batch | [[1,1.0],...] | Per-step latency by batch size |
| decode.per_token_ms | null | Fixed per-token decode latency (ms); when set, overrides step_ms_by_batch curve (e.g. 45.0 ≈ DeepSeek V3 ~22 tok/s) |
| jitter_pct | 0.0 | Random jitter (±%) |

### Runtime HTTP API
| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Engine health check |
| /snapshot | GET | JSON snapshot of engine state |
| /metrics | GET | Prometheus-format metrics |
| /inject | POST | Inject fault (type, delay_ms, n, etc.) |
| /clear_inject | POST | Clear all fault injections |
| /set_perf | POST | Override prefill_ms, decode_step_ms, jitter |
| /set_kv_pressure | POST | Set KV pressure (`active_kv_tokens` absolute / `tokens` additive) |
| /set_queue_depth | POST | Set queue depth limit (real enqueue rejection) |
| /stop_engine | POST | Stop engine (simulate crash) |
| /start_engine | POST | Restart stopped engine (auto-clears faults) |
| /requests | GET | List recent request/task records |

The control server listens on `baseGrpcPort - 1` of the mock cluster.

**Decode pending-queue capacity**: when `queue_depth_limit` is not explicitly set (via
`/set_queue_depth` or fault injection), the effective decode pending cap defaults to
`max(256, decode_max_concurrency × 2)`. This bounds the decode wait queue so that
under overload the engine rejects excess requests with backpressure rather than
queuing unbounded. Use `/set_queue_depth` to override at runtime.

**Prefill waiting-queue cap**: `prefill.max_waiting_batches` (default 4) bounds the
number of QUEUED prefill batches per engine — running batches never count toward the
cap. When the queue is full, `enqueueBatch` returns a per-request error
(`prefill waiting queue full (backpressure): waiting=N cap=M`) and `generateStreamCall`
fails the stream, so the master sees an explicit rejection instead of a silent
timeout. This gate is batch-level and independent of the request-level fault-injection
`queue_depth_limit` check at the RPC entry; both stack.

**Queue metrics — four-state naming and units**: the periodic `java_mock_stats` log
line (interval configurable via `--stats-interval-ms`, default 5000 ms; the
`run_online_eval.sh` env passthrough is `JAVA_MOCK_STATS_INTERVAL_MS`) reports
symmetric P/D queue states:

| Field | Unit | Meaning |
|-------|------|---------|
| `ts_epoch_ms` | epoch ms | Sampling wall-clock timestamp (`System.currentTimeMillis()`), aligns with client `send_start_epoch_ms` |
| `prefill_waiting` | requests | Queued (not running) prefill requests, sum over prefill engines |
| `prefill_running` | batches | Running prefill batches (a batch may hold several requests), sum |
| `max_prefill_waiting` | requests | Peak single-engine queued prefill requests |
| `decode_waiting` | requests | Queued (not running) decode requests, sum over decode engines |
| `decode_running` | requests | Running decode requests, sum |
| `decode_run_min` | requests | Min single-engine running decode requests (mean = `decode_running` / n_decode) |
| `decode_run_max` | requests | Max single-engine running decode requests |
| `max_decode_waiting` | requests | Peak single-engine queued decode requests (symmetric with `max_prefill_waiting`) |
| `decode_done` | requests | Decode requests completed since the previous sample (window counter) |
| `decode_exec_p50` | ms | Window p50 of decode execution time (end − running-start; bounded reservoir approximation) |
| `decode_exec_p95` | ms | Window p95 of decode execution time |
| `decode_exec_max` | ms | Window max of decode execution time (exact) |

The old `prefill_pending` (waiting + running mixed) and `max_prefill_pending` fields
are gone. All other pre-existing fields are unchanged (additive-only evolution).
`/snapshot` additionally exposes a top-level `ts_epoch_ms` (sampling timestamp) and
`prefill_waiting_batches` per prefill
engine — the queued BATCH count, i.e. the same unit as `prefill.max_waiting_batches`
(the `waiting` snapshot field counts requests).

**Monitoring / Prometheus target contract**: since the Java rewrite the whole
cluster is a single process and `/metrics` is served **only** on the control
port `baseGrpcPort - 1` (aggregated by role by default, `?per_engine=true`
for per-engine series). Configure exactly one scrape target
(`<host>:<baseGrpcPort - 1>`). The Python-era shard aggregation port formula
(`base + n_prefill + n_decode + 100 + shard_id`) no longer exists.

**Engine addressing**: POST bodies accept either `{"engine": "prefill-0"}` (engine
name, same naming scheme as the cluster) or `{"port": N}` (gRPC port).

**Python compat notes**:

- `/snapshot` returns the legacy shape `{"engines": [...], "cluster_counters": {...}}`.
- `/inject` accepts both the Java format (`{"type": ..., "enabled": ...}`) and the
  legacy Python format (`{"config": {"enqueue_error": bool, ...}}`).
- `/set_kv_pressure`: `active_kv_tokens` sets the absolute active-KV-token count
  (legacy Python semantics); `tokens` adds pressure tokens (original Java semantics).
- `/set_queue_depth`: the `queue_depth` field name is accepted for compatibility,
  but unlike the legacy Python behavior (a display-only value bumping the snapshot
  `waiting` counter), the Java engine implements it as real enqueue rejection.
- `/metrics`: aggregated by role by default; append `?per_engine=true` for
  per-engine labels (`engine_name`/`role`/`grpc_port`/`engine_ip`).

## Test Suite (68 test methods)

| Test | Methods | Description |
|------|---------|-------------|
| JavaLoadClientParityTest | 14 | Load client parity with the legacy Python client |
| PythonCompatControlApiTest | 11 | Python control-plane compatibility layer |
| ComprehensiveFaultInjectionTest | 8 | All fault types |
| ClusterConfigParamTest | 7 | Cluster CLI/config parameters |
| InflightLeakTest | 6 | Inflight leak detection |
| FaultInjectionConfigTest | 5 | Builder pattern |
| JavaMockEngineClusterTest | 3 | Core engine functionality |
| CodeReviewFixTest | 3 | Review-fix regressions |
| ConcurrentDoubleSchedulingTest | 2 | Double-scheduling guard |
| MultiShardRoutingTest | 2 | Multi-shard routing |
| CancelMidFlightTest | 1 | Cancel mid-flight requests |
| EngineCrashRecoveryTest | 1 | Engine crash/restart recovery |
| HighConcurrencyStressTest | 1 | 500 requests @ 100 concurrency |
| InflightTtlExpiryTest | 1 | TTL cleanup mechanism |
| MatrixSweepTest | 1 | P/D config × concurrency sweep |
| MetricsValidationTest | 1 | /metrics + /snapshot validation |
| RealisticTimingTest | 1 | Real timing verification |

## JavaLoadClient

Standalone replay/load client (`org.flexlb.mockengine.JavaLoadClient`), configured
entirely through environment variables (`Config.fromEnv`):

| Env var | Default | Description |
|---------|---------|-------------|
| TRACE_FILE | "" | Replay trace jsonl path (empty = no trace replay) |
| TARGET_ADDR | 127.0.0.1:7001 | flexlb-api HTTP address |
| GRPC_TARGET | derived | flexlb gRPC address (default: TARGET_ADDR host, port+2) |
| DURATION_S | 0 | Max run duration in seconds (0 = until trace exhausts) |
| MAX_CONCURRENCY | 999999999 | Client-side concurrent request cap |
| REPLAY_SPEED | 10.0 | Trace replay speed multiplier |
| LOAD_CLIENT_WORKERS | 1 | Replay worker count |
| OUTPUT_DIR | load_client_output | Output dir (summary.json, per_request.jsonl) |
| NUM_SHARDS | 1 | Number of trace shards |
| SHARD_INDEX | 0 | Shard index replayed by this instance |
| LIMIT | 0 | Max requests to replay (0 = all) |
| TIMEOUT_MS | 3600000 | Global run timeout in ms |
| SLA_TTFT_MS | 500.0 | TTFT SLA threshold for the report |
| ZERO_OUTPUT_POLICY | skip | Zero-output trace rows: skip / one / default100 |
| SCHEDULE_ONLY | false | Schedule (enqueue) only, skip FetchResponse |
| FLEXLB_EXPECT_FETCH_RESPONSE | "" | Override fetch-response behavior (0/false/no disables) |
| LOOP | false | Loop the trace |
| N_CHANNELS | 8 | gRPC channels |
| EVENT_LOOP_THREADS | 32 | Netty event-loop threads |
| START_AT_EPOCH_MS | 0 | Aligned start epoch ms (0 = start immediately) |
| RESPONSE_TIMEOUT | 120 | Per-request response timeout in seconds |
| SKIP_SERVER_LATENCY | false | Skip /server_latency sampling |
| MODEL | engine_service | Model name on requests |
| API_KEY | "" | API key header |
| GRADIENT | false | Gradient (ramp-up) replay mode |
| GRADIENT_START_SPEED | 10 | Gradient start speed |
| GRADIENT_MAX_SPEED | 1000 | Gradient max speed |
| MAX_INPUT_LEN | 0 | Truncate input tokens beyond this length (0 = off) |
| MAX_OUTPUT_LEN | 0 | Truncate output tokens beyond this length (0 = off) |
| PUSHGATEWAY_URL | "" | Push Prometheus metrics to this Pushgateway |
| ENABLE_FALLBACK | false | Enable fallback prefill via ENDPOINTS_FILE |
| ENDPOINTS_FILE | "" | endpoints.json written by JavaMockEngineCluster |
| DRY_RUN | false | Parse and validate only, no traffic |
| SEND_MODE | replay | Arrival process: replay (trace ts pacing) / uniform (fixed interval) |
| SEND_MODE_QPS | 0 | uniform mode total target QPS (per shard = QPS / NUM_SHARDS) |

## Architecture

```
JavaMockEngineCluster
├── MockPerformanceModel      — formula evaluation, timing calculation
├── FaultInjectionConfig      — fault injection configuration
├── MockControlServer         — HTTP control endpoints
├── ScheduledExecutorService  — timing simulation (schedule completions)
├── responseExecutor          — blocking queue poll for response delivery
├── FastRpcService            — gRPC service (enqueue, generate, status, cancel)
└── MockCacheStore            — LRU KV cache with prefix matching
```

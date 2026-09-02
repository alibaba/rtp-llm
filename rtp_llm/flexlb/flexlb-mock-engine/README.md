# FlexLB Mock Engine

A Java-based mock engine for FlexLB load balancing testing. Simulates real GPU inference timing with configurable performance formulas, fault injection, and monitoring.

## Features

- **Realistic timing simulation**: Uses `ScheduledExecutorService.schedule()` to wait for formula-computed prefill/decode durations
- **Performance formula**: Supports `PrefillTimeFormula` AST evaluation with batch/input/hit-cache/compute token variables
- **Fault injection**: 9 fault types (enqueue_error, generate_error, fetch_error, no_respond, kv_pressure, queue_depth, crash_after, enqueue_delay, generate_delay)
- **HTTP control**: 14 endpoints for runtime control (/snapshot, /inject, /clear_inject, /health, /requests, /set_perf, /set_kv_pressure, /set_queue_depth, /stop_engine, /start_engine, /cancel_request, /add_engine, /remove_engine, /metrics)
- **Inflight leak detection**: 30s periodic check with 60s grace period
- **KV cache modeling**: block-pool capacity model v2 — heterogeneous prefill/decode pools, LRU-coupled admission/eviction (LACK_MEM), per-step decode KV growth, pressure simulation
- **Concurrency modeling**: Prefill batch-level wait queue (inflight capped by `max_prefill_concurrency`, default 1 per DP rank; queued batches capped by `prefill.max_waiting_batches`, default 0 = zero-waiting fail-fast, with backpressure rejection), decode wait queue + hard concurrency gate (`decode_max_concurrency`, default 132) with backpressure rejection when the pending queue is full

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
| prefill.max_waiting_batches | 0 | Cap on queued (not-running) prefill batches per engine; excess enqueues are rejected (backpressure). `0` (shipped default) = zero-waiting fail-fast: a batch is accepted only when the engine is idle (a `max_prefill_concurrency` slot is free), otherwise rejected immediately. Positive n allows up to n queued batches — rule of thumb: n ≈ SLO_ms / batch_ms − 1 (e.g. SLO 1000 ms, batch 150 ms → 4, deepest wait 600 ms + 150 ms execution leaves ~25% headroom); for 1x-scale runs where a batch takes ~330–400 ms, use 1–2. Absent field or negative = unbounded queue (legacy behavior) |
| decode.scale | 1.0 | Decode-specific multiplier |
| decode.step_base_ms | 19.5 | Per-step decode latency intercept of the linear production fit: step_ms = step_base_ms + step_per_running_ms × running (production DSv4 fit, task #68). Applies when no `step_ms_by_batch` curve is declared |
| decode.step_per_running_ms | 0.175 | Per-step decode latency slope per running stream (production DSv4 fit) |
| decode.tokens_per_step | 2.6 | MTP acceptance fold: tokens produced per running stream per decode step (production DSv4 accepts 2.54–2.88). Steps for output_len tokens = ceil(output_len / tokens_per_step) |
| decode.step_ms_by_batch | null | Explicit per-step latency curve [[batch, step_ms], ...]; when declared it overrides the linear fit (mutually exclusive with step_base_ms/step_per_running_ms). Absent → linear production fit |
| ~~decode.per_token_ms~~ | — | REMOVED (task #69): fixed per-token latency was a V3-era no-MTP single-stream caliber that overstated low-batch decode ~5.5× and full-batch ~2.8×. Declaring it now fails fast with a migration hint |
| jitter_pct | 0.0 | Random jitter (±%) |

### Runtime HTTP API
| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Engine health check |
| /snapshot | GET | JSON snapshot of engine state |
| /metrics | GET | Prometheus-format metrics |
| /inject | POST | Inject fault (type, delay_ms, n, etc.) |
| /clear_inject | POST | Clear all fault injections |
| /set_perf | POST | Override prefill_fixed_ms, decode_scale, max_prefill_concurrency |
| /set_kv_pressure | POST | Set KV pressure (`active_kv_tokens` absolute) |
| /set_queue_depth | POST | Set queue depth limit (real enqueue rejection) |
| /stop_engine | POST | Stop engine (simulate crash) |
| /start_engine | POST | Restart stopped engine (auto-clears faults) |
| /cancel_request | POST | Cancel an in-flight request by request id |
| /add_engine | POST | Dynamically add an engine to the running cluster |
| /remove_engine | POST | Dynamically remove an engine (graceful drain) |
| /requests | GET | List recent request/task records |

The control server listens on `baseGrpcPort - 1` of the mock cluster.

**Dynamic engine management**: `DynamicEngineManager` serves `/add_engine` and
`/remove_engine` — a running cluster can be scaled up/down over HTTP without a
restart. New engines register on the next port, write into the endpoints/
discovery files, and are picked up by the master's file-discovery watcher;
removal drains gracefully so in-flight requests finish first.

**Prefill waiting-queue cap**: `prefill.max_waiting_batches` bounds the
number of QUEUED prefill batches per engine — running batches never count toward the
cap. Semantics: `cap >= 0` is enforced; the shipped default `0` is zero-queue
fail-fast mode — the engine accepts a batch only when it is idle (a concurrency
slot is free) and rejects immediately otherwise, so requests never wait in the
engine; absent field or negative = unbounded (legacy). When the queue is full,
`enqueueBatch` returns a per-request error
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
| `prefill_running_reqs` | requests | Requests inside running prefill batches, sum over prefill engines (running-request companion to `prefill_running`) |
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

**Production-caliber TPS series (`rtp_llm_*`, 20260901)**: `/metrics` reports
three series under the production metric names so mock dashboards read like
the real engine's — `rtp_llm_context_tps` (computed context tokens = input −
cache hits), `rtp_llm_context_tps_with_cache` (input tokens including cache
hits, the DeepSeek-style "input tokens/s incl. cache hits" caliber) and
`rtp_llm_generate_tps` (generated output tokens). They are **pure accounting
on completion events** (prefill completions feed the context pair, decode
completions feed generate; cancelled completions are excluded — only tokens
actually accepted and generated count, matching production semantics) and
every scrape settles the window first, so the window = scrape interval (the
eval G1 poller is 1s → the value is tokens/s by construction). PD split:
prefill engines carry the context pair, decode engines the generate series
(off-role series stay 0). Caliber note: the mock's execution time is itself a
formula product, so unlike production there is no execute/wall dual
denominator — the fixed 1s window is the whole denominator. These are
**accounting-style simulation readings that measure scheduling organization
efficiency, not GPU compute**: do not compare absolute values against
production dashboards directly (the caliber semantics map one-to-one, the
physics does not). `/snapshot` exposes the cumulative `hit_tokens_total`
per engine (never drained — the `cache_saved_tokens` source via
`final_snapshot`). The report layer renders these three series as P/D
role charts at the **per-engine average** (cluster sum ÷ engine count —
the production dashboard's single-instance-series read, so per-instance
magnitudes line up; the aggregate's `mock_tps_ts` keeps the raw cluster
sums and the division is presentation-layer, with a cluster-sum fallback
caption 「集群和（引擎数未知）」 + stderr warning when the engine count
is unavailable — chain: `run_meta.json` params > `engine_dist` > mock
final_snapshot role counts); the client-side token reconciliation is
not a report panel but the aggregate's fail-closed validity item
`validity_checks.token_reconciliation_ok`: per input/output side,
`|client completed tokens − (Σ mock_tps_ts + in-flight Σ)| ≤ max(5% ×
client, 5 × peak per-second tokens)` — the in-flight term adds the
Σil/Σol of ok rows whose rid is absent from the engine-terminal done sets
(`mock_prefill_done` / `mock_decode_done`, the same join full_e2e uses):
fire-and-forget runs record ok at schedule success with the expected
output_len, so requests still decoding at run end never feed the mock Σ
(measured 7.1M / 15.3% of client output tokens on run 20260901_200108);
runs without engine terminal logs degrade to in-flight = 0 (legacy
formula). 5% absorbs scrape-window edge / clock residue and
cancelled-request one-sided accounting, 5 × peak bounds the post-scrape
drain tail; a missing series → `null` (no false failure).

**Block-pool observability series (`mock_engine_*`, 20260902)**: `/metrics`
reports the KV v2 block-pool state as time series in BOTH emission modes
(per-engine and role-aggregated) — four per-scrape GAUGES
`mock_engine_cache_blocks` / `mock_engine_available_blocks` /
`mock_engine_held_blocks` / `mock_engine_referenced_blocks` (the three-state
split: available = free + pure LRU, held = in-flight keyless leases,
referenced = in-flight-referenced key blocks — the same snapshot fields the
`/snapshot` terminal view exposes) and four cumulative COUNTERS
`mock_engine_cache_evictions_total` (LRU evictions, allocation-coupled),
`mock_engine_kv_admission_fails_total` (DECODE degradations: un-pooled
continue + growth stall), `mock_engine_lack_mem_rejects_total` (PREFILL
synchronous 602 rejections — prefill REJECTS, decode DEGRADES, the two
surfaces never cross-book; healthy runs keep both at 0, overload runs light
them up each on its own role) and `mock_engine_decode_reuse_blocks_total`
(the KV v2 fix #5 net-demand deduction: hit keys against the engine's OWN
LRU at decode admission, never drained — "the more decode runs, the more
it saves"). All seven ride the report chain: the G1 whitelist →
aggregate `kv_blocks_ts_by_role` (per-role cluster sums) → the canvas 5. KV
engine-side block-pool panels at the **per-engine average** (the same
engine-count chain as the TPS charts; counter columns rendered as
adjacent-bucket cumulative diffs ÷ bucket gap, counter resets clamped to
zero). Caliber note: the gauges are the mock's simulated block pool at
1s scrape granularity (the KV v2 capacity model's own state), not real GPU
memory pages — read the shapes and the admission/reuse events, not
absolute block counts against production.

**Cache key-hit series (`mock_engine_*`, 20260902)**: `/metrics` also reports
two cumulative counters aligning the mock with the production
`recent_cache_key_hit_count / total_count` caliber —
`mock_engine_cache_key_hits_total` (Σ block keys matched by the
admission-time `prefixHitBlocks` call: keys the engine could reuse) and
`mock_engine_cache_keys_requested_total` (Σ requested `blockKeys`;
empty-block-hash requests contribute 0/0 by construction). Both are booked
at the SAME point — prefill admission's prefix-match computation — so the
pair is self-consistent (never one-sided); they carry per-engine AND role
labels in both emission modes, are never drained (cumulative since engine
start), and `/snapshot`'s terminal view adds the final `cache_key_hits` /
`cache_keys_requested` per engine. They ride the eval chain: G1 whitelist →
aggregate `cache_hit_ts` (engine-key column) → the canvas 5c cache
hit-rate panels (the "key-level (theoretical)" caliber; the full
caliber trio — master routing / engine key-level / engine token-level —
is documented in the online_eval README).

**Engine addressing**: POST bodies accept either `{"engine": "prefill-0"}` (engine
name, same naming scheme as the cluster) or `{"port": N}` (gRPC port).

**Python compat notes**:

- `/snapshot` returns the shape `{"engines": [...]}` (plus a top-level `ts_epoch_ms`
  sampling timestamp).
- `/inject` accepts both the Java format (`{"type": ..., "enabled": ...}`) and the
  legacy Python format (`{"config": {"enqueue_error": bool, ...}}`).
- `/set_kv_pressure`: `active_kv_tokens` sets the absolute active-KV-token count
  (Python semantics; the field is the only accepted form).
- `/set_queue_depth`: the `queue_depth` field name is accepted for compatibility,
  but unlike the legacy Python behavior (a display-only value bumping the snapshot
  `waiting` counter), the Java engine implements it as real enqueue rejection.
- `/metrics`: aggregated by role by default; append `?per_engine=true` for
  per-engine labels (`engine_name`/`role`/`grpc_port`/`engine_ip`).

## Test Suite (231 test methods)

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
| MetricsValidationTest | 3 | /metrics + /snapshot validation, KV block-pool tracking + pressure-surface consistency |
| TpsMetricsAccountingTest | 3 | rtp_llm_* TPS series: completion-event accounting, drain semantics, cancelled exclusion, hit_tokens_total |
| BlockPoolMetricsObservabilityTest | 3 | Block-pool series in both /metrics modes: three-state gauges over a request's life, prefill-602 vs decode-degrade counter split, decode reuse accumulation (cumulative, never drained) |
| CacheKeyHitMetricsTest | 2 | Cache key-hit counters: prefix-match run accumulation (hit/requested key sums across requests, /metrics + /snapshot terminal fields) and empty-block-hash 0/0 contribution |
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
| OUTPUT_DIR | load_client_output | Output dir (per_request.jsonl, server_latency.json) |
| NUM_SHARDS | 1 | Number of trace shards |
| SHARD_INDEX | 0 | Shard index replayed by this instance |
| LIMIT | 0 | Max requests to replay (0 = all) |
| TIMEOUT_MS | 3600000 | Global run timeout in ms |
| SLA_TTFT_MS | 500.0 | TTFT SLA threshold for the report |
| ZERO_OUTPUT_POLICY | skip | Zero-output trace rows: skip / one / default100 |
| FETCH_OUTPUT_STREAM | true | Client reads engine output streams after Schedule; 0 skips the client-side stream read while the engine still executes prefill+decode in full (BATCH dispatcher only) |
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
| PRIORITY | 50 | Env-level default QoS priority (per-record trace priority overrides; explicit 0 = leave unset on the wire) |
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
└── MockLruBlockCache         — KV block pool + LRU prefix cache (capacity model v2)
```

## Current-branch extensions (auto-tpm / priority)

The sections above describe the shared `feat/flexlb_mock_engine_v2` baseline.
This branch (`intake3-rebuild`, based on
`codex/auto-tpm-request-mode`) additionally carries the following capabilities.

### Unique engine advertisement IPs

Every engine advertises a unique 127.x.y.z loopback IP (default on; `--unique-engine-ips=false`
reverts to the legacy shared `--host`) instead of all declaring 127.0.0.1, so the master-side
`engineIp` Prometheus label stays distinct per engine — with a shared host, per-engine gauge
series (batcher queue / KV / inflight) overwrote each other. The gRPC bind stays wildcard
(`forPort`), only the advertised address changes (worker status, `DOMAIN_ADDRESS`,
endpoints.json, `/metrics` `engine_ip` label); Linux routes all of 127.0.0.0/8 to loopback, but
macOS only reaches 127.0.0.1 by default — disable the flag for local macOS runs that connect
across engine addresses.

### Cancel channel

- **`MockEngineCancelChannel`** (`src/main/java/org/flexlb/mockengine/MockEngineCancelChannel.java`):
  an in-process test channel exposing the same cancel contract the cluster's
  `FastRpcService.cancelRequest` implements, so tests can drive cancellation
  without going through gRPC.
- **`POST /cancel_request`** on the MockControlServer (control port
  `baseGrpcPort - 1`, body `{"engine"|"port": ..., "request_id": ...}`): a
  test-only cancel injection that drives the same three-branch
  `cancelRequest` contract —
  1. **found**: the request is removed and a CANCELLED completion surfaces in
     the next WorkerStatus finished list (WorkerStatus stays the sole
     release-confirmation source);
  2. **already finished**: reported via the `already_finished` flag;
  3. **unknown / not found**: `{"status": "NOT_FOUND", "found": false}`.
  The response carries `{status, found, phase, already_finished, engine,
  port, request_id}` with `phase` as the TaskPhase enum name (null unless
  found). A **Decode** target returns HTTP 501 `UNIMPLEMENTED`, matching the
  production role contract (decode engines do not accept client cancels).

### Priority preemption (auto-tpm)

End-to-end QoS priority, from trace to engine tombstone:

- **Load client**: `PRIORITY` env sets the default priority (50, the
  neutral QoS level — priority 0 is rejected by master admission, so it must
  never be the load-test default; an explicit `PRIORITY=0` leaves the field
  unset on the wire);
  a per-record `priority` field in the trace **overrides** the env default;
  the winner is put on the wire via `ScheduleRequest.setPriority`. Shard summaries
  expose `priority_stats` (per-priority `{total, completed, rejected,
  avg_schedule_ms}`, built by `priorityBreakdown`).
- **Engine side**: on eviction, the finished TaskInfo carries error code
  **8429 (`PRIORITY_PREEMPTED_ERROR_CODE`, "preempted by higher-priority
  request")** — an idempotent tombstone that masters can re-observe safely
  after restarts. The cancelled entry preserves the ACTUAL phase the request
  was cancelled in (a queued decode request surfaces
  `TASK_PHASE_KV_ALLOCATED`, a queued prefill `TASK_PHASE_RECEIVED`;
  `RUNNING` remains the fallback).
- **P→D ownership tables**: each prefill engine tracks its downstream decode
  owner per request (`downstreamDecodeOwners`) and each decode engine its
  upstream prefill owner (`upstreamPrefillOwners`), so cancel/finish on
  either side can release the counterpart's inflight entry exactly once.

### KV capacity model v2 (block pool)

The KV cache is now a **token-counted block pool** (`MockLruBlockCache`), aligning the
mock's KV behaviour with the real engine's CacheManager/BlockCache semantics so the
master-side curves are indistinguishable from production:

- **Pool derivation**: `totalBlocks = ceil(totalKvTokens / spb)` — a block is the unit
  of admission, eviction and accounting.
  `available_kv_tokens = (free + evictable-LRU blocks) × spb − kv_pressure`:
  pure-LRU blocks count as AVAILABLE (evictable at the cost of prefix reuse),
  mirroring the real `available = free + LRU` semantics. Held (running) blocks and
  referenced (running-with-keys) LRU blocks are NOT available.
- **Role-heterogeneous defaults**: prefill pool defaults to 6,291,456 tokens
  (6,144 blocks), decode to 4,194,304 (4,096) — decode holds each request's KV for
  its full lifetime, so the smaller pool lets the master's cross-engine comparisons
  (`min kvCacheUsed` / KV% gates) actually see role divergence. Per-role overrides:
  `--prefill-total-kv-tokens` / `--decode-total-kv-tokens`; the legacy
  `--total-kv-tokens` still sets BOTH pools uniformly.
- **Lease lifecycle**: prefill holds `ceil(inputLen/spb)` blocks for the duration of
  the batch and hands them to LRU on completion (release ≠ delete — availability
  recovers, `cache_keys` stay). Decode provisions its blocks at run start (opt-in
  queued-mode provisions at enqueue) by re-matching the request's block keys
  against its **OWN LRU** (KV v2 fix #5, production `reuse_block_size =
  generate_stream->reuseBlockSize()`) and allocating the **NET demand**
  `ceil(inputLen/spb) − hitBlocks` (floor 0): reused blocks are referenced in the
  LRU layer (ref+1, never re-allocated into the running layer), and the
  prefix-match read itself refreshes LRU recency — decode positive feedback, the
  more a prefix is matched the later its blocks sit in the eviction order. The
  lease then **grows per decode step** toward `ceil((inputLen + generated)/spb)`
  (production `incrMalloc` semantics — growth only allocates the generation
  delta, so the reuse deduction holds for the stream's full lifetime), handing
  over to LRU at completion. A cancelled stream never admits: cancel runs free()
  (blocks return to the pool, no LRU handover) — including the race where the
  cancel lands after the step-boundary terminal claim.
- **Admission/eviction coupling**: prefill allocation needs `keys.size` blocks
  (hash-present requests) or `ceil(inputLen/spb)` (empty-key requests, computed on
  the spot) and its TOTAL_AND_AVAILABLE gate evaluates the total demand; decode
  admission evaluates the **net demand** (`total − reuse`) — production reuse
  reduces `need_blocks` BEFORE the capacity gate, so a fully-reused decode
  request admits whenever the 5% reserve watermark holds, even when
  `total > available`.
  Free-block shortage first evicts LRU-tail blocks (sacrificing prefix reuse), and
  only returns **LACK_MEM** (error 602 `MALLOC_FAILED`, synchronous in the
  `enqueue_batch` ack `errors` with `request_states: "rejected"`; direct
  `generate_stream` fails the stream) when eviction cannot satisfy the request —
  the master's rejection-response path is now observable under load. Running leases
  reference their LRU blocks, making them non-evictable and non-available: the
  hotter the cache while requests run, the lower available drops — the
  scheduling-pressure shape the previous model could not express.
- **Decode never rejects**: on pool exhaustion a decode request degrades to un-pooled
  execution and bumps the `kv_admission_fails` counter (production
  `waiting_streams_` semantics — parking, not rejecting).
- **Flag semantics change**: `--prefill-cache-blocks` / `--decode-cache-blocks` have
  RETIRED their old meaning ("max cached key count") and now override the pool
  block count (default `0` = derive from token capacity). The 6000/3000 defaults
  still passed by `run_online_eval.sh` / `lib_load_client.sh` / `harness.py` remain
  valid — they now size the pools (6,000 blocks = 6,144,000 tokens prefill;
  3,000 = 3,072,000 decode) instead of capping key counts.
- **Surface alignment**: `block_size` in snapshots now reports the actual spb (was
  hardcoded 1024); `/snapshot` and `/metrics` expose `total_kv_tokens`,
  `cache_blocks`, `available_blocks`, `held_blocks`, `referenced_blocks` and
  `kv_admission_fails` alongside the existing `active_kv_tokens` /
  `available_kv_tokens` (per-engine series preserved); injected `kv_pressure` now
  consistently lowers availability across ALL surfaces (`getCacheStatus`,
  `WorkerStatus.availableKvCache`, `/snapshot`, `/metrics`).

**Baseline caliber (IMPORTANT)**: v2 changes the *shape* of the available/active KV
curves (prefill now occupies capacity while running; LRU holdings reduce available;
decode grows per step). KV v2 fix #5 further changes the decode *pressure level*:
decode admission now deducts local LRU reuse (net allocation = `total − hit_blocks`,
net-demand gate), so under prefix-heavy traffic the decode pool fills later and
less than the pre-fix caliber — which allocated the full `ceil(inputLen/spb)` per
request and systematically overstated decode pool pressure (earlier saturation,
earlier LACK_MEM / `kv_admission_fails`, lower master-side decode `available`).
Numbers from the mock3 baseline (available ≈ total − in-flight decode inputLen) are
NOT comparable across this version — re-baseline before any cross-version comparison.

### Decode hard-admission gate (unconditional)

`decodeMaxConcurrency` (default 132, overridable via
`--decode-max-concurrency`) is an **unconditional hard admission gate** with
an **unbounded engine-side pending queue** — production `waiting_streams_`
semantics: once all running slots are taken, new decode requests park in
`decodePendingQueue` (surfaced as `decode_waiting` in `java_mock_stats`)
and drain one-for-one as completions free slots. Nothing is ever rejected
on the decode side for queue pressure. Historically this was opt-in via
performance JSON `decode.max_pending_requests`; that key no longer exists —
the hard gate is the default and only behavior.

Companion flag `decode.report_queued_as_kv_allocated` (default false):
when enabled, queued decode requests are reported as
`TASK_PHASE_KV_ALLOCATED` in WorkerStatus (KV-fidelity semantics for the
accepted layer), so a master observing KV_ALLOCATED sees what a production
engine would have admitted.

**Note — the pending queue is unbounded (differs from v2's default cap)**:
the v2 baseline text above ("Decode pending-queue capacity": when
`queue_depth_limit` is not set, "the effective decode pending cap defaults to
`max(256, decode_max_concurrency × 2)`") describes the
`feat/flexlb_mock_engine_v2` behavior. This branch deliberately keeps the
pending queue **unbounded** (a production engine's `waiting_streams_` has
no engine-side queue cap either — backpressure is the scheduler's job).
The only queue bound is the request-level fault-injection `queue_depth_limit`,
which stays disabled until explicitly injected.

### Prefill waiting-queue cap — semantic difference (IMPORTANT)

The v2 baseline text above (`prefill.max_waiting_batches`: "0 = zero-waiting
fail-fast … absent or negative = unbounded") describes the
`feat/flexlb_mock_engine_v2` semantics. **This branch keeps a different,
opt-in semantic and does not adopt the v2 fail-fast default**:

- `max_waiting_batches > 0`: cap enabled — excess queued batches are
  rejected with backpressure (`prefill waiting queue full (backpressure):
  waiting=N cap=M`);
- `0` / absent (default `DEFAULT_MAX_WAITING_PREFILL_BATCHES = 0`):
  **unbounded** queue (legacy behavior, `<= 0` disables the cap).

The opt-in default is deliberate: Auto-TPM queue-eviction E2E scenarios need
deep engine-side queues, so the cap must be explicitly requested. Do not
"fix" the default to fail-fast without revisiting those scenarios.

### Test-suite size on this branch

The v2 baseline table above ("Test Suite (68 test methods)") is outdated
here: this branch's test surface totals **225 test methods**. Three v2
baseline classes grew — `JavaLoadClientParityTest` 14 → **15**,
`ClusterConfigParamTest` 7 → **9** (one auto-tpm param + one per-role KV pool
override) and `MetricsValidationTest` 1 → **3** (KV pool tracking +
pressure-surface consistency) — and the remainder of the delta are the new
classes listed below (plus a few new fixtures such as
`ClusterStatsDecodeWindowTest`, `LoopShardRidDisjointTest`,
`PrefillWaitingQueueCapTest`, `ShutdownDrainTest` and `UniformSendModeTest`).

### Additional tests on this branch

Representative additions beyond the v2 baseline suite (full list under
`src/test/java/org/flexlb/mockengine/`):

- `AutoTpmE2EHarness` — E2E harness for the auto-tpm scenarios
- `PriorityLatencyE2ETest` — priority-driven latency differentiation E2E
- `MockEngineCancelChannelTest` — in-process cancel channel contract
- `DecodePendingQueueHardGateTest` — unconditional decode hard gate semantics
- `KvAllocatedReportOptInTest` — queued-as-KV_ALLOCATED reporting
- `BlockPoolCapacityTest` — KV block-pool admit/evict/match: LRU-coupled
  allocation, LACK_MEM rejection, reserve watermark, lease growth/handover
- `LoadClientPriorityTest` — PRIORITY env vs trace-record priority, wire
  propagation, priority_stats
- `PreemptionPhasesE2ETest` — preemption phase fidelity across stages
- `HttpMockCancelIntegrationTest` — `/cancel_request` HTTP integration
- `DecodeCancelRaceTest` — cancel/decode completion races
- `FaultInjectionE2ETest`, `LeakCanaryLongRunE2ETest`,
  `BaselineParityE2ETest`, `TelemetryEmissionSurfaceTest`,
  `MockMasterConfig` (shared fixtures)

## Java-only cleanup on this branch

The Python mock engine / Python load client implementations have been
**removed** from this branch (`tools/online_eval/mock_engine.py`,
`mock_engine_cluster.py`, `mock_engine_shard_launcher.py`,
`flexlb_load_client.py`, `run_single_engine.py`, `test_resolve_decode.py` and
`tests/test_mock_engine.py`), together with `tools/online_eval/run_batch_smoke_only.sh`
(a matrix subset with no CI references — its coverage is subsumed by
`run_online_eval.sh` / `run_matrix_smoke.sh`) and the stale
`tools/online_eval/BUILD` filegroup that still referenced the deleted Python
files: nine files in total. The `MOCK_ENGINE_IMPL` / `LOAD_CLIENT_IMPL`
orchestration switches and their Python branches are gone as well, so the
Java stack described in this README is the only implementation.

All seven orchestration test scripts have been converted to the Java stack and
now drive JavaMockEngineCluster / JavaLoadClient through the shared
`tools/online_eval/lib_load_client.sh` helpers
(`start_java_mock_cluster` / `wait_mock_cluster_ready` / `mock_http` /
`stop_java_mock_cluster` / `run_java_load_client`):

- `flexlb_behavior_test.sh`
- `engine_kill_restart_test.sh`
- `run_cancel_smoke.sh`
- `run_matrix_smoke.sh`
- `master_kill_restart_test.sh`
- `master_recovery_ttft_test.sh`
- `engine_disconnect_ttft_test.sh`

The Python **smoke client family** is retained on purpose (it is tooling, not
the mock engine): `flexlb_smoke_base.py`, `priority_preemption_smoke.py`
and the analysis tooling talk to the Java cluster over
its gRPC + HTTP control plane. `encode_unique_key` now lives in
`online_eval/proto_utils.py`, so the smoke base no longer depends on the
deleted mock engine module.

Cancel transport surface: since the Java-only intake the Java mock cluster
also implements the gRPC `RpcService/Cancel` method, so the cancel intent can
now travel over three channels that share one cancel contract — the gRPC
`Cancel` handler, the HTTP `/cancel_request` control endpoint and the
in-process channel. The smoke suites default to the HTTP control plane (the
explicit `--flexlb.test.mock-cancel-control-url` wiring); pointing that
setting at the gRPC endpoint exercises the production channel instead.

Exit-code contract: the `engine_kill_restart` / `master_kill_restart` /
`flexlb_behavior` fault suites now propagate scenario failures to their exit
code — the pre-conversion scripts always exited 0. Anything wiring these
suites into automation should re-check pass/fail on the exit code instead of
scraping logs.

Assertion semantics: the engine-kill suite's assertion 5 (cancelled request
counting) is now faithful — since the Java mock implements gRPC `Cancel`,
cancels issued on the master's active-eviction path really reach the engine
and are counted. Historical "undercounted PASS" baselines will turn red;
that is a semantic alignment, not a regression.

# FlexLB Online Evaluation

This tool chain evaluates a running SpringBoot `flexlb-api` against a mock
rtp-llm engine cluster. The mock engine cluster and the load client are
Java-only (`flexlb-mock-engine`, JDK 21+); the Python mock engine / Python
load client implementations have been removed. The retained Python tools —
the smoke client family (`flexlb_smoke_base.py`,
`priority_preemption_smoke.py`), `stability_monitor.py`, and the
`analyze_*.py` / `sanitize_*.mjs` tooling —
run on the system `python3` and are intended for the `luoli_gpu` container,
where `grpcio`, `grpcio-tools`, and `protobuf` are available.

The legacy standalone smoke/chaos scripts (`cancel_smoke.py`,
`scheduling_smoke.py`, `anomaly_smoke.py`, `flexlb_behavior_test.sh`,
`engine_kill_restart_test.sh`, `master_kill_restart_test.sh`,
`master_recovery_ttft_test.sh`, `engine_disconnect_ttft_test.sh`,
`run_matrix_smoke.sh`, `run_cancel_smoke.sh`) have been removed: their
coverage now lives in the `flexlb_ft/` functional-test framework
(`flexlb_functional_tests.py --suite smoke|chaos --profile batch-window|single-nonbatch|single-batch|window-nonbatch`).

## One-command run

Run inside `luoli_gpu`:

```bash
docker exec -it luoli_gpu bash
cd <repo-root>

rtp_llm/flexlb/tools/online_eval/run_online_eval.sh
```

The run directory defaults to `rtp_llm/flexlb/tools/online_eval/run/<timestamp>/`.
During the load window the script also runs four per-second collectors (see
**Second-by-second collectors** below): mock per-engine Prometheus metrics,
master Prometheus business metrics, master inflight snapshots, and CPU/RSS
sampling of the three JVM groups. `JAVA_MOCK_STATS_INTERVAL_MS` defaults to
`1000` (1s mock stats cadence; was 5000) for fine-grained timelines, and
`FLEXLB_MONITOR_MODE` defaults to `all` so the master exposes the full
business metric surface (see the note under **Run output layout**).
After completion, the important outputs are (see the **Run output layout**
section below for the full table):

- `load_client/summary.json` (kept at the legacy path)
- `run_meta.json`, `mock.json` / `mock.log`, `master.json` / `master.log`,
  `client.json` / `client.log` (one JSON + one log per component)
- `per_request.jsonl` (or `per_request.jsonl.gz` for larger runs)
- `load_client/report.md`

Common overrides:

```bash
PROCESS_CONFIG_FILE=rtp_llm/flexlb/tools/online_eval/data/config/master_fixed_window.json \
DURATION_S=300 \
LIMIT=5000 \
REPLAY_SPEED=20 \
N_PREFILL=4 \
N_DECODE=16 \
SLA_TTFT_MS=800 \
FLEXLB_CONFIG='{"schemaVersion":2,"scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY","defaultPriority":50},"decision":{"type":"FIXED_WINDOW","maxRequests":32,"maxCollectionWaitMs":200}},"dispatcher":{"type":"BATCH"},"router":{"roles":{"prefill":{"selector":{"type":"ESTIMATED_TTFT","candidateChoice":{"type":"LEAST_RECENTLY_USED_IN_POOL","pool":{"type":"RATIO","ratio":0.3,"minimumWorkers":1}}}},"decode":{"availability":{"maxKvUsagePercent":90,"maxEngineRequests":64},"selector":{"type":"KV_USAGE_WEIGHTED_RANDOM"}}}}}' \
rtp_llm/flexlb/tools/online_eval/run_online_eval.sh
```

`FLEXLB_CONFIG` is the only FlexLB behavior document. In particular, the
prefill performance formula is
`router.roles.prefill.executionTimeEstimator.expression`; there is no separate
formula environment variable. Omitting the estimator applies the code default
(production DSv4 prefill fit, `RoutingConfig.FormulaEstimatorConfig.DEFAULT_EXPRESSION`),
which is also what the shipped `master_fixed_window.json` now relies on.

If `flexlb-api` is already running, use:

```bash
START_FLEXLB=0 \
FLEXLB_HTTP_ADDR=127.0.0.1:7001 \
rtp_llm/flexlb/tools/online_eval/run_online_eval.sh
```

If the default jar is not built, the script runs `./mvnw -pl flexlb-api -am package -DskipTests`.
The script auto-selects Java 21 from system alternatives when available; otherwise set `JAVA21_HOME` or `JAVA_HOME`.
It also defaults to `MAVEN_PROFILES=opensource,!internal` so an adjacent `internal_source` directory does not accidentally activate internal-only dependencies.

## Data layout

- `data/online_logs/trace_30min.jsonl`: sanitized replay shape derived from online logs.
- `data/online_logs/pod1_arrivals.tsv`: sanitized relative-time arrival analysis source.
- `data/online_logs/sample_access.json`: sanitized request-shape fixture with pseudonymous token IDs.
- `data/performance/dsv4_flash_performance.sample.json`: mock latency model.
- `data/config/master_fixed_window.json`: master process env config for the fixed-window baseline.

## Run output layout

`run_online_eval.sh` ends each run by consolidating the run directory
(`consolidate_run_outputs.py`) into one JSON + one log per component:

| File | Content |
|---|---|
| `run_meta.json` | `flexlb_env.txt` + `client_env.json` contents (the 36 JavaLoadClient env effective values snapshotted at client launch), endpoints summary, and the startup parameter snapshot (`--param` values incl. `FLEXLB_CONFIG`, JVM sizing, cache blocks, timeouts, `FLEXLB_MONITOR_MODE`, and the trace file's sha256 + line count). Also embeds the full config inputs (`performance_json` / `process_config_json` — the contents of the `performance_file` / `process_config_file` params; `null` when the file is missing) and the `process_usage` per-second CPU/RSS timeline of the mock / master / client JVMs |
| `mock.json` | `java_mock_stats` timeline (`stats` array, source field names `ts_epoch_ms` / `prefill_waiting` / ...). Note the parsers capture 26 of the 28 fields — `decode_exec_p50` / `decode_exec_p95` carry digits in the key and are skipped; the verbatim lines stay in `mock.log`. Also holds the final cluster `/snapshot` from the control plane (when reachable), the endpoints summary, and the A-split pointer `per_engine_file` + `per_engine_sample_count` — the per-second per-engine Prometheus timeline itself lives in `mock_per_engine_timeseries.json.gz` |
| `mock_per_engine_timeseries.json.gz` | the A-split target: the G1 per-second per-engine Prometheus timeline as gzip-streamed `[{ts, metrics: {"name{labels}": value}}]` groups (engine series like `mock_engine_running{engine_name="prefill-0",...}`). Splitting it out of `mock.json` keeps the main file lightweight — at 1250 engines × 120s the embedded key used to approach **~1GB** of pretty-printed JSON (the raw per-sample text is ~2.2KB × N_engines; the JSON-ified embedded form inflates ~2.5-3×). After the split + the G1 whitelist the same run writes a ~65MB `.json.gz` and `mock.json` stays in the KB range |
| `mock.log` | The original `mock_engine.log` verbatim (tail-friendly) with the JVM GC log appended under a `=====` separator |
| `master.json` | `master_counters_timeseries.txt` as a timeline array, `master_prometheus_after.prom` as a flat `{"name{labels}": value}` dict (HELP/TYPE skipped), `prometheus_timeseries` — the per-second master Prometheus timeline (same `[{ts, metrics}]` grouped shape; whitelisted to the analyzer-consumed series: `flexlb_app_cache_*` / batcher + routing queue gauges / inflight max age / dispatch reason counters, plus `jvm_memory_used` / `jvm_gc_pause` / `process_cpu` / `system_cpu`), `inflight_timeseries` — per-second `/rtp_llm/inflight_status` snapshots (`[{"ts_epoch_ms", "inflight": {...}}]` JSONL rows), `master_info_before/after.json` payloads, SLO batch summary fields |
| `master.log` | `flexlb_logs/application.log` verbatim prefix with `flexlb.log` (structured dispatch/complete lines), `sync.log`, `sync_consistency.log` and the run-root `flexlb.log` (master stdout) appended |
| `client.json` | `load_client/summary.json` base merged with `server_latency.json`, the full `slo_batch_analysis.json`, and a per-second aggregated timeline (`per_second`, same shape as the canvas aggregation) |
| `client.log` | All `client_shard_*.stdout` merged with `===== client_shard_N =====` separators (single-worker runs rename `client.stdout` directly) |
| `per_request.jsonl` / `.gz` | Merged per-request streams. Under 10 MB total the merge stays plain `per_request.jsonl` (uniform-mode runs — no unpack step needed); larger runs gzip into `per_request.jsonl.gz` (~10x smaller) |

### Second-by-second collectors

`run_online_eval.sh` starts four 1s pollers next to the master counter
poller right before the load clients launch, and stops them at the same point
(right after all clients finish); consolidation merges the files afterwards.
All of them are best-effort — a failed sample or a missing dependency (e.g.
no `ps` binary) only logs a WARNING and never blocks the load test. None of
them needs `curl`; they use Python's urllib. Their output files are one-shot
sources: merged into the component JSON and then deleted, same treatment as
`master_counters_timeseries.txt`.

| File (pre-consolidation) | Source | Lands in |
|---|---|---|
| `mock_metrics_per_engine.prom` | mock control port (`MOCK_BASE_GRPC_PORT-1`) `/metrics?per_engine=true`; the poller keeps only the six analyzer-consumed series per engine (`mock_engine_running` / `waiting` / `active_kv_tokens` / `available_kv_tokens` / `accepted_total` / `completed_total`, C whitelist ≈ ÷4 bytes vs the full ~22-series surface), each sample appended after a `# ts=<epoch_ms>` separator (~2.2KB × N_engines per sample) | `mock_per_engine_timeseries.json.gz` (A-split) |
| `master_prometheus_timeseries.prom` | management port `/actuator/prometheus` (fallback `/prometheus`), whitelisted to the analyzer-consumed series (`flexlb_app_cache_*`, `flexlb_app_flexlb_batcher_queue_size`, `flexlb_app_routing_queue_length`, `flexlb_app_flexlb_inflight_max_age_ms`, `flexlb_app_engine_balancing_master_dispatch_reason_total`, `jvm_memory_used` / `jvm_gc_pause` / `process_cpu` / `system_cpu`), same `# ts=` grouping | `master.json` `prometheus_timeseries` |
| `master_inflight_timeseries.jsonl` | master HTTP port `/rtp_llm/inflight_status`, one JSON line `{"ts_epoch_ms", "inflight"}` per second | `master.json` `inflight_timeseries` |
| `process_usage_timeseries.txt` | `ps -o pid,%cpu,rss,etime` over the mock / master / load-client JVM pids (`ts_epoch_ms=... label=... pid=... cpu_pct=... rss_kb=... etime=...` kv lines; exited pids tolerated) | `run_meta.json` `process_usage` |

Volume sizing (M5, formula instead of the old flat "~5MB"): the three
non-per-engine files stay in the KB-per-second range; the dominant term is
G1 — raw text ≈ **~2.2KB × N_engines per sample** at 1s cadence, i.e.
`2.2KB × N_engines × duration_s` total, which the A-split gzip then
compresses ~4x (the 998MB anchor: 1250 engines × 120s × 1s cadence used to
produce a ~998MB embedded `per_engine_timeseries` in `mock.json`; the same
run now writes a ~65MB `mock_per_engine_timeseries.json.gz`). `MOCK_PER_ENGINE_POLL_INTERVAL_S`
(default 1) scales the per-engine timeline volume without touching the other
collectors; `SECONDARY_POLL_INTERVAL_S` (default 1) retunes all of them. Set
`FLEXLB_SECONDARY_POLLERS_ENABLED=0` to disable all four pollers entirely —
zero observation overhead for A/B comparisons. (The retired Mac-local
`run_stability_test.sh` / `run_burst_test.sh` lines used to pin this to 0;
their scenarios are now covered by the `flexlb_ft/` framework and the remote
skill eval chain.)

### FLEXLB_MONITOR_MODE

The script defaults to `FLEXLB_MONITOR_MODE=all`: the critical-only mode
filters the master's Prometheus exposition down to ~6 `flexlb_*` series,
which drops exactly the KV / inflight / batcher / cache-hit business metrics
the per-second master collector exists for. Explicitly set
`FLEXLB_MONITOR_MODE=critical-only` to restore the trimmed metric set.

Skill-driven runs are **not** affected by this default change: the
flexlb-online-eval skill exports `FLEXLB_MONITOR_MODE=full` unconditionally
(`MONITOR_MODE="${MONITOR_MODE:-full}"` in its launcher), and the Java side
treats any value other than `critical-only` as the full metric surface —
`full` and `all` are equivalent. So skill runs have always collected the
full business metric surface; the `all` default here only matters for
direct invocations of `run_online_eval.sh` that do not set the variable.
`JAVA_MOCK_STATS_INTERVAL_MS` likewise defaults to `1000` (was `5000`) so
`java_mock_stats` lines land at 1s granularity; the mock JVM startup
argument is the only thing that changes.

Kept in place after consolidation:

- `endpoints.json`, `flexlb_env.txt` — discovery artifacts (also snapshotted into `run_meta.json`)
- `flexlb_profile.jfr` — JFR recording, untouched
- `load_client/summary.json` — **kept at the exact legacy path**: the flexlb-online-eval skill's `do_result` reads `run/load_client/summary.json` directly
- `load_client/server_latency.json` — **kept at the exact legacy path**: the skill's `fetch_server_latency` reads that file
- `load_client/report.md` — human-readable summary
- `flexlb_logs/pv.log` — only populated with `FLEXLB_PV_LOG=on` (see below)

The master's per-request `pv.log` (`pvLogger` in `logback-spring.xml`) is
**off by default**: the master is started with
`--logging.level.pvLogger=WARN`, so INFO-level per-request lines are
suppressed and the file is **kept empty by default** (logback's
FileAppender pre-creates it at startup) — only ERROR-level entries for
failed requests still land in it. `FLEXLB_START_CMD` mode is not covered:
a user-supplied start command does not get the property injected. Set
`FLEXLB_PV_LOG=on` to keep the full pv log (a Spring Boot command-line
property passed to the process under test — no production code change);
the file then survives consolidation untouched. Note the skill-driven
path needs `FLEXLB_PV_LOG` added to the skill script's explicit env
export whitelist before it takes effect there.

`consolidate_run_outputs.py` is idempotent and retro-runnable — it can be
re-run on an already consolidated directory (no-op; a regenerated
`slo_batch_analysis.json` only refreshes the `slo_batch_summary` keys) or
applied to a legacy run directory to produce the same layout. Legacy fat
`mock.json` files (embedded `per_engine_timeseries`) are migrated by the
A-split on the next consolidation that rewrites `mock.json`; the unified
analyzer reads both layouts (`.json.gz` first, embedded key, then the raw
`.prom`), so old runs stay fully analyzable. The
consumers (`analyze_slo_batch.py`, `aggregate_canvas_run.py`) read the
**legacy source files first** and fall back to the consolidated ones —
a successful consolidation deletes the legacy files, so a legacy file that
is present always means fresher data (RUN_DIR reuse), and **pre-consolidation
run directories remain fully analyzable**.

One skill caveat: `fetch_error_detail` in the current flexlb-online-eval
skill still reads the per-shard `load_client/shard_*/per_request.jsonl`
files, which consolidation deletes — upgrade the skill to read the run-root
`per_request.jsonl[.gz]`; until then error-detail retrieval on consolidated
runs degrades (summary metrics are unaffected).

## Manual flow

### 1. Start mock engines

```bash
java -jar rtp_llm/flexlb/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar \
  --n-prefill 2 \
  --n-decode 4 \
  --base-grpc-port 55151 \
  --performance rtp_llm/flexlb/tools/online_eval/data/performance/dsv4_flash_performance.sample.json \
  --master-config rtp_llm/flexlb/tools/online_eval/data/config/master_fixed_window.json \
  --endpoint-file rtp_llm/flexlb/tools/online_eval/run/endpoints.json \
  --env-file rtp_llm/flexlb/tools/online_eval/run/flexlb_env.txt
```

`--endpoint-file`, `--performance`, and `--master-config` are required by the
Java CLI. The cluster writes:

- `rtp_llm/flexlb/tools/online_eval/run/endpoints.json`
- `rtp_llm/flexlb/tools/online_eval/run/flexlb_env.txt`

The cluster also serves an HTTP control API on `--base-grpc-port - 1`
(55150 here); see `flexlb-mock-engine/README.md` for the endpoint schema.

The orchestration scripts source `lib_load_client.sh`, which provides the
lifecycle helpers `start_java_mock_cluster <run_dir>` / `wait_mock_cluster_ready
<base_port> <n_engines>` / `mock_http <method> <port> <path> [json_body]` /
`stop_java_mock_cluster <run_dir>` (cluster tuned via `MOCK_*` environment
variables) — prefer those over hand-rolled `java -jar` invocations.

Use the `env ... <your-flexlb-api-start-command>` snippet from
`flexlb_env.txt` when starting `flexlb-api`. The `DOMAIN_ADDRESS:*`
environment keys contain `:`, so they must be passed through `env`; bash cannot
`export` them directly.

### 2. Start flexlb-api

Start the full SpringBoot `flexlb-api` with the environment variables generated
by the mock cluster. FlexLB's own gRPC port is `server.port + 2`.

Backend mock engines use the rtp-llm convention `http_port + 1 == grpc_port`.
The generated service route uses `"protocol": "http"`, so FlexLB treats the
service discovery port as the engine HTTP port and derives gRPC as `http + 1`.

### 3. Run load client

```bash
TRACE_FILE=rtp_llm/flexlb/tools/online_eval/data/online_logs/trace_30min.jsonl \
TARGET_ADDR=127.0.0.1:7001 \
REPLAY_SPEED=10 \
LIMIT=1000 \
OUTPUT_DIR=rtp_llm/flexlb/tools/online_eval/run/load_client \
java -cp rtp_llm/flexlb/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar \
  org.flexlb.mockengine.JavaLoadClient
```

The client is configured entirely through environment variables
(`JavaLoadClient.Config.fromEnv`); the full list lives in
`flexlb-mock-engine/README.md`. `lib_load_client.sh`'s
`run_java_load_client VAR=value ...` wrapper is the single source of truth
for that mapping (unpassed vars are blanked so no ambient environment leaks
in).

For master-enqueued batch requests (BATCH dispatcher), the client follows the
frontend behavior: it calls `FetchResponse` on the selected prefill engine. For
frontend-sent requests (NON_BATCH dispatcher), it calls `GenerateStreamCall`
directly on the routed prefill engine.

Migration note (2026-08, task #55): the legacy v1 `--mode batch|direct|queue`
axis no longer exists — all three v1 modes mapped to the same v2 configuration,
so the mode axis was dead. The functional-test runner now selects a scheduling
profile (`--profile batch-window|single-nonbatch|single-batch|window-nonbatch`:
QUEUE + FIFO ordering × SINGLE/FIXED_WINDOW decision × BATCH/NON_BATCH
dispatcher, injected as the schema-v2 `FLEXLB_CONFIG` document). The v1
"direct" mode name was doubly misleading: v1 direct still routed through the
master (only the delivery leg was frontend-sent), and in v2 frontend-sending
is a dispatcher axis (`non_batch`), not a scheduling mode.

Outputs:

- `summary.json`: throughput, latency percentiles, SLA violations, load balance.
- `per_request.jsonl`: one row per request with routing and latency details.
- `report.md`: readable report for comparing FlexLB configs.

## Validation

```bash
python3 -m unittest discover -s rtp_llm/flexlb/tools/online_eval/tests
```

Raw online logs must not be committed. Generate sanitized fixtures before adding data:

```bash
node rtp_llm/flexlb/tools/online_eval/sanitize_online_log_fixtures.mjs \
  /path/to/raw/online_logs \
  rtp_llm/flexlb/tools/online_eval/data/online_logs
```

The sanitizer drops request/header identity data, converts timestamps to relative
time, pseudonymizes endpoints and block hashes, and remaps plus shuffles token IDs.
The random mapping is not written to disk.

## Capacity reading

Use `completed_qps`, not only `offered_qps`, as the throughput signal. A config
is healthy only if:

- `completed_qps` tracks offered load;
- TTFT p99 stays under the target SLA;
- error and timeout rate remain low;
- prefill/decode load distribution is not strongly skewed;
- mock engine running/available-KV snapshots do not show unbounded backlog.

For capacity search, run the same trace with increasing `REPLAY_SPEED` or use
different trace slices. The practical capacity point is the highest completed
QPS before TTFT p99, error rate, or queue backlog bends upward.

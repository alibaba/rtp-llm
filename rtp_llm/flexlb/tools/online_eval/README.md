# FlexLB Online Evaluation

This tool chain evaluates a running SpringBoot `flexlb-api` against a mock
rtp-llm engine cluster. The mock engine cluster and the load client are
Java-only (`flexlb-mock-engine`, JDK 21+); the Python mock engine / Python
load client implementations have been removed. The retained Python tools —
the smoke client family (`flexlb_smoke_base.py`,
`priority_preemption_smoke.py`) and the
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

- `aggregate.json` (run-level derived metrics incl. `summary.test_valid`,
  the run-validity verdict; written by the in-run aggregate step; since
  20260901 also carries the client-side token throughput fields
  `summary.input_token_tps` / `output_token_tps` (completed-request caliber)
  and `summary.cache_saved_tokens` (cumulative engine-side cache-reuse
  tokens), the per-second arrival-caliber `input_tokens` / `output_tokens`
  columns, and the mock self-reported production-caliber TPS timeline
  `mock_tps_ts` — the report generator's 2.3 P/D role TPS charts consume
  it (P role = the context with/without-cache pair, D role = generate;
  same read as the production dashboard's hippo_role split, engine-
  self-reported only — no client-side TPS concept; mock TPS values are
  accounting-style simulation readings over a fixed 1s window, not GPU
  compute). The 2.3 charts plot the **per-engine average** (cluster sum
  ÷ engine count — the production dashboard's single-instance-series
  read; the former cluster-sum presentation made a 12P run read p50
  3.88M tok/s vs ~58k per production instance, a 67x apparent
  magnitude mismatch). Engine-count chain, most reliable first:
  sibling `run_meta.json` `params.n_prefill` / `n_decode` (deployed
  config truth) > `engine_dist.engine_count` (observed engines that
  received traffic) > `mock.json` final_snapshot engines counted by
  role; when none is available the charts fall back to the cluster sum
  with an explicit 「集群和（引擎数未知）」 caption and a stderr warning
  (standard runs always expose the count; the fallback is defensive
  only). `mock_tps_ts` itself keeps the raw cluster sums — the division
  is a presentation-layer unit choice (same class as k/M rescaling).
  The 20260901 correction moved the client-side token
  reconciliation from report panels into the fail-closed validity item
  `validity_checks.token_reconciliation_ok` (detects dropped requests /
  inflated self-reporting): per input/output side,
  `|client completed tokens − (Σ mock_tps_ts + in-flight Σ)| ≤ max(5% ×
  client, 5 × peak per-second tokens)` — the in-flight term reuses the
  engine-terminal rid sets (`mock_prefill_done` / `mock_decode_done`, the
  same join full_e2e/engine_exec uses): ok rows whose rid is absent from
  the done set were still in flight at run end (fire-and-forget runs
  record ok at schedule success with the expected output_len, so their
  tokens never enter the mock Σ — measured 7.1M / 15.3% of client output
  tokens on run 20260901_200108), and their Σil/Σol joins the mock side
  symmetrically (input joins the prefill done set, output the decode one;
  runs without engine terminal logs degrade to in-flight = 0, keeping the
  legacy formula and the `null` semantics). The 5% relative term absorbs
  scrape-window edge / clock-alignment residue and cancelled-request
  one-sided accounting (healthy runs measure ~1%), the 5 × peak absolute
  term bounds the post-scrape drain tail (completed after the last scrape
  but never drained); missing mock TPS data → `null` (no false failure),
  and `summary.test_valid` aggregates it via `all`; the per-side
  client/mock/in-flight/residual/tolerance numbers are surfaced in
  `summary.token_reconciliation` for forensics); since 20260902 also the
  engine-side KV v2 block-pool timeline `kv_blocks_ts_by_role` —
  `{role: [{t, total/available/held/referenced_blocks (three-state
  gauges), cache_evictions / kv_admission_fails / lack_mem_rejects /
  decode_reuse_blocks (cumulative counters)}]}` summed across engines per
  role (raw cluster sums; the canvas 5. KV engine-side block-pool panels
  divide by the engine count for the per-engine average via the same
  three-level chain as the TPS charts and render the counter columns as
  adjacent-bucket cumulative diffs ÷ bucket gap with counter resets
  clamped to zero — the prefill-602-rejection vs decode-degradation split,
  decode reuse as the fix #5 net-demand deduction readout, evictions as
  the allocation-coupled LRU pressure readout; healthy runs keep both
  admission surfaces at zero, non-zero = overload signal; old aggregates
  without the series → the whole panel group silently omitted)
- `run_meta.json`, `mock.json` / `mock.log`, `master.json` / `master.log`,
  `client.json` / `client.log` (one JSON + one log per component)
- `per_request.jsonl` (or `per_request.jsonl.gz` for larger runs)

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
| `client.json` | `server_latency.json` + the full `slo_batch_analysis.json` + `per_request_source` row-count metadata (Phase B: the legacy `load_client/summary.json` base and the `per_second` timeline are gone — derived statistics live in `aggregate.json`) |
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
| `mock_metrics_per_engine.prom` | mock control port (`MOCK_BASE_GRPC_PORT-1`) `/metrics?per_engine=true`; the poller keeps only the analyzer-consumed series per engine — the queue-depth pair (`mock_engine_running` / `waiting`), the production-caliber TPS trio (`rtp_llm_context_tps` / `rtp_llm_context_tps_with_cache` / `rtp_llm_generate_tps`), the KV v2 block-pool family (`mock_engine_cache_blocks` / `available_blocks` / `held_blocks` / `referenced_blocks` + `mock_engine_cache_evictions_total` / `kv_admission_fails_total` / `lack_mem_rejects_total` / `decode_reuse_blocks_total`) and the cache key-hit pair (`mock_engine_cache_key_hits_total` / `mock_engine_cache_keys_requested_total`, the production recent_cache_key_hit caliber) — 15 series total, every entry with a downstream consumer (aggregate `mock_tps_ts` / `kv_blocks_ts_by_role` / `cache_hit_ts` → the report-layer 2.3 / 5. KV / 5c cache hit-rate panels; no dead keys; C whitelist ≈ ÷4 bytes vs the full ~29-series surface), each sample appended after a `# ts=<epoch_ms>` separator (~2.2KB × N_engines per sample) | `mock_per_engine_timeseries.json.gz` (A-split) |
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
flexlb-mock-engine-test skill exports `FLEXLB_MONITOR_MODE=full` unconditionally
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
- `aggregate.json` — run-level derived metrics written by the in-run aggregate step (`aggregate_canvas_run.py`); `summary.test_valid` is the run-validity verdict (`false` → `INVALID PERFORMANCE RUN`, exit 1; a missing or unparsable file is a WARNING only). Phase B: the client no longer writes `load_client/summary.json` / `load_client/report.md`
- `load_client/server_latency.json` — **kept at the exact legacy path**: the skill's `fetch_server_latency` reads that file
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

- `per_request.jsonl`: one row per request with routing and latency details.
- Derived statistics (throughput, latency percentiles, validity) are no
  longer client outputs — drive the run through `run_online_eval.sh` (its
  in-run aggregate step writes `aggregate.json`) or run
  `aggregate_canvas_run.py` on the run directory afterwards.

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

## Cache hit-rate calibers

Since 20260902 the canvas report carries a cache hit-rate observation built
from THREE calibers plus two gap semantics. The master's own metric family is
untouched — everything master-side is parsed from the existing whale-lb
`flexlb_app_cache_*` series already whitelisted in G3; the only new
instrumentation is the engine's cache key-hit counter pair (see the
flexlb-mock-engine README):

- **master routing caliber** — "how much the master thinks it can reuse":
  the `flexlb_app_cache_routing_selected_match_hit_tokens_total` /
  `flexlb_app_cache_routing_selected_match_total_tokens_total` counter pair
  from `master.json`'s Prometheus timeseries (adjacent-sample positive diffs
  per window, counter resets clamped away; run level = last-sample ratio).
  Aligns with the production whale-lb `app.cache` family.
- **engine key-level (theoretical)** — "matched keys / requested keys":
  `mock_engine_cache_key_hits_total` / `mock_engine_cache_keys_requested_total`
  (booked at prefill admission's `prefixHitBlocks` call; empty-block-hash
  requests contribute 0/0 by construction). Aligns with production
  `recent_cache_key_hit_count / total_count`.
- **engine token-level (actual)** — "ΣhitTokens / Σ input tokens": run
  level = `summary.cache_saved_tokens` ÷ ok-row Σinput_len; the timeline
  derives `(context_tps_with_cache − context_tps) / context_tps_with_cache`
  per `mock_tps_ts` window (engine-side P-completion accounting, so numerator
  and denominator share the window). Aligns with production reuse/input.

The 5c panels render: (1) "master routing vs engine execution" dual line —
the gap = **scheduling loss** (the master matched a prefix but the engine did
not reuse it: routed to a non-holding engine / affinity not honored / LRU
eviction inside the execution window); (2) "key-level (theoretical) vs
token-level (actual)" dual line — the gap = **hit-depth coverage** (partial
prefix hits: a key matched but the prefix broke at block N, so the reused
tokens fall short of the matched key count); (3) a run-level summary bar
chart whose caption names the production-alignment mapping. Fail-closed:
whenever a panel is rendered the HTML must carry its scope annotations
(「master 路由口径」「engine 执行口径」「调度损耗」 / 「key 级理论口径」「token
级实际口径」「命中深度覆盖」 / 「对齐生产」). Old aggregates without the new
counters/series degrade per caliber — a missing caliber simply drops its
column/panel, never fabricates 0.

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

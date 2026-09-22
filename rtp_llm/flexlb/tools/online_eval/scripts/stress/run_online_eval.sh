#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Root aliases and the canonical stress/ entry share one implementation.
ONLINE_EVAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FLEXLB_DIR="$(cd "${ONLINE_EVAL_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${FLEXLB_DIR}/../.." && pwd)"

# Shared JavaLoadClient helpers: env-var mapping (run_java_load_client) and
# JDK 21 detection (java_major/detect_java21_home/require_java21).
# JAVA_LOAD_CLIENT_HEAP_SIZE (historical knob) feeds the lib's Xmx default;
# export either JAVA_LOAD_CLIENT_HEAP_SIZE or JAVA_LOAD_CLIENT_JVM_XMX/XMS
# before this script runs to override the load client JVM sizing.
JAVA_LOAD_CLIENT_JVM_XMX="${JAVA_LOAD_CLIENT_JVM_XMX:-${JAVA_LOAD_CLIENT_HEAP_SIZE:-16g}}"
source "${SCRIPT_DIR}/lib/load_client.sh"

FLEXLB_NETWORK_ISOLATED="${FLEXLB_NETWORK_ISOLATED:-0}"
if [[ "${FLEXLB_NETWORK_ISOLATED}" == "1" \
      && "${FLEXLB_NETWORK_NAMESPACE_ACTIVE:-0}" != "1" ]]; then
  exec unshare -Urn bash -c \
    'ip link set lo up; export FLEXLB_NETWORK_NAMESPACE_ACTIVE=1; exec "$@"' \
    bash bash "$0" "$@"
fi
FLEXLB_FAIL_ON_CONCURRENT_TEST="${FLEXLB_FAIL_ON_CONCURRENT_TEST:-1}"

# Fail fast on the removed Python implementation switches: the Python mock
# engine and Python load client no longer exist on this branch (Java-only),
# so a stale ambient value must be a loud error instead of a silent
# fallback to the Java stack. The switch definitions themselves were
# deleted, hence the ":-" reads to stay safe under set -u.
if [[ "${LOAD_CLIENT_IMPL:-}" == "python" || "${MOCK_ENGINE_IMPL:-}" == "python" ]]; then
  echo "ERROR: LOAD_CLIENT_IMPL/MOCK_ENGINE_IMPL=python is no longer supported: the Python mock engine and Python load client implementations have been removed on this branch (Java-only). Unset the variable(s) to run the Java stack." >&2
  exit 1
fi

TRACE_FILE="${TRACE_FILE:-${ONLINE_EVAL_DIR}/data/online_logs/trace_30min.jsonl}"
PERFORMANCE_FILE="${PERFORMANCE_FILE:-${ONLINE_EVAL_DIR}/data/performance/dsv4_flash_performance.fast_ab.json}"
RUN_ROOT="${RUN_ROOT:-${ONLINE_EVAL_DIR}/run}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
# Evidence is task-specific; all performance curves come from Prometheus.
export COLLECTION_PROFILE="${COLLECTION_PROFILE:-aggregate}"
export CLIENT_MONITORING=true
if [[ "${COLLECTION_PROFILE}" != diagnostic ]]; then export SKIP_SERVER_LATENCY=true; fi
case "${COLLECTION_PROFILE}" in aggregate|request|diagnostic) ;; *) echo "invalid COLLECTION_PROFILE" >&2; exit 2;; esac
command -v "${PROMETHEUS_BIN:-prometheus}" >/dev/null || { echo "Prometheus required (set PROMETHEUS_BIN)" >&2; exit 2; }
MONITOR_PID=""
MOCK_EVENTS_FILE=""
MOCK_STATS_STDOUT=false
if [[ "${COLLECTION_PROFILE}" == diagnostic ]]; then
  MOCK_EVENTS_FILE="${RUN_DIR}/engine_events.jsonl"
  MOCK_STATS_STDOUT=true
fi
# Optional portable archive. Keep disabled for the default fast path.
EXPERIMENT_ARCHIVE_PATH="${EXPERIMENT_ARCHIVE_PATH:-}"
FLEXLB_LOG_PATH="${FLEXLB_LOG_PATH:-${RUN_DIR}/flexlb_logs}"
# FLEXLB_CONFIG SSOT knobs: tools/online_eval/flexlb_cfg.py renders BOTH
# the master env string and the --master-config envelope (single render,
# two projections — the channels cannot diverge).  FLEXLB_PROFILE picks
# the base document (default stress-na130 = the retired
# data/config/master_fixed_window.json semantics, field for field);
# FLEXLB_CONFIG_OVERRIDE="k=v,..." layers per-field overrides
# (flexlb_cfg.parse_overrides vocabulary); FLEXLB_JVM_HEAP_SIZE rides the
# envelope (master -Xms/-Xmx via the heap extraction below).  An
# explicitly exported FLEXLB_CONFIG still wins for the master env
# (escape hatch — see the generator block above START_MOCK).
FLEXLB_MASTER_MODE_EXPLICIT="${FLEXLB_MASTER_MODE+x}"
FLEXLB_PROFILE_EXPLICIT="${FLEXLB_PROFILE+x}"
if [[ -z "${FLEXLB_MASTER_MODE_EXPLICIT}" && -n "${FLEXLB_PROFILE:-}" ]]; then
  FLEXLB_MASTER_MODE="$(python3 "${ONLINE_EVAL_DIR}/mode_profiles.py" \
    --runtime stress --profile-to-master "${FLEXLB_PROFILE}")"
fi
FLEXLB_MASTER_MODE="${FLEXLB_MASTER_MODE:-wb}"
MODE_PROFILE="$(python3 "${ONLINE_EVAL_DIR}/mode_profiles.py" \
  --runtime stress --master "${FLEXLB_MASTER_MODE}" --field master_profile)"
if [[ -n "${FLEXLB_MASTER_MODE_EXPLICIT:-}" && -n "${FLEXLB_PROFILE:-}" \
      && "${FLEXLB_PROFILE}" != "${MODE_PROFILE}" ]]; then
  echo "ERROR: FLEXLB_PROFILE disagrees with explicit FLEXLB_MASTER_MODE" >&2
  exit 2
fi
FLEXLB_PROFILE="${FLEXLB_PROFILE:-${MODE_PROFILE}}"
FLEXLB_CONFIG_OVERRIDE="${FLEXLB_CONFIG_OVERRIDE:-}"
FLEXLB_JVM_HEAP_SIZE="${FLEXLB_JVM_HEAP_SIZE:-32g}"
PROCESS_CONFIG_FILE="${PROCESS_CONFIG_FILE:-${RUN_DIR}/master_config.json}"

# Default load (user-approved 2026-09-02, replay profile): 12P/40D + replay
# mode (trace-timestamp pacing), duration 120s; cyclic refill requires explicit LOOP=1.
# REPLAY_SPEED is caller-supplied (the upstream orchestrator
# auto-calibrates it from the trace to the nominal 650 QPS target); this
# script never invents a speed. Baseline break (2026-09-02): the mock3
# uniform-650 default is retired to an explicit opt-in (SEND_MODE=uniform
# SEND_MODE_QPS=650) — numbers from the uniform-default era are NOT directly
# comparable with replay-default runs.
N_PREFILL="${N_PREFILL:-12}"
N_DECODE="${N_DECODE:-40}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-61000}"
# NOTE: the three assignments below (JAVA_MOCK_ENGINE_JAR, JAVA_LOAD_CLIENT_JAR,
# MAVEN_PROFILES) duplicate defaults already applied by lib/load_client.sh at
# source time (same values), so they are no-ops here. They are kept as
# self-documentation of this script's tunable knobs; the effective defaults
# live in the lib.
JAVA_MOCK_ENGINE_JAR="${JAVA_MOCK_ENGINE_JAR:-${FLEXLB_DIR}/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar}"
JAVA_LOAD_CLIENT_JAR="${JAVA_LOAD_CLIENT_JAR:-${FLEXLB_DIR}/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar}"  # no-op see NOTE above
JAVA_LOAD_CLIENT_HEAP_SIZE="${JAVA_LOAD_CLIENT_HEAP_SIZE:-16g}"
JAVA_MOCK_EVENT_LOOP_THREADS="${JAVA_MOCK_EVENT_LOOP_THREADS:-32}"
JAVA_MOCK_COMPLETION_THREADS="${JAVA_MOCK_COMPLETION_THREADS:-16}"
# java_mock_stats sampling interval, passed straight to --stats-interval-ms
# (single env, no renaming). Default 1000ms: the unified analysis needs a
# 1s-granularity mock timeline across the pressure window (the historical
# default was 5000). Set JAVA_MOCK_STATS_INTERVAL_MS=5000 to restore the
# coarse cadence.
JAVA_MOCK_STATS_INTERVAL_MS="${JAVA_MOCK_STATS_INTERVAL_MS:-1000}"
# Passed straight to --decode-max-concurrency (single env, no renaming).
# Default matches the mock engine's DEFAULT_DECODE_MAX_CONCURRENCY (132).
# The hard admission gate is unconditional (production semantics): once the
# cap is reached, excess decode requests park in the engine-side waiting
# queue (reported as decode_waiting; with report_queued_as_kv_allocated they
# surface in the KV_ALLOCATED/accepted layer).
# 128 = CONCURRENCY_LIMIT-aligned (production anchor; previously 132).
JAVA_MOCK_DECODE_MAX_CONCURRENCY="${JAVA_MOCK_DECODE_MAX_CONCURRENCY:-128}"
JAVA_MOCK_ENGINE_HEAP_SIZE="${JAVA_MOCK_ENGINE_HEAP_SIZE:-32g}"
JAVA_MOCK_JVM_XMS="${JAVA_MOCK_JVM_XMS:-${JAVA_MOCK_ENGINE_HEAP_SIZE}}"
JAVA_MOCK_JVM_XMX="${JAVA_MOCK_JVM_XMX:-${JAVA_MOCK_ENGINE_HEAP_SIZE}}"
ENDPOINT_READY_TIMEOUT_S="${ENDPOINT_READY_TIMEOUT_S:-120}"
# Per-role KV pool BLOCK counts, passed straight to
# --prefill-kv-pool-blocks / --decode-kv-pool-blocks (the env-var names
# keep the historical "cache blocks" wording; the value has been the
# total pool block count since KV v2, not a key count).
PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS:-6000}"
DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS:-3000}"

FLEXLB_HTTP_ADDR="${FLEXLB_HTTP_ADDR:-127.0.0.1:7001}"
FLEXLB_HTTP_PORT="${FLEXLB_HTTP_ADDR##*:}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-7002}"
FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"
START_FLEXLB="${START_FLEXLB:-1}"
START_MOCK="${START_MOCK:-1}"
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"  # no-op, see NOTE above (lib default)

LIMIT="${LIMIT:-1000}"
DURATION_S="${DURATION_S:-120}"
# REPLAY_SPEED: replay pacing multiplier, pure pass-through. The caller is
# expected to supply it (the orchestrator auto-calibrates from the trace:
# SPEED = max(1, round(target_qps * valid_span_s / valid_requests)), where
# valid = ol>0 rows are filtered client-side). The
# bare 10 fallback below equals the JavaLoadClient built-in default and only
# applies to direct invocations that omit the variable; it is NOT calibrated.
REPLAY_SPEED="${REPLAY_SPEED:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-999999999}"
TIMEOUT_MS="${TIMEOUT_MS:-3600000}"
SLA_TTFT_MS="${SLA_TTFT_MS:-500}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-0}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-0}"
GRADIENT="${GRADIENT:-0}"
GRADIENT_MAX_SPEED="${GRADIENT_MAX_SPEED:-1000}"
GRADIENT_START_SPEED="${GRADIENT_START_SPEED:-10}"
# Client-side output-stream reading: 1 (default) = the load client reads engine
# output streams after Schedule; 0 = skip client stream reads while the engine
# still executes prefill+decode in full (BATCH dispatcher only).
FETCH_OUTPUT_STREAM="${FETCH_OUTPUT_STREAM:-1}"
# One switch controls both ends: skipping Fetch is an explicit mock shortcut.
case "${FETCH_OUTPUT_STREAM,,}" in
  0|false) JAVA_MOCK_AUTO_FETCH=true ;;
  1|true) JAVA_MOCK_AUTO_FETCH=false ;;
  *) echo "FETCH_OUTPUT_STREAM must be 0/1/false/true" >&2; exit 1 ;;
esac
# Client fallback (direct-to-engine escape hatch on Schedule-RPC / stream-read
# failure): default OFF and it must stay OFF for load tests — a fallback send
# bypasses the master entirely (admission, routing, schedule-latency leg), so
# any fallback traffic pollutes the calibers the aggregator computes.
# ENABLE_FALLBACK=1 (plus ENDPOINTS_FILE=<endpoints.json>) is the explicit
# opt-in for case-test-style direct-connect scenarios only; this script never
# turns it on by itself — the default below is 0 and a deliberate opt-in
# (exporting ENABLE_FALLBACK=1) is the only way to enable it.
ENABLE_FALLBACK="${ENABLE_FALLBACK:-0}"
# FORCE_PRIORITY pins every replayed request to one Auto-TPM QoS level,
# overriding both the per-record trace priority and the PRIORITY env default.
# Defaults to 50 (single-QoS baseline runs): all requests share one priority,
# so priority-based preemption finds no victim and behaves as if disabled
# (the scheduler.ordering.preemption config block may stay — behaviour is
# equivalent under a uniform priority). Multi-priority experiments opt out
# explicitly with FORCE_PRIORITY=0 so per-record trace priority wins.
FORCE_PRIORITY="${FORCE_PRIORITY:-50}"
# LOOP=1 explicitly opts into cyclic refill; default is one finite pass.
# uniform mode also obeys explicit LOOP/MAX_LAPS; it no longer implies looping.
LOOP="${LOOP:-0}"
# Send mode is a pure pass-through (single env-var layer). Default replay
# (trace-timestamp pacing; user-approved 2026-09-02, speed auto-calibrated
# upstream). uniform is the explicit opt-in for the strictest scheduling
# regime / pressure-boundary / capacity-critical scans and requires
# SEND_MODE_QPS. Baseline break: pre-2026-09-02 uniform-default runs are not
# directly comparable with replay-default runs.
SEND_MODE="${SEND_MODE:-replay}"
SEND_MODE_QPS="${SEND_MODE_QPS:-650}"
# Traffic ramp-up for uniform mode: QPS climbs linearly 0 -> SEND_MODE_QPS
# over RAMP_UP_SECONDS, then stays constant (replay mode ignores ramp-up).
# Default 30s (mock3-aligned uniform-opt-in tier); 0 disables it. Distinct
# from FLEXLB_WARMUP_SECONDS above
# (the no-traffic prepare sleep before load starts): ramp-up shapes the
# arrival process once traffic begins.
RAMP_UP_SECONDS="${RAMP_UP_SECONDS:-30}"
PUSHGATEWAY_URL="${PUSHGATEWAY_URL:-}"
LOAD_CLIENT_WORKERS="${LOAD_CLIENT_WORKERS:-8}"
LOAD_CLIENT_START_DELAY_SECONDS="${LOAD_CLIENT_START_DELAY_SECONDS:-10}"
CLIENT_PACING_LAG_P99_LIMIT_MS="${CLIENT_PACING_LAG_P99_LIMIT_MS:-100}"
SLO_BATCH_DRAIN_SECONDS="${SLO_BATCH_DRAIN_SECONDS:-0}"
# The master's pvLogger writes a per-request pv.log under FLEXLB_LOG_PATH
# (logback-spring.xml "pvLogger" -> PV appender). That per-request telemetry
# is not consumed by the consolidation flow, so by default the master is
# started with --logging.level.pvLogger=WARN: INFO-level per-request lines
# are suppressed and the file is kept EMPTY by default (logback's
# FileAppender pre-creates it at startup) — only ERROR-level entries for
# failed requests still land in it. Both effects come from a Spring Boot
# command-line property passed to the process under test — no production
# code change. FLEXLB_START_CMD mode is not covered: a user-supplied start
# command does not get the property injected. Set FLEXLB_PV_LOG=on to keep
# the full pv log; the file then survives consolidation untouched (see
# analysis/consolidate.py).
FLEXLB_PV_LOG="${FLEXLB_PV_LOG:-off}"
JFR_FILE="${JFR_FILE:-${RUN_DIR}/flexlb_profile.jfr}"
JFR_DURATION="${JFR_DURATION:-300s}"
FLEXLB_MONITOR_ENABLED="${FLEXLB_MONITOR_ENABLED:-true}"
# Limit metric families at the producer; Prometheus stores the exposed series.
# A blank whitelist exposes no flexlb metrics; flexlb_ exposes all families.
FLEXLB_MONITOR_METRIC_WHITELIST="${FLEXLB_MONITOR_METRIC_WHITELIST:-flexlb_app_cache_,flexlb_app_flexlb_batcher_queue_size,flexlb_app_flexlb_inflight_max_age_ms,flexlb_app_flexlb_inflight_ttl,flexlb_app_engine_balancing_master_dispatch_reason_total,flexlb_app_engine_balancing_master_batch_size,flexlb_auto_tpm_request_count,flexlb_app_engine_balancing_master_all_qps,flexlb_app_flexlb_scheduler_inflight_size,flexlb_app_flexlb_inflight_batch_count,flexlb_app_flexlb_inflight_request_count,flexlb_auto_tpm_decode_reserved_count,flexlb_auto_tpm_decode_running_count}"
# HIPPO_ROLE: ZK election role id of the master (lock path
# /master_lb_leader/{HIPPO_ROLE}); a blank value aborts master startup
# (ZookeeperMasterElectService / LBStatusConsistencyService). The eval
# line runs a single master without ZK, so the default is just a
# non-empty label. This is the ONLY default-assignment site.
HIPPO_ROLE="${HIPPO_ROLE:-test}"

# FLEXLB_CONFIG (the scheduler document) is GENERATED by flexlb_cfg.py at
# start time — the retired inline default JSON above and the static
# data/config/master_fixed_window.json template are both superseded by
# the FLEXLB_PROFILE/FLEXLB_CONFIG_OVERRIDE knobs; the generator block
# above START_MOCK materializes the env string and the envelope file in
# one render.
OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN:-.*}"
OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-none}"

# Optional file-based service discovery (dynamic engine add/remove).
# Empty (default) = disabled: mock engine keeps the env-file (NoOp discovery)
# path, master falls back to NoOpServiceDiscovery — behavior unchanged.
# Set to a path (or "auto"/"1" = ${RUN_DIR}/discovery.json when START_MOCK=1)
# to enable: the mock engine writes the domain→hosts mapping via
# --discovery-file (kept in sync by /add_engine + /remove_engine) and the
# master consumes it via FLEXLB_DISCOVERY_FILE (→ flexlb.discovery.file).
FLEXLB_DISCOVERY_FILE="${FLEXLB_DISCOVERY_FILE:-}"

# These are independent transport settings, not FLEXLB_CONFIG fields.
export FLEXLB_GRPC_EXECUTOR_CORE_SIZE="${FLEXLB_GRPC_EXECUTOR_CORE_SIZE:-128}"
export FLEXLB_GRPC_EXECUTOR_MAX_SIZE="${FLEXLB_GRPC_EXECUTOR_MAX_SIZE:-128}"
# FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE: no script default — code default (1000) applies
# unless the caller exports it explicitly (still forwarded via the environment).

MOCK_PID=""
FLEXLB_PID=""
CLIENT_PIDS=()
JAVA_MODULE_OPTS=(
  --add-modules ALL-SYSTEM
  --add-opens java.base/java.lang=ALL-UNNAMED
  --add-opens java.base/java.lang.invoke=ALL-UNNAMED
  --add-opens java.base/java.util=ALL-UNNAMED
  --add-opens java.base/java.util.concurrent=ALL-UNNAMED
  --add-opens=java.base/jdk.internal.misc=ALL-UNNAMED
  --add-opens java.base/java.nio=ALL-UNNAMED
  --add-opens java.base/sun.nio.ch=ALL-UNNAMED
  --add-opens java.instrument/sun.instrument=ALL-UNNAMED
)

# Limit Reactor boundedElastic scheduler threads to prevent thread explosion
JVM_SYSTEM_PROPS=(-Dreactor.schedulers.defaultBoundedElasticSize=64)

# java_major / detect_java21_home are provided by lib/load_client.sh (sourced
# above); do not redefine them here.

JAVA21_HOME_DETECTED="$(detect_java21_home || true)"
if [[ -n "${JAVA21_HOME_DETECTED}" ]]; then
  export JAVA_HOME="${JAVA21_HOME_DETECTED}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

cleanup() {
  local run_exit_status=$?
  # One stop covers all collector threads (G1/G3/G5) of the single
  # secondary collector process.
  stop_monitoring || run_exit_status=1
  for pid in "${CLIENT_PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  if [[ -n "${FLEXLB_PID}" ]]; then
    kill "${FLEXLB_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${MOCK_PID}" ]]; then
    kill "${MOCK_PID}" >/dev/null 2>&1 || true
  fi
  sleep 1
  for pid in "${CLIENT_PIDS[@]}" "${FLEXLB_PID}" "${MOCK_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  done
  if [[ -n "${EXPERIMENT_ARCHIVE_PATH}" && -d "${RUN_DIR}" ]]; then
    local archive_status=incomplete
    [[ "${run_exit_status}" -eq 0 && -s "${RUN_DIR}/aggregate.json" ]] && archive_status=complete
    if PYTHONPATH="${ONLINE_EVAL_DIR}/src:${ONLINE_EVAL_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python3 -m flexlb_eval.artifacts.archive create \
      --kind stress --status "${archive_status}" \
      --source "run=${RUN_DIR}" --out "${EXPERIMENT_ARCHIVE_PATH}" \
      >/dev/null; then
      echo "experiment_archive=${EXPERIMENT_ARCHIVE_PATH} status=${archive_status} run_exit=${run_exit_status}"
    else
      echo "WARNING: experiment archive creation failed: ${EXPERIMENT_ARCHIVE_PATH}" >&2
    fi
  fi
}
trap cleanup EXIT

# Archive the monitor once on either client completion or failure.
CONSOLIDATED=0

# sha256 of a file (Linux sha256sum), with shasum -a 256 (macOS local smoke
# runs) and md5 fallbacks; empty string when the file is missing or no digest
# tool exists. The prefix on the md5 fallbacks keeps the algorithm auditable
# in run_meta.json.
compute_file_digest() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo ""
    return 0
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${file}" | awk '{print $1}'
  elif command -v md5sum >/dev/null 2>&1; then
    echo "md5:$(md5sum "${file}" | awk '{print $1}')"
  elif command -v md5 >/dev/null 2>&1; then
    echo "md5:$(md5 -q "${file}")"
  else
    echo ""
  fi
}

count_file_lines() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo ""
    return 0
  fi
  wc -l <"${file}" | tr -d '[:space:]'
}

consolidate_run_outputs_now() {
  if [[ "${CONSOLIDATED}" == "1" ]]; then
    return 0
  fi
  CONSOLIDATED=1
  # Stop and archive the owned Prometheus session before rendering.
  stop_monitoring
  # No raw-log consolidation or per-request replication on the monitor path.
  return 0
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  python3 - "$host" "$port" "$timeout_s" <<'PY'
import socket
import sys
import time

host, port, timeout_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.time() + timeout_s
last_error = None
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            sys.exit(0)
    except OSError as exc:
        last_error = exc
        time.sleep(0.5)
print(f"timeout waiting for {host}:{port}: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

assert_ports_free() {
  # SO_REUSEADDR lets a check socket bind against a port still in TIME_WAIT
  # (no process listening, but kernel-held) — the common state right after a
  # previous run is killed. Without it socket.bind() gives a false failure.
  # Poll up to 5s for ports to drain; each check binds with SO_REUSEADDR so
  # TIME_WAIT ports pass immediately.
  python3 - "$@" <<'PY'
import socket
import sys
import time

max_wait = 5.0
interval = 0.5
deadline = time.monotonic() + max_wait
last_errors = {}

while True:
    last_errors.clear()
    ok = True
    for raw_port in sys.argv[1:]:
        port = int(raw_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            ok = False
            last_errors[port] = exc
        finally:
            sock.close()
    if ok:
        sys.exit(0)
    if time.monotonic() >= deadline:
        for port, exc in last_errors.items():
            print(f"required port {port} is not available after {max_wait:.0f}s: {exc}", file=sys.stderr)
        sys.exit(1)
    time.sleep(interval)
PY
}

assert_no_concurrent_flexlb_test() {
  local matches
  matches="$(pgrep -af 'flexlb-api-[^ ]*\.jar|flexlb-mock-engine-[^ ]*\.jar' || true)"
  if [[ -n "${matches}" ]]; then
    echo "Concurrent FlexLB performance processes detected on the host:" >&2
    echo "${matches}" >&2
    echo "Wait for them to finish, or set FLEXLB_FAIL_ON_CONCURRENT_TEST=0 to override." >&2
    return 1
  fi
}

wait_for_endpoints_ready() {
  local master_port=$1
  local expected_prefill=$2
  local expected_decode=$3
  local max_wait="${ENDPOINT_READY_TIMEOUT_S}"
  local elapsed=0

  echo "[wait_for_endpoints_ready] Waiting for ${expected_prefill} prefill + ${expected_decode} decode endpoints to be discovered and alive..."

  while [ "${elapsed}" -lt "${max_wait}" ]; do
    local response
    response=$(curl -s -X POST "http://127.0.0.1:${master_port}/rtp_llm/master/info" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d '{}' 2>/dev/null) || true

    if [ -n "${response}" ]; then
      local result
      result=$(echo "${response}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    ready = data.get('ready', False)
    ws = data.get('worker_summary', {})
    prefill = ws.get('PREFILL', {})
    decode = ws.get('DECODE', {})
    p_disc = prefill.get('discovered', 0)
    p_alive = prefill.get('alive', 0)
    d_disc = decode.get('discovered', 0)
    d_alive = decode.get('alive', 0)
    print(f'{ready}|{p_disc}|{p_alive}|{d_disc}|{d_alive}')
except Exception:
    print('False|0|0|0|0')
" 2>/dev/null) || result="False|0|0|0|0"

      local ready p_disc p_alive d_disc d_alive
      IFS='|' read -r ready p_disc p_alive d_disc d_alive <<< "${result}"

      if [ "${ready}" = "True" ] && [ "${p_disc}" -ge "${expected_prefill}" ] && [ "${p_alive}" -ge "${expected_prefill}" ] && [ "${d_disc}" -ge "${expected_decode}" ] && [ "${d_alive}" -ge "${expected_decode}" ]; then
        echo "[wait_for_endpoints_ready] All endpoints ready: prefill=${p_alive}/${expected_prefill}, decode=${d_alive}/${expected_decode} (${elapsed}s)"
        return 0
      fi

      echo "[wait_for_endpoints_ready] Not ready yet: ready=${ready}, prefill discovered=${p_disc}/${expected_prefill} alive=${p_alive}/${expected_prefill}, decode discovered=${d_disc}/${expected_decode} alive=${d_alive}/${expected_decode} (${elapsed}s)"
    fi

    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo "[wait_for_endpoints_ready] ERROR: Timeout after ${max_wait}s waiting for endpoints." >&2
  return 1
}

save_master_info() {
  local output=$1
  curl -fsS -X POST "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/master/info" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{}' >"${output}"
}

SECONDARY_POLL_INTERVAL_S="${SECONDARY_POLL_INTERVAL_S:-1}"

start_monitoring() {
  local targets=()
  if [[ "${START_MOCK}" == 1 ]]; then
    targets+=(--target "mock=http://127.0.0.1:$((MOCK_BASE_GRPC_PORT - 1))/metrics?per_engine=true")
  fi
  if [[ "${START_FLEXLB}" == 1 ]]; then
    targets+=(--target "master-single=http://127.0.0.1:${FLEXLB_MANAGEMENT_PORT}/prometheus")
  fi
  PYTHONPATH="${ONLINE_EVAL_DIR}/src:${ONLINE_EVAL_DIR}" python3 -m flexlb_eval.monitoring.session serve \
    --run-dir "${RUN_DIR}" --interval "${SECONDARY_POLL_INTERVAL_S}" --clients "${LOAD_CLIENT_WORKERS}" "${targets[@]}" \
    >"${RUN_DIR}/monitor.log" 2>&1 &
  MONITOR_PID=$!
  local count=0
  until [[ -f "${RUN_DIR}/monitor-ready" ]]; do
    kill -0 "${MONITOR_PID}" 2>/dev/null || { cat "${RUN_DIR}/monitor.log" >&2; return 1; }
    sleep 0.1
    count=$((count + 1))
    if (( count > 300 )); then echo "monitor startup timed out" >&2; return 1; fi
  done
}

stop_monitoring() {
  if [[ -z "${MONITOR_PID}" ]]; then return 0; fi
  touch "${RUN_DIR}/monitor-stop"
  local monitor_status=0
  wait "${MONITOR_PID}" || monitor_status=$?
  MONITOR_PID=""
  if [[ "${monitor_status}" != 0 || ! -f "${RUN_DIR}/monitor-complete" ]]; then
    echo "Prometheus collection/archive failed" >&2
    return 1
  fi
}

assert_mock_engine_healthy() {
  if [[ "${START_MOCK}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${MOCK_PID}" ]] || ! kill -0 "${MOCK_PID}" >/dev/null 2>&1; then
    echo "Mock engine is not running" >&2
    tail -80 "${RUN_DIR}/mock_engine.log" >&2 || true
    return 1
  fi
  if grep -q "OutOfMemoryError" "${RUN_DIR}/mock_engine.log" 2>/dev/null; then
    echo "Mock engine encountered OutOfMemoryError" >&2
    tail -80 "${RUN_DIR}/mock_engine.log" >&2 || true
    return 1
  fi
}

mkdir -p "${RUN_DIR}"
python3 - "${ONLINE_EVAL_DIR}" "${FLEXLB_MASTER_MODE}" "${FLEXLB_PROFILE}" \
  "${FLEXLB_PROFILE_EXPLICIT}" "${RUN_DIR}/mode_plan.json" <<'PY'
import json
import sys
sys.path.insert(0, sys.argv[1])
from mode_profiles import resolve_mode
plan = resolve_mode("stress", sys.argv[2])
plan["master_profile"] = sys.argv[3]
plan["profile_source"] = "explicit" if sys.argv[4] else "runtime_default"
with open(sys.argv[5], "w", encoding="utf-8") as stream:
    json.dump(plan, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
PY
mkdir -p "${FLEXLB_LOG_PATH}"
echo "run_dir=${RUN_DIR}"
echo "load client: JavaLoadClient (trace priority passthrough via lib/load_client.sh)"
if [[ "$(java_major java)" -lt 21 ]]; then
  echo "Java 21 is required to run JavaLoadClient. Set JAVA21_HOME or JAVA_HOME." >&2
  exit 1
fi
if [[ ! -f "${JAVA_LOAD_CLIENT_JAR}" ]]; then
  echo "Java load client jar not found, auto-building: ${JAVA_LOAD_CLIENT_JAR} (first build may take several minutes)"
  if ! (cd "${FLEXLB_DIR}" && ./mvnw -P"${MAVEN_PROFILES}" -pl flexlb-mock-engine -am package -DskipTests); then
    echo "Failed to build Java load client jar via Maven (this may take several minutes on a cold cache)" >&2
    exit 1
  fi
  if [[ ! -f "${JAVA_LOAD_CLIENT_JAR}" ]]; then
    echo "Failed to build Java load client jar: ${JAVA_LOAD_CLIENT_JAR}" >&2
    exit 1
  fi
fi

if [[ "${FLEXLB_FAIL_ON_CONCURRENT_TEST}" == "1" ]]; then
  assert_no_concurrent_flexlb_test
fi

ENDPOINT_FILE="${RUN_DIR}/endpoints.json"
FLEXLB_ENV_FILE="${RUN_DIR}/flexlb_env.txt"

JAVA_MOCK_DISCOVERY_ARGS=()
MOCK_DISCOVERY_FILE=""
if [[ "${START_MOCK}" == "1" ]]; then
  MOCK_DISCOVERY_FILE="${RUN_DIR}/discovery.json"
  JAVA_MOCK_DISCOVERY_ARGS=(--discovery-file "${MOCK_DISCOVERY_FILE}")
fi

# ---------------------------------------------------------------------------
# FLEXLB_CONFIG SSOT generator (runs before the mock engine boots: the
# --master-config file must exist by then).  One render, two projections:
# the envelope written to PROCESS_CONFIG_FILE (mock engine --master-config
# + heap extraction + run_meta consolidation) and the env string injected
# into the master below — the two channels cannot diverge by construction.
# ---------------------------------------------------------------------------
run_cfg_generator() {
  python3 - "${ONLINE_EVAL_DIR}" "${PROCESS_CONFIG_FILE}" "${FLEXLB_PROFILE}" \
    "${FLEXLB_CONFIG_OVERRIDE}" "${FLEXLB_JVM_HEAP_SIZE}" <<'PY'
import sys, os, json

sys.path.insert(0, sys.argv[1])
from flexlb_cfg import parse_overrides, render_env, render_process_config

out_path, profile, override_spec, heap = sys.argv[2:6]
overrides = parse_overrides(override_spec) if override_spec.strip() else None
doc = json.loads(render_env(profile, overrides))
for env_key, field in (("FLEXLB_GRPC_EXECUTOR_CORE_SIZE", "executorCoreSize"),
                       ("FLEXLB_GRPC_EXECUTOR_MAX_SIZE", "executorMaxSize"),
                       ("FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE", "executorQueueSize")):
    if env_key in os.environ:
        doc.setdefault("grpcServer", {})[field] = int(os.environ[env_key])
raw = json.dumps(doc, separators=(",", ":"))
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(render_process_config(profile, jvm_heap=heap, raw_config=raw))
sys.stdout.write(raw)
PY
}

# Legacy STRIP_PREEMPTION=1 call shape (orchestrator scripts) maps onto the
# override channel: the strip_preemption flag drops
# scheduler.ordering.preemption from the RENDERED document (both
# projections).  The former in-memory JSON edit channel is retired.
STRIP_PREEMPTION="${STRIP_PREEMPTION:-0}"
if [[ "${STRIP_PREEMPTION}" == "1" ]]; then
  if [[ -n "${FLEXLB_CONFIG_OVERRIDE}" ]]; then
    FLEXLB_CONFIG_OVERRIDE="${FLEXLB_CONFIG_OVERRIDE},strip_preemption"
  else
    FLEXLB_CONFIG_OVERRIDE="strip_preemption"
  fi
fi

if [[ -z "${FLEXLB_CONFIG:-}" ]]; then
  FLEXLB_CONFIG="$(run_cfg_generator)"
else
  # Escape hatch: an explicitly exported FLEXLB_CONFIG bypasses the
  # generator for the MASTER ENV ONLY.  The envelope file still carries
  # the RENDERED document (the mock engine needs a well-formed file), so
  # master and mock may run different documents in this mode —
  # caller-owned, warned loudly.
  run_cfg_generator >/dev/null
  echo "WARNING: explicit FLEXLB_CONFIG is set — the master env uses it verbatim while ${PROCESS_CONFIG_FILE} keeps the ${FLEXLB_PROFILE} render (possible two-source divergence)." >&2
fi

if [[ "${START_MOCK}" == "1" ]]; then
  mapfile -t JAVA_MOCK_PORTS < <(seq "${MOCK_BASE_GRPC_PORT}" \
    "$((MOCK_BASE_GRPC_PORT + N_PREFILL + N_DECODE - 1))")
  assert_ports_free "${JAVA_MOCK_PORTS[@]}"
  if [[ ! -f "${JAVA_MOCK_ENGINE_JAR}" ]]; then
    echo "Java mock engine jar not found, auto-building: ${JAVA_MOCK_ENGINE_JAR} (first build may take several minutes)"
    if ! (cd "${FLEXLB_DIR}" && ./mvnw -P"${MAVEN_PROFILES}" -pl flexlb-mock-engine -am package -DskipTests); then
      echo "Failed to build Java mock engine jar via Maven (this may take several minutes on a cold cache)" >&2
      exit 1
    fi
    if [[ ! -f "${JAVA_MOCK_ENGINE_JAR}" ]]; then
      echo "Failed to build Java mock engine jar: ${JAVA_MOCK_ENGINE_JAR}" >&2
      exit 1
    fi
  fi
  java -Xms"${JAVA_MOCK_JVM_XMS}" -Xmx"${JAVA_MOCK_JVM_XMX}" \
    -XX:+ExitOnOutOfMemoryError \
    -Xlog:gc*,safepoint:"${RUN_DIR}/mock_engine_gc.log":time,uptime,level,tags:filecount=3,filesize=20m \
    -jar "${JAVA_MOCK_ENGINE_JAR}" \
    --n-prefill "${N_PREFILL}" \
    --n-decode "${N_DECODE}" \
    --base-grpc-port "${MOCK_BASE_GRPC_PORT}" \
    --event-loop-threads "${JAVA_MOCK_EVENT_LOOP_THREADS}" \
    --completion-threads "${JAVA_MOCK_COMPLETION_THREADS}" \
    --stats-interval-ms "${JAVA_MOCK_STATS_INTERVAL_MS}" \
    --stats-stdout "${MOCK_STATS_STDOUT}" \
    --auto-fetch "${JAVA_MOCK_AUTO_FETCH}" \
    --events-file "${MOCK_EVENTS_FILE}" \
    --decode-max-concurrency "${JAVA_MOCK_DECODE_MAX_CONCURRENCY}" \
    --performance "${PERFORMANCE_FILE}" \
    --master-config "${PROCESS_CONFIG_FILE}" \
    --prefill-kv-pool-blocks "${PREFILL_CACHE_BLOCKS}" \
    --decode-kv-pool-blocks "${DECODE_CACHE_BLOCKS}" \
    --endpoint-file "${ENDPOINT_FILE}" \
    --env-file "${FLEXLB_ENV_FILE}" \
    "${JAVA_MOCK_DISCOVERY_ARGS[@]}" \
    >"${RUN_DIR}/mock_engine.log" 2>&1 &
  MOCK_PID="$!"
  echo "Java mock engine heap: Xms=${JAVA_MOCK_JVM_XMS}, Xmx=${JAVA_MOCK_JVM_XMX}"
  echo "Java mock engine stats interval: ${JAVA_MOCK_STATS_INTERVAL_MS}ms"
  if [[ -n "${MOCK_DISCOVERY_FILE}" ]]; then
    echo "File service discovery: engine maintains ${MOCK_DISCOVERY_FILE} (add_engine/remove_engine keep it in sync)"
  fi
  # The Java process writes discovery files only after every gRPC port is bound.
  wait_for_port "127.0.0.1" "$((MOCK_BASE_GRPC_PORT + N_PREFILL + N_DECODE - 1))" 60
  if ! kill -0 "${MOCK_PID}" >/dev/null 2>&1; then
    echo "Java mock engine exited during startup" >&2
    tail -50 "${RUN_DIR}/mock_engine.log" >&2 || true
    exit 1
  fi
  for _ in $(seq 1 100); do
    if ! kill -0 "${MOCK_PID}" >/dev/null 2>&1; then
      echo "Java mock engine exited before writing discovery files" >&2
      tail -50 "${RUN_DIR}/mock_engine.log" >&2 || true
      exit 1
    fi
    if [[ -s "${ENDPOINT_FILE}" ]]; then
      break
    fi
    sleep 0.1
  done
  if [[ ! -s "${ENDPOINT_FILE}" ]]; then
    echo "Java mock engine did not write endpoint file: ${ENDPOINT_FILE}" >&2
    exit 1
  fi
else
  if [[ ! -f "${ENDPOINT_FILE}" ]]; then
    echo "START_MOCK=0 requires ENDPOINT_FILE at ${ENDPOINT_FILE}" >&2
    exit 1
  fi
fi

mapfile -t FLEXLB_ENV_ARGS < <(python3 - "${ENDPOINT_FILE}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print("MODEL_SERVICE_CONFIG=" + payload["env"]["MODEL_SERVICE_CONFIG"])
PY
)

RUNTIME_OVERRIDE_ENV_ARGS=()
OVERRIDE_ENV_KEYS=(
  HIPPO_ROLE
  FLEXLB_MONITOR_ENABLED
  FLEXLB_MONITOR_METRIC_WHITELIST
)
for key in "${OVERRIDE_ENV_KEYS[@]}"; do
  if declare -p "${key}" >/dev/null 2>&1; then
    RUNTIME_OVERRIDE_ENV_ARGS+=("${key}=${!key}")
  fi
done

JAVA_HEAP_OPTS=()
JVM_HEAP_SIZE=""
if [[ -f "${PROCESS_CONFIG_FILE}" ]]; then
  JVM_HEAP_SIZE="$(python3 - "${PROCESS_CONFIG_FILE}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for item in payload.get("zone_process_setting", {}).get("process_info", {}).get("envs", []):
    if isinstance(item, list) and len(item) == 2 and item[0] == "FLEXLB_JVM_HEAP_SIZE":
        print(item[1])
        break
PY
)"
fi
JVM_XMS="${FLEXLB_JVM_XMS:-${JVM_HEAP_SIZE}}"
JVM_XMX="${FLEXLB_JVM_XMX:-${JVM_HEAP_SIZE}}"
if [[ -n "${JVM_XMS}" ]]; then
  JAVA_HEAP_OPTS+=(-Xms"${JVM_XMS}")
fi
if [[ -n "${JVM_XMX}" ]]; then
  JAVA_HEAP_OPTS+=(-Xmx"${JVM_XMX}")
fi

if [[ "${START_FLEXLB}" == "1" ]]; then
  assert_ports_free "${FLEXLB_HTTP_PORT}" "${FLEXLB_MANAGEMENT_PORT}" "$((FLEXLB_HTTP_PORT + 2))"
  if [[ "$(java_major java)" -lt 21 ]]; then
    echo "Java 21 is required to build/start flexlb-api. Set JAVA21_HOME or JAVA_HOME." >&2
    exit 1
  fi
  if [[ -n "${FLEXLB_START_CMD:-}" ]]; then
    env "${FLEXLB_ENV_ARGS[@]}" "${RUNTIME_OVERRIDE_ENV_ARGS[@]}" \
      "FLEXLB_CONFIG=${FLEXLB_CONFIG}" \
      bash -lc "${FLEXLB_START_CMD}" >"${RUN_DIR}/flexlb.log" 2>&1 &
  else
    if [[ ! -f "${FLEXLB_JAR}" ]]; then
      (cd "${FLEXLB_DIR}" && ./mvnw -P"${MAVEN_PROFILES}" -pl flexlb-api -am package -DskipTests)
    fi
    # Suppress pv.log unless explicitly enabled (see the FLEXLB_PV_LOG note
    # above). FLEXLB_START_CMD mode is caller-owned and left untouched.
    MASTER_LOG_ARGS=()
    if [[ "${FLEXLB_PV_LOG}" != "on" ]]; then
      MASTER_LOG_ARGS+=(--logging.level.pvLogger=WARN)
    fi
    env "${FLEXLB_ENV_ARGS[@]}" "${RUNTIME_OVERRIDE_ENV_ARGS[@]}" \
      "FLEXLB_CONFIG=${FLEXLB_CONFIG}" \
      java -XX:StartFlightRecording=filename=${JFR_FILE},settings=profile,duration=${JFR_DURATION},disk=true,maxsize=256m,dumponexit=true "${JAVA_HEAP_OPTS[@]}" "${JAVA_MODULE_OPTS[@]}" "${JVM_SYSTEM_PROPS[@]}" -jar "${FLEXLB_JAR}" \
      --server.port="${FLEXLB_HTTP_PORT}" \
      --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
      --spring.profiles.active="${SPRING_PROFILE:-default}" \
      --flexlb.log.path="${FLEXLB_LOG_PATH}" \
      --flexlb.monitor.metric-whitelist="${FLEXLB_MONITOR_METRIC_WHITELIST}" \
      ${MASTER_LOG_ARGS[@]+"${MASTER_LOG_ARGS[@]}"} \
      >"${RUN_DIR}/flexlb.log" 2>&1 &
  fi
  FLEXLB_PID="$!"
  echo "FlexLB heap: Xms=${JVM_XMS:-JVM-default}, Xmx=${JVM_XMX:-JVM-default}"
  if ! wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60; then
    if ! kill -0 "${FLEXLB_PID}" >/dev/null 2>&1; then
      flexlb_exit_code=0
      wait "${FLEXLB_PID}" || flexlb_exit_code=$?
      echo "FlexLB exited before opening port ${FLEXLB_HTTP_PORT} (exit_code=${flexlb_exit_code})" >&2
    fi
    exit 1
  fi
  wait_for_endpoints_ready "${FLEXLB_HTTP_PORT}" "${N_PREFILL}" "${N_DECODE}"
  if [[ "${FLEXLB_WARMUP_SECONDS:-10}" -gt 0 ]]; then
    echo "Warming up FlexLB for ${FLEXLB_WARMUP_SECONDS:-10}s before starting load..."
    sleep "${FLEXLB_WARMUP_SECONDS:-10}"
  fi
  assert_mock_engine_healthy
  # Discovery can be healthy once and then degrade during warmup. Revalidate the
  # complete engine set immediately before applying load.
  wait_for_endpoints_ready "${FLEXLB_HTTP_PORT}" "${N_PREFILL}" "${N_DECODE}"
  if [[ "${COLLECTION_PROFILE}" == diagnostic ]]; then save_master_info "${RUN_DIR}/master_info_before.json"; fi
fi

CLIENT_START_EPOCH_MS="$(python3 - "${LOAD_CLIENT_START_DELAY_SECONDS}" <<'PY'
import sys
import time
print(int(time.time() * 1000 + float(sys.argv[1]) * 1000))
PY
)"
echo "Load clients will start at epoch_ms=${CLIENT_START_EPOCH_MS}"
echo "Send mode: ${SEND_MODE:-replay} (SEND_MODE_QPS=${SEND_MODE_QPS:-0})"
echo "warmup(prepare)=${FLEXLB_WARMUP_SECONDS:-10}s before any traffic; ramp-up=${RAMP_UP_SECONDS}s linear QPS climb (uniform mode)"

# Monitor startup must succeed before launching traffic.
start_monitoring

# JavaLoadClient reads its configuration exclusively from environment
# variables (no CLI flags); lib/load_client.sh's run_java_load_client is
# the single source of truth for that mapping — every JavaLoadClient env
# var is exported explicitly there (unpassed ones blanked), so no ambient
# environment can leak in. PRIORITY is deliberately not passed: priority
# comes from the trace records, and records without one fall back to
# JavaLoadClient's built-in default 50 (the neutral QoS level — priority 0
# is rejected by master admission); the lib blanks ambient PRIORITY for us.
# Callers needing an env-level default (or the legacy PRIORITY=0
# leave-unset-on-the-wire behavior) must add "PRIORITY=<n>" to the explicit
# env list in launch_java_load_client below. FORCE_PRIORITY (single-QoS
# pin, overrides trace priority) IS passed explicitly — it defaults to 50, so unattended runs
# replay every request at one uniform priority; multi-priority experiments
# pass 0 to opt out.
# M9: archive the JavaLoadClient env effective values at the client launch
# point. Receives the exact KEY=VALUE argv launch_java_load_client forwards
# (PRIORITY is recorded empty — this script deliberately never passes it);
# writes run_root/client_env.json once (first shard wins: the values are
# identical across shards except OUTPUT_DIR / SHARD_INDEX /
# SKIP_SERVER_LATENCY, and the worker layout itself is captured by
# LOAD_CLIENT_WORKERS). analysis/consolidate.py embeds it into
# run_meta.json as client_env, sibling of flexlb_env. Phase B also records
# CLIENT_PACING_LAG_P99_LIMIT_MS here: not a Java env (the client never
# reads it) but the pacing validity limit analysis/aggregate.py reads
# from run_meta.client_env — the merge heredoc that used to take it as
# argv is gone, so this snapshot is its only path into the aggregate.
write_client_env_snapshot() {
  python3 - "${RUN_DIR}/client_env.json" "PRIORITY=" "$@" <<'PY'
import json
import sys

out_path, items = sys.argv[1], sys.argv[2:]
payload = {}
for item in items:
    key, _, value = item.partition("=")
    payload[key] = value
with open(out_path, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
}

launch_java_load_client() {
  local output_dir="$1"
  local num_shards="$2"
  local shard_index="$3"
  local max_concurrency="$4"
  local skip_server_latency="$5"
  if [[ "${COLLECTION_PROFILE}" != diagnostic ]]; then skip_server_latency=true; fi
  local playback_args=("COLLECTION_PROFILE=${COLLECTION_PROFILE}" "CLIENT_MONITORING=true")
  local playback_key
  for playback_key in MAX_LAPS LAP_IDENTITY LAP_RETAIN_PROBABILITY PLAYBACK_SEED \
      BURST_FACTOR BURST_PERIOD_SECONDS BURST_DUTY DIURNAL_AMPLITUDE DIURNAL_PERIOD_SECONDS; do
    if [[ -n "${!playback_key:-}" ]]; then
      playback_args+=("${playback_key}=${!playback_key}")
    fi
  done
  # M9: one client_env.json per run, written at the first launch.
  if [[ ! -f "${RUN_DIR}/client_env.json" ]]; then
    write_client_env_snapshot "${playback_args[@]}" \
      "TRACE_FILE=${TRACE_FILE}" \
      "TARGET_ADDR=${TARGET_ADDR:-${FLEXLB_HTTP_ADDR}}" \
      "GRPC_TARGET=${GRPC_TARGET:-}" \
      "DURATION_S=${DURATION_S}" \
      "MAX_CONCURRENCY=${max_concurrency}" \
      "REPLAY_SPEED=${REPLAY_SPEED}" \
      "LOAD_CLIENT_WORKERS=${LOAD_CLIENT_WORKERS}" \
      "OUTPUT_DIR=${output_dir}" \
      "NUM_SHARDS=${num_shards}" \
      "SHARD_INDEX=${shard_index}" \
      "LIMIT=${LIMIT}" \
      "TIMEOUT_MS=${TIMEOUT_MS}" \
      "SLA_TTFT_MS=${SLA_TTFT_MS}" \
      "FETCH_OUTPUT_STREAM=${FETCH_OUTPUT_STREAM}" \
      "FORCE_PRIORITY=${FORCE_PRIORITY}" \
      "LOOP=${LOOP}" \
      "SEND_MODE=${SEND_MODE}" \
      "SEND_MODE_QPS=${SEND_MODE_QPS}" \
      "RAMP_UP_SECONDS=${RAMP_UP_SECONDS}" \
      "N_CHANNELS=${N_CHANNELS:-}" \
      "EVENT_LOOP_THREADS=${EVENT_LOOP_THREADS:-}" \
      "START_AT_EPOCH_MS=${CLIENT_START_EPOCH_MS}" \
      "RESPONSE_TIMEOUT=${RESPONSE_TIMEOUT:-}" \
      "SKIP_SERVER_LATENCY=${skip_server_latency}" \
      "MODEL=${MODEL:-}" \
      "API_KEY=${API_KEY:-}" \
      "GRADIENT=${GRADIENT}" \
      "GRADIENT_START_SPEED=${GRADIENT_START_SPEED}" \
      "GRADIENT_MAX_SPEED=${GRADIENT_MAX_SPEED}" \
      "MAX_INPUT_LEN=${MAX_INPUT_LEN}" \
      "MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN}" \
      "PUSHGATEWAY_URL=${PUSHGATEWAY_URL}" \
      "ENABLE_FALLBACK=${ENABLE_FALLBACK}" \
      "ENDPOINTS_FILE=${ENDPOINTS_FILE:-}" \
      "DRY_RUN=${DRY_RUN:-0}" \
      "CLIENT_PACING_LAG_P99_LIMIT_MS=${CLIENT_PACING_LAG_P99_LIMIT_MS}"
  fi
  run_java_load_client "${playback_args[@]}" \
    "TRACE_FILE=${TRACE_FILE}" \
    "TARGET_ADDR=${TARGET_ADDR:-${FLEXLB_HTTP_ADDR}}" \
    "GRPC_TARGET=${GRPC_TARGET:-}" \
    "DURATION_S=${DURATION_S}" \
    "MAX_CONCURRENCY=${max_concurrency}" \
    "REPLAY_SPEED=${REPLAY_SPEED}" \
    "LOAD_CLIENT_WORKERS=${LOAD_CLIENT_WORKERS}" \
    "OUTPUT_DIR=${output_dir}" \
    "NUM_SHARDS=${num_shards}" \
    "SHARD_INDEX=${shard_index}" \
    "LIMIT=${LIMIT}" \
    "TIMEOUT_MS=${TIMEOUT_MS}" \
    "SLA_TTFT_MS=${SLA_TTFT_MS}" \
    "FETCH_OUTPUT_STREAM=${FETCH_OUTPUT_STREAM}" \
    "FORCE_PRIORITY=${FORCE_PRIORITY}" \
    "LOOP=${LOOP}" \
    "SEND_MODE=${SEND_MODE}" \
    "SEND_MODE_QPS=${SEND_MODE_QPS}" \
    "RAMP_UP_SECONDS=${RAMP_UP_SECONDS}" \
    "N_CHANNELS=${N_CHANNELS:-}" \
    "EVENT_LOOP_THREADS=${EVENT_LOOP_THREADS:-}" \
    "START_AT_EPOCH_MS=${CLIENT_START_EPOCH_MS}" \
    "RESPONSE_TIMEOUT=${RESPONSE_TIMEOUT:-}" \
    "SKIP_SERVER_LATENCY=${skip_server_latency}" \
    "MODEL=${MODEL:-}" \
    "API_KEY=${API_KEY:-}" \
    "GRADIENT=${GRADIENT}" \
    "GRADIENT_START_SPEED=${GRADIENT_START_SPEED}" \
    "GRADIENT_MAX_SPEED=${GRADIENT_MAX_SPEED}" \
    "MAX_INPUT_LEN=${MAX_INPUT_LEN}" \
    "MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN}" \
    "PUSHGATEWAY_URL=${PUSHGATEWAY_URL}" \
    "ENABLE_FALLBACK=${ENABLE_FALLBACK}" \
    "ENDPOINTS_FILE=${ENDPOINTS_FILE:-}" \
    "DRY_RUN=${DRY_RUN:-0}"
}

if [[ "${LOAD_CLIENT_WORKERS}" -le 1 ]]; then
  launch_java_load_client "${RUN_DIR}/load_client" 1 0 "${MAX_CONCURRENCY}" 0 \
    | tee "${RUN_DIR}/client.stdout"
else
  mkdir -p "${RUN_DIR}/load_client"
  if [[ "${COLLECTION_PROFILE}" == diagnostic ]]; then curl -fsS -X POST "http://${FLEXLB_HTTP_ADDR}/rtp_llm/server_latency/reset" >/dev/null; fi
  SHARD_MAX_CONCURRENCY=$(( (MAX_CONCURRENCY + LOAD_CLIENT_WORKERS - 1) / LOAD_CLIENT_WORKERS ))
  for ((shard = 0; shard < LOAD_CLIENT_WORKERS; shard++)); do
    shard_dir="${RUN_DIR}/load_client/shard_${shard}"
    launch_java_load_client "${shard_dir}" "${LOAD_CLIENT_WORKERS}" "${shard}" \
      "${SHARD_MAX_CONCURRENCY}" 1 \
      >"${RUN_DIR}/client_shard_${shard}.stdout" 2>&1 &
    CLIENT_PIDS+=("$!")
  done

  CLIENT_EXIT=0
  for pid in "${CLIENT_PIDS[@]}"; do
    wait "${pid}" || CLIENT_EXIT=$?
  done
fi

# Optional terminal snapshot is dedicated diagnostic evidence, never a curve source.
if [[ "${START_FLEXLB}" == "1" && "${COLLECTION_PROFILE}" == diagnostic ]]; then
  if curl -fsS "http://${FLEXLB_HTTP_ADDR}/rtp_llm/server_latency" \
      >"${RUN_DIR}/load_client/server_latency.json.tmp" 2>/dev/null; then
    mv "${RUN_DIR}/load_client/server_latency.json.tmp" \
      "${RUN_DIR}/load_client/server_latency.json"
  else
    rm -f "${RUN_DIR}/load_client/server_latency.json.tmp"
    echo "WARNING: terminal server_latency fetch failed (master-side validity items will be data-missing)" >&2
  fi
fi

if [[ "${CLIENT_EXIT:-0}" -ne 0 ]]; then
  # R12: best-effort consolidation on the load-client failure path so a
  # failed client still leaves a consolidated (analyzable) directory; the
  # CONSOLIDATED sentinel keeps this to one pass per script invocation.
  consolidate_run_outputs_now
  exit "${CLIENT_EXIT}"
fi

stop_monitoring
assert_mock_engine_healthy

if [[ "${SLO_BATCH_DRAIN_SECONDS}" -gt 0 ]]; then
  echo "Waiting ${SLO_BATCH_DRAIN_SECONDS}s for mock task status to drain..."
  sleep "${SLO_BATCH_DRAIN_SECONDS}"
fi

assert_mock_engine_healthy
if [[ "${START_FLEXLB}" == "1" ]]; then
  wait_for_endpoints_ready "${FLEXLB_HTTP_PORT}" "${N_PREFILL}" "${N_DECODE}"
  if [[ "${COLLECTION_PROFILE}" == diagnostic ]]; then save_master_info "${RUN_DIR}/master_info_after.json"; fi
fi

echo "aggregate=${RUN_DIR}/aggregate.json"
echo "monitoring=${RUN_DIR}/telemetry/0"
consolidate_run_outputs_now

# Standard monitor query archive is the only source of performance curves.
PYTHONPATH="${ONLINE_EVAL_DIR}/src:${ONLINE_EVAL_DIR}" python3 -m flexlb_eval.monitoring.session report --run-dir "${RUN_DIR}"

# TEST_VERDICT: read test_valid from the aggregate summary (the merge
# heredoc's old summary.json verdict is gone). A missing aggregate.json or
# an unparsable one is a WARNING only — the client exit code stays the
# script's verdict, we never fabricate one. test_valid=null (None) means
# data-missing: not false, no exit 1. test_valid=false keeps the INVALID
# PERFORMANCE RUN semantics.
AGGREGATE_FILE="${RUN_DIR}/aggregate.json"
if [[ -f "${AGGREGATE_FILE}" ]]; then
  TEST_VALID="$(python3 - "${AGGREGATE_FILE}" <<'PY'
import json
import sys

try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))["summary"]["test_valid"]
except (OSError, ValueError, KeyError, TypeError):
    print("parse_error")
else:
    print("unknown" if value is None else str(bool(value)).lower())
PY
)"
  if [[ "${TEST_VALID}" == "parse_error" ]]; then
    echo "WARNING: could not read summary.test_valid from ${AGGREGATE_FILE}; no verdict (client exit code preserved)" >&2
  elif [[ "${TEST_VALID}" == "false" ]]; then
    echo "INVALID PERFORMANCE RUN: see validity_checks in ${AGGREGATE_FILE}" >&2
    exit 1
  fi
else
  echo "WARNING: ${AGGREGATE_FILE} missing; no test_valid verdict (client exit code preserved)" >&2
fi

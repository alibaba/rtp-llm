#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${FLEXLB_DIR}/../.." && pwd)"

# Shared JavaLoadClient helpers: env-var mapping (run_java_load_client) and
# JDK 21 detection (java_major/detect_java21_home/require_java21).
# JAVA_LOAD_CLIENT_HEAP_SIZE (historical knob) feeds the lib's Xmx default;
# export either JAVA_LOAD_CLIENT_HEAP_SIZE or JAVA_LOAD_CLIENT_JVM_XMX/XMS
# before this script runs to override the load client JVM sizing.
JAVA_LOAD_CLIENT_JVM_XMX="${JAVA_LOAD_CLIENT_JVM_XMX:-${JAVA_LOAD_CLIENT_HEAP_SIZE:-16g}}"
source "${SCRIPT_DIR}/lib_load_client.sh"

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

TRACE_FILE="${TRACE_FILE:-${SCRIPT_DIR}/data/online_logs/trace_30min.jsonl}"
PERFORMANCE_FILE="${PERFORMANCE_FILE:-${SCRIPT_DIR}/data/performance/dsv4_flash_performance.sample.json}"
PROCESS_CONFIG_FILE="${PROCESS_CONFIG_FILE:-${SCRIPT_DIR}/data/config/master_fixed_window.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
FLEXLB_LOG_PATH="${FLEXLB_LOG_PATH:-${RUN_DIR}/flexlb_logs}"

N_PREFILL="${N_PREFILL:-2}"
N_DECODE="${N_DECODE:-4}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-61000}"
# NOTE: the three assignments below (JAVA_MOCK_ENGINE_JAR, JAVA_LOAD_CLIENT_JAR,
# MAVEN_PROFILES) duplicate defaults already applied by lib_load_client.sh at
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
# Default matches the mock engine's DEFAULT_DECODE_MAX_CONCURRENCY (132);
# lower it to trip the opt-in hard admission gate (decode.max_pending_requests)
# so the engine queues requests into the KV_ALLOCATED/accepted layer.
JAVA_MOCK_DECODE_MAX_CONCURRENCY="${JAVA_MOCK_DECODE_MAX_CONCURRENCY:-132}"
JAVA_MOCK_ENGINE_HEAP_SIZE="${JAVA_MOCK_ENGINE_HEAP_SIZE:-32g}"
JAVA_MOCK_JVM_XMS="${JAVA_MOCK_JVM_XMS:-${JAVA_MOCK_ENGINE_HEAP_SIZE}}"
JAVA_MOCK_JVM_XMX="${JAVA_MOCK_JVM_XMX:-${JAVA_MOCK_ENGINE_HEAP_SIZE}}"
ENDPOINT_READY_TIMEOUT_S="${ENDPOINT_READY_TIMEOUT_S:-120}"
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
DURATION_S="${DURATION_S:-0}"
REPLAY_SPEED="${REPLAY_SPEED:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-999999999}"
TIMEOUT_MS="${TIMEOUT_MS:-3600000}"
SLA_TTFT_MS="${SLA_TTFT_MS:-500}"
ZERO_OUTPUT_POLICY="${ZERO_OUTPUT_POLICY:-skip}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-0}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-0}"
GRADIENT="${GRADIENT:-0}"
GRADIENT_MAX_SPEED="${GRADIENT_MAX_SPEED:-1000}"
GRADIENT_START_SPEED="${GRADIENT_START_SPEED:-10}"
# Client-side output-stream reading: 1 (default) = the load client reads engine
# output streams after Schedule; 0 = skip client stream reads while the engine
# still executes prefill+decode in full (BATCH dispatcher only).
FETCH_OUTPUT_STREAM="${FETCH_OUTPUT_STREAM:-1}"
LOOP="${LOOP:-0}"
# Send mode is a pure pass-through (single env-var layer): empty SEND_MODE
# means JavaLoadClient's built-in default (replay), identical to before.
SEND_MODE="${SEND_MODE:-}"
SEND_MODE_QPS="${SEND_MODE_QPS:-}"
PUSHGATEWAY_URL="${PUSHGATEWAY_URL:-}"
LOAD_CLIENT_WORKERS="${LOAD_CLIENT_WORKERS:-8}"
LOAD_CLIENT_START_DELAY_SECONDS="${LOAD_CLIENT_START_DELAY_SECONDS:-10}"
CLIENT_PACING_LAG_P99_LIMIT_MS="${CLIENT_PACING_LAG_P99_LIMIT_MS:-100}"
SLO_BATCH_ANALYSIS="${SLO_BATCH_ANALYSIS:-1}"
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
# consolidate_run_outputs.py).
FLEXLB_PV_LOG="${FLEXLB_PV_LOG:-off}"
JFR_FILE="${JFR_FILE:-${RUN_DIR}/flexlb_profile.jfr}"
JFR_DURATION="${JFR_DURATION:-300s}"
FLEXLB_MONITOR_ENABLED="${FLEXLB_MONITOR_ENABLED:-true}"
# Default "all": the unified analysis needs the full flexlb business metric
# surface (KV usage, inflight, batcher, cache-hit, dispatch reasons...) that
# critical-only filters down to ~6 flexlb_* series. Set
# FLEXLB_MONITOR_MODE=critical-only explicitly to restore the trimmed metric
# set (the master's prometheus endpoint then emits ~50-100 fewer lines/s).
FLEXLB_MONITOR_MODE="${FLEXLB_MONITOR_MODE:-all}"
HIPPO_ROLE="${HIPPO_ROLE:-test}"

DEFAULT_FLEXLB_CONFIG='{
  "schemaVersion": 2,
  "scheduler": {
    "type": "QUEUE",
    "ordering": {"type": "PRIORITY", "defaultPriority": 50},
    "decision": {
      "type": "FIXED_WINDOW",
      "maxRequests": 32,
      "maxCollectionWaitMs": 10,
      "maxPredictedExecutionMs": 550
    },
    "capacity": {
      "maxOutstandingRequestsGlobal": 1000000,
      "maxWaitingRequestsPerPrefillWorker": 1024
    }
  },
  "dispatcher": {
    "type": "BATCH",
    "enqueueRpcTimeoutMs": 5000
  },
  "router": {
    "availabilityHysteresisPercent": 30,
    "roles": {
      "prefill": {
        "availability": {"maxPendingRequests": 100000},
        "executionTimeEstimator": {"type": "FORMULA"},
        "selector": {
          "type": "ESTIMATED_TTFT",
          "candidateChoice": {"type": "RANDOM_WITHIN_TOLERANCE"}
        }
      },
      "decode": {
        "availability": {"maxKvUsagePercent": 90, "maxEngineRequests": 132},
        "kvReservation": {"maxOutputTokensForEstimate": 1000},
        "selector": {"type": "KV_USAGE_WEIGHTED_RANDOM"}
      },
      "vit": {"selector": {"type": "RANDOM"}}
    }
  },
  "observability": {
    "cacheHit": {
      "recentKeyWindow": {
        "writeEnabled": true,
        "durationMs": 1800000,
        "maxKeyOccurrences": 10000000
      },
      "metricsEnabled": true,
      "requestTraceLogEnabled": false
    }
  }
}'
OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN:-.*}"
OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-none}"
HIPPO_ROLE="${HIPPO_ROLE:-flexlb_eval_master}"

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
MASTER_COUNTER_POLLER_PID=""
# Secondary 1s collectors (see start_secondary_pollers below): mock per-engine
# prometheus, master prometheus/inflight time series, process CPU/RSS sampling.
MOCK_PER_ENGINE_POLLER_PID=""
MASTER_PROMETHEUS_POLLER_PID=""
MASTER_INFLIGHT_POLLER_PID=""
PROCESS_USAGE_POLLER_PID=""
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

# java_major / detect_java21_home are provided by lib_load_client.sh (sourced
# above); do not redefine them here.

JAVA21_HOME_DETECTED="$(detect_java21_home || true)"
if [[ -n "${JAVA21_HOME_DETECTED}" ]]; then
  export JAVA_HOME="${JAVA21_HOME_DETECTED}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

cleanup() {
  stop_master_counter_poller
  stop_secondary_pollers
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
}
trap cleanup EXIT

# Best-effort run output consolidation (runs at most once — CONSOLIDATED
# sentinel). Called AFTER the client exit code is known and the summary=
# line is printed: consolidation (notably the per_request gzip pass) can
# take tens of seconds on large runs while the flexlb-online-eval skill's
# timeout window is only DURATION+180s, so the exit code and the summary
# line must be decided first — a slow machine then reports the correct
# result instead of a spurious TIMEOUT with a half-written directory.
# Two call sites: the load-client failure path (right before exit) and the
# happy path (after the artifact echo, before the test_valid verdict).
# Startup failures that never reach the load client do not consolidate.
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
  # Stop the secondary pollers first: consolidation merges their one-shot
  # output files and deletes them, so no writer may still be appending (a
  # writer holding an unlinked fd would keep writing into the void). The
  # stop is idempotent — also called at the load-client stop point / cleanup.
  stop_secondary_pollers
  local consolidate_mock_port_args=()
  if [[ "${START_MOCK}" == "1" ]]; then
    consolidate_mock_port_args+=(--mock-http-port "$((MOCK_BASE_GRPC_PORT - 1))")
  fi
  # Consolidate the run directory into the per-component JSON+log layout
  # (run_meta/mock/master/client .json + .log, merged per_request.jsonl[.gz];
  # see consolidate_run_outputs.py's docstring for the full keep/delete list).
  # Runs while the mock cluster and master are still alive (cleanup kills them
  # on EXIT), so the final cluster snapshot is captured from the control plane.
  # Kept in place on purpose: endpoints.json, flexlb_env.txt, flexlb_profile.jfr,
  # load_client/summary.json (the flexlb-online-eval skill reads that exact
  # path), load_client/server_latency.json (skill fetch_server_latency) and
  # load_client/report.md.
  local trace_file_sha256 trace_file_lines
  trace_file_sha256="$(compute_file_digest "${TRACE_FILE}")"
  trace_file_lines="$(count_file_lines "${TRACE_FILE}")"
  python3 "${SCRIPT_DIR}/consolidate_run_outputs.py" \
    --run-dir "${RUN_DIR}" \
    ${consolidate_mock_port_args[@]+"${consolidate_mock_port_args[@]}"} \
    --param "n_prefill=${N_PREFILL}" \
    --param "n_decode=${N_DECODE}" \
    --param "mock_base_grpc_port=${MOCK_BASE_GRPC_PORT}" \
    --param "flexlb_http_port=${FLEXLB_HTTP_PORT}" \
    --param "flexlb_management_port=${FLEXLB_MANAGEMENT_PORT}" \
    --param "start_mock=${START_MOCK}" \
    --param "start_flexlb=${START_FLEXLB}" \
    --param "replay_speed=${REPLAY_SPEED}" \
    --param "send_mode=${SEND_MODE:-replay}" \
    --param "send_mode_qps=${SEND_MODE_QPS:-}" \
    --param "limit=${LIMIT}" \
    --param "duration_s=${DURATION_S}" \
    --param "max_concurrency=${MAX_CONCURRENCY}" \
    --param "load_client_workers=${LOAD_CLIENT_WORKERS}" \
    --param "sla_ttft_ms=${SLA_TTFT_MS}" \
    --param "zero_output_policy=${ZERO_OUTPUT_POLICY}" \
    --param "fetch_output_stream=${FETCH_OUTPUT_STREAM}" \
    --param "loop=${LOOP}" \
    --param "gradient=${GRADIENT}" \
    --param "trace_file=${TRACE_FILE}" \
    --param "trace_file_sha256=${trace_file_sha256}" \
    --param "trace_file_lines=${trace_file_lines}" \
    --param "performance_file=${PERFORMANCE_FILE}" \
    --param "process_config_file=${PROCESS_CONFIG_FILE}" \
    --param "java_mock_jvm_xms=${JAVA_MOCK_JVM_XMS}" \
    --param "java_mock_jvm_xmx=${JAVA_MOCK_JVM_XMX}" \
    --param "java_mock_event_loop_threads=${JAVA_MOCK_EVENT_LOOP_THREADS}" \
    --param "java_mock_completion_threads=${JAVA_MOCK_COMPLETION_THREADS}" \
    --param "prefill_cache_blocks=${PREFILL_CACHE_BLOCKS}" \
    --param "decode_cache_blocks=${DECODE_CACHE_BLOCKS}" \
    --param "timeout_ms=${TIMEOUT_MS}" \
    --param "response_timeout=${RESPONSE_TIMEOUT:-}" \
    --param "flexlb_monitor_mode=${FLEXLB_MONITOR_MODE}" \
    --param "java_mock_stats_interval_ms=${JAVA_MOCK_STATS_INTERVAL_MS}" \
    --param "java_mock_decode_max_concurrency=${JAVA_MOCK_DECODE_MAX_CONCURRENCY}" \
    --param "flexlb_pv_log=${FLEXLB_PV_LOG}" \
    --param "flexlb_config=${FLEXLB_CONFIG}" \
    || echo "WARNING: run output consolidation failed (original files kept as-is)" >&2
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

save_master_prometheus() {
  local output=$1
  local path
  for path in prometheus actuator/prometheus; do
    if curl -fsS "http://127.0.0.1:${FLEXLB_MANAGEMENT_PORT}/${path}" >"${output}"; then
      return 0
    fi
  done
  rm -f "${output}"
  echo "WARNING: unable to save Master Prometheus snapshot" >&2
  return 1
}

# Per-second master arrival/completion counter time series. The management
# Prometheus endpoint has no arrival/completion counters, but the master
# already exposes cumulative arrival_count/completion_count on the existing
# GET /rtp_llm/server_latency endpoint — poll that (no master code change).
# Counters are cumulative within the recorder window; the multi-worker path
# resets the window right after the poller starts, visible as a counter drop.
MASTER_COUNTERS_FILE="${RUN_DIR}/master_counters_timeseries.txt"
MASTER_COUNTER_POLL_INTERVAL_S="${MASTER_COUNTER_POLL_INTERVAL_S:-1}"

start_master_counter_poller() {
  if [[ "${START_FLEXLB}" != "1" ]]; then
    return 0
  fi
  python3 - "${FLEXLB_HTTP_ADDR}" "${MASTER_COUNTERS_FILE}" \
    "${MASTER_COUNTER_POLL_INTERVAL_S}" <<'PY' &
import json
import sys
import time
import urllib.request

addr, out_path, interval_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
url = f"http://{addr}/rtp_llm/server_latency"
with open(out_path, "a", encoding="utf-8") as out:
    while True:
        started = time.time()
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                data = json.load(response)
            out.write(
                f"ts_epoch_ms={int(started * 1000)} "
                f"arrival_count={data.get('arrival_count', 0)} "
                f"completion_count={data.get('completion_count', 0)}\n")
            out.flush()
        except Exception:
            pass  # master briefly unavailable; skip this sample
        time.sleep(max(0.0, interval_s - (time.time() - started)))
PY
  MASTER_COUNTER_POLLER_PID="$!"
}

stop_master_counter_poller() {
  if [[ -n "${MASTER_COUNTER_POLLER_PID}" ]]; then
    kill "${MASTER_COUNTER_POLLER_PID}" >/dev/null 2>&1 || true
    MASTER_COUNTER_POLLER_PID=""
  fi
}

# ---- Secondary 1s collectors (unified-analysis data audit G1/G3/G4/G5) ----
# All four pollers follow the master counter poller pattern: a python3
# background heredoc that appends to a one-shot file under RUN_DIR, a *POLLER_PID
# bookkeeping variable, best-effort semantics (a failed sample or a missing
# dependency — e.g. no `ps` binary — is a WARNING, never a load-test blocker),
# and a stop path wired into the load-client stop point, consolidate_run_outputs_now
# and the EXIT trap. None of them needs curl: urllib covers both HTTP planes.
# Consolidation later merges each file into its component JSON and deletes it
# (same one-shot-source treatment as master_counters_timeseries.txt).

MOCK_PER_ENGINE_METRICS_FILE="${RUN_DIR}/mock_metrics_per_engine.prom"
MASTER_PROMETHEUS_TS_FILE="${RUN_DIR}/master_prometheus_timeseries.prom"
MASTER_INFLIGHT_TS_FILE="${RUN_DIR}/master_inflight_timeseries.jsonl"
PROCESS_USAGE_TS_FILE="${RUN_DIR}/process_usage_timeseries.txt"
# "<pid> <label>" per line; re-read by the process poller every round so
# CLIENT_PIDS can be appended after the workers fork. Removed on stop.
PROCESS_POLL_PID_FILE="${RUN_DIR}/process_poll_pids.txt"
SECONDARY_POLL_INTERVAL_S="${SECONDARY_POLL_INTERVAL_S:-1}"
# M7: A/B switch for the four secondary pollers. Default 1 (full per-second
# collection for the unified analyzer); set FLEXLB_SECONDARY_POLLERS_ENABLED=0
# to skip all of them entirely (zero observation overhead — the
# stability/burst baselines pin this to keep historical numbers comparable).
FLEXLB_SECONDARY_POLLERS_ENABLED="${FLEXLB_SECONDARY_POLLERS_ENABLED:-1}"
# M7: the G1 per-engine poller is the volume driver (~2.2KB x N_engines per
# sample even after the C whitelist below); a larger interval trades timeline
# granularity for disk (e.g. 1250 engines x 120s: 1s -> ~260MB text, 5s ->
# ~52MB) without touching the other 1s pollers.
MOCK_PER_ENGINE_POLL_INTERVAL_S="${MOCK_PER_ENGINE_POLL_INTERVAL_S:-1}"

# G1: per-second mock per-engine Prometheus time series. The mock control
# plane (MOCK_BASE_GRPC_PORT-1) already serves /metrics?per_engine=true
# (~22 series per engine, engine names prefill-N/decode-N); the poller keeps
# only the six series the analyzer consumes (C whitelist: running / waiting /
# active_kv_tokens / available_kv_tokens / accepted_total / completed_total),
# cutting the on-disk footprint to ~1/4 (~2.2KB x N_engines per sample;
# 1250 engines x 120s x 1s interval ≈ 260MB raw text -> ~65MB gzipped in the
# A-split file) — the server-side scrape cost is unchanged. Each sample is
# appended after a "# ts=<epoch_ms>" separator comment so
# consolidate_run_outputs.py can regroup the flat file into a [{ts, metrics}]
# timeline.
start_mock_per_engine_poller() {
  if [[ "${START_MOCK}" != "1" ]]; then
    return 0
  fi
  python3 - "$((MOCK_BASE_GRPC_PORT - 1))" "${MOCK_PER_ENGINE_METRICS_FILE}" \
    "${MOCK_PER_ENGINE_POLL_INTERVAL_S}" <<'PY' &
import sys
import time
import urllib.request

port, out_path, interval_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
keep = {
    "mock_engine_running", "mock_engine_waiting",
    "mock_engine_active_kv_tokens", "mock_engine_available_kv_tokens",
    "mock_engine_accepted_total", "mock_engine_completed_total",
}
url = f"http://127.0.0.1:{port}/metrics?per_engine=true"
with open(out_path, "a", encoding="utf-8") as out:
    while True:
        started = time.time()
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                body = response.read().decode("utf-8", "replace")
            # C: keep only the six analyzer-consumed series — the raw
            # endpoint still emits the full ~22-per-engine surface (server
            # cost unchanged), but the appended bytes drop to ~1/4.
            kept = [
                line for line in body.splitlines()
                if not line.startswith("#")
                and line.split("{", 1)[0].split(" ", 1)[0] in keep
            ]
            if kept:
                out.write(f"# ts={int(started * 1000)}\n")
                out.write("\n".join(kept) + "\n")
                out.flush()
        except Exception:
            pass  # control plane briefly unavailable; skip this sample
        time.sleep(max(0.0, interval_s - (time.time() - started)))
PY
  MOCK_PER_ENGINE_POLLER_PID="$!"
}

# G3: per-second master business-metric time series. /actuator/prometheus on
# the management port is whitelisted down to exactly the series the unified
# analyzer consumes (C: flexlb_app_cache_* KV / hit-ratio family, the batcher
# and routing queue gauges, inflight max age, dispatch reason counters, plus
# the JVM/system health quartet) before appending — a strict subset of the
# old flexlb_app_* prefix filter, so previously collected (fatter) runs stay
# analyzable. FLEXLB_MONITOR_MODE=all is still required upstream:
# critical-only trims the master's own exposition to ~6 flexlb_* series and
# the whitelist below would match almost nothing. Same "# ts=" grouped layout
# as G1.
start_master_prometheus_poller() {
  if [[ "${START_FLEXLB}" != "1" ]]; then
    return 0
  fi
  python3 - "${FLEXLB_MANAGEMENT_PORT}" "${MASTER_PROMETHEUS_TS_FILE}" \
    "${SECONDARY_POLL_INTERVAL_S}" <<'PY' &
import sys
import time
import urllib.request

port, out_path, interval_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
urls = [f"http://127.0.0.1:{port}/{path}" for path in ("actuator/prometheus", "prometheus")]
# C whitelist — every entry is a consumer-backed series (B3 queue curves,
# M3/S7 inflight age, S7 hit-ratio curves, dispatch reasons, cpu/mem):
prefixes = (
    "flexlb_app_cache_",
    "flexlb_app_flexlb_batcher_queue_size",
    "flexlb_app_routing_queue_length",
    "flexlb_app_flexlb_inflight_max_age_ms",
    "flexlb_app_engine_balancing_master_dispatch_reason_total",
    "jvm_memory_used",
    "jvm_gc_pause",
    "process_cpu",
    "system_cpu",
)
with open(out_path, "a", encoding="utf-8") as out:
    while True:
        started = time.time()
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    body = response.read().decode("utf-8", "replace")
            except Exception:
                continue  # try the next path / skip this sample
            kept = [line for line in body.splitlines() if line.startswith(prefixes)]
            if kept:
                out.write(f"# ts={int(started * 1000)}\n")
                out.write("\n".join(kept) + "\n")
                out.flush()
            break
        time.sleep(max(0.0, interval_s - (time.time() - started)))
PY
  MASTER_PROMETHEUS_POLLER_PID="$!"
}

# G4: per-second inflight snapshot. GET /rtp_llm/inflight_status on the
# master HTTP port returns a JSON object; each sample is appended as one
# JSONL line {"ts_epoch_ms": ..., "inflight": {...}} so consolidation can
# json.loads each line independently (tolerating a torn trailing line).
start_master_inflight_poller() {
  if [[ "${START_FLEXLB}" != "1" ]]; then
    return 0
  fi
  python3 - "${FLEXLB_HTTP_ADDR}" "${MASTER_INFLIGHT_TS_FILE}" \
    "${SECONDARY_POLL_INTERVAL_S}" <<'PY' &
import json
import sys
import time
import urllib.request

addr, out_path, interval_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
url = f"http://{addr}/rtp_llm/inflight_status"
with open(out_path, "a", encoding="utf-8") as out:
    while True:
        started = time.time()
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.load(response)
            out.write(json.dumps({"ts_epoch_ms": int(started * 1000), "inflight": payload}) + "\n")
            out.flush()
        except Exception:
            pass  # master briefly unavailable; skip this sample
        time.sleep(max(0.0, interval_s - (time.time() - started)))
PY
  MASTER_INFLIGHT_POLLER_PID="$!"
}

# G5: per-second CPU/RSS sampling of the three JVM groups (mock cluster,
# flexlb master, load client workers). The pid list lives in
# PROCESS_POLL_PID_FILE because CLIENT_PIDS is filled in only after the
# workers fork; the poller re-reads the list every round. Exited pids are
# tolerated (ps just omits them; a wholly dead pidlist makes ps exit non-zero
# and the round is skipped).
start_process_usage_poller() {
  if ! command -v ps >/dev/null 2>&1; then
    echo "WARNING: ps not found; process CPU/RSS sampling disabled" >&2
    return 0
  fi
  python3 - "${PROCESS_POLL_PID_FILE}" "${PROCESS_USAGE_TS_FILE}" \
    "${SECONDARY_POLL_INTERVAL_S}" <<'PY' &
import subprocess
import sys
import time

pid_file, out_path, interval_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(out_path, "a", encoding="utf-8") as out:
    while True:
        started = time.time()
        try:
            entries = []
            with open(pid_file, "r", encoding="utf-8") as pids:
                for line in pids:
                    parts = line.split()
                    if len(parts) == 2 and parts[0].isdigit():
                        entries.append((parts[0], parts[1]))
            if entries:
                result = subprocess.run(
                    ["ps", "-o", "pid,%cpu,rss,etime", "-p",
                     ",".join(pid for pid, _ in entries)],
                    capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    rows = {}
                    for line in result.stdout.splitlines()[1:]:
                        cols = line.split(None, 3)
                        if len(cols) == 4:
                            rows[cols[0]] = cols
                    for pid, label in entries:
                        cols = rows.get(pid)
                        if cols:
                            out.write(
                                f"ts_epoch_ms={int(started * 1000)} label={label} "
                                f"pid={pid} cpu_pct={cols[1]} rss_kb={cols[2]} "
                                f"etime={cols[3]}\n")
                    out.flush()
        except Exception:
            pass  # best-effort sampling; skip this round
        time.sleep(max(0.0, interval_s - (time.time() - started)))
PY
  PROCESS_USAGE_POLLER_PID="$!"
}

# Seed the poller pid list with the processes started so far; load client
# workers are appended by the multi-worker launch loop below.
write_process_poll_pids() {
  : >"${PROCESS_POLL_PID_FILE}"
  if [[ -n "${MOCK_PID}" ]]; then
    echo "${MOCK_PID} mock" >>"${PROCESS_POLL_PID_FILE}"
  fi
  if [[ -n "${FLEXLB_PID}" ]]; then
    echo "${FLEXLB_PID} master" >>"${PROCESS_POLL_PID_FILE}"
  fi
}

append_process_poll_pid() {
  echo "$1 $2" >>"${PROCESS_POLL_PID_FILE}"
}

start_secondary_pollers() {
  # M7: FLEXLB_SECONDARY_POLLERS_ENABLED=0 skips all four pollers — zero
  # observation overhead for A/B comparisons and the pinned historical
  # baselines (run_stability_test.sh / run_burst_test.sh export 0).
  if [[ "${FLEXLB_SECONDARY_POLLERS_ENABLED}" != "1" ]]; then
    echo "Secondary pollers disabled (FLEXLB_SECONDARY_POLLERS_ENABLED=${FLEXLB_SECONDARY_POLLERS_ENABLED})"
    return 0
  fi
  write_process_poll_pids
  start_mock_per_engine_poller
  start_master_prometheus_poller
  start_master_inflight_poller
  start_process_usage_poller
}

stop_secondary_pollers() {
  local pid
  for pid in "${MOCK_PER_ENGINE_POLLER_PID}" "${MASTER_PROMETHEUS_POLLER_PID}" \
    "${MASTER_INFLIGHT_POLLER_PID}" "${PROCESS_USAGE_POLLER_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  MOCK_PER_ENGINE_POLLER_PID=""
  MASTER_PROMETHEUS_POLLER_PID=""
  MASTER_INFLIGHT_POLLER_PID=""
  PROCESS_USAGE_POLLER_PID=""
  rm -f "${PROCESS_POLL_PID_FILE}"
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
mkdir -p "${FLEXLB_LOG_PATH}"
echo "run_dir=${RUN_DIR}"
echo "load client: JavaLoadClient (trace priority passthrough via lib_load_client.sh)"
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
MASTER_DISCOVERY_ENV=()
FLEXLB_DISCOVERY_FILE_PATH=""
if [[ -n "${FLEXLB_DISCOVERY_FILE}" ]]; then
  if [[ "${FLEXLB_DISCOVERY_FILE}" == "auto" || "${FLEXLB_DISCOVERY_FILE}" == "1" ]]; then
    if [[ "${START_MOCK}" != "1" ]]; then
      echo "FLEXLB_DISCOVERY_FILE=auto requires START_MOCK=1 (mock engine writes the file); set an explicit path instead" >&2
      exit 1
    fi
    FLEXLB_DISCOVERY_FILE_PATH="${RUN_DIR}/discovery.json"
  else
    FLEXLB_DISCOVERY_FILE_PATH="${FLEXLB_DISCOVERY_FILE}"
  fi
  JAVA_MOCK_DISCOVERY_ARGS=(--discovery-file "${FLEXLB_DISCOVERY_FILE_PATH}")
  MASTER_DISCOVERY_ENV=("FLEXLB_DISCOVERY_FILE=${FLEXLB_DISCOVERY_FILE_PATH}")
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
    --decode-max-concurrency "${JAVA_MOCK_DECODE_MAX_CONCURRENCY}" \
    --performance "${PERFORMANCE_FILE}" \
    --master-config "${PROCESS_CONFIG_FILE}" \
    --prefill-cache-blocks "${PREFILL_CACHE_BLOCKS}" \
    --decode-cache-blocks "${DECODE_CACHE_BLOCKS}" \
    --endpoint-file "${ENDPOINT_FILE}" \
    --env-file "${FLEXLB_ENV_FILE}" \
    "${JAVA_MOCK_DISCOVERY_ARGS[@]}" \
    >"${RUN_DIR}/mock_engine.log" 2>&1 &
  MOCK_PID="$!"
  echo "Java mock engine heap: Xms=${JAVA_MOCK_JVM_XMS}, Xmx=${JAVA_MOCK_JVM_XMX}"
  echo "Java mock engine stats interval: ${JAVA_MOCK_STATS_INTERVAL_MS}ms"
  if [[ -n "${FLEXLB_DISCOVERY_FILE_PATH}" ]]; then
    echo "File service discovery: engine maintains ${FLEXLB_DISCOVERY_FILE_PATH} (add_engine/remove_engine keep it in sync)"
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
for key, value in payload["env"].items():
    print(f"{key}={value}")
PY
)

PROCESS_ENV_ARGS=()
if [[ -f "${PROCESS_CONFIG_FILE}" ]]; then
  mapfile -d '' -t PROCESS_ENV_ARGS < <(python3 - "${PROCESS_CONFIG_FILE}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
envs = payload.get("zone_process_setting", {}).get("process_info", {}).get("envs", [])
for item in envs:
    if not isinstance(item, list) or len(item) != 2:
        continue
    sys.stdout.write(f"{str(item[0])}={str(item[1])}\0")
PY
)
fi

PROCESS_FLEXLB_CONFIG=""
if [[ -f "${PROCESS_CONFIG_FILE}" ]]; then
  PROCESS_FLEXLB_CONFIG="$(python3 - "${PROCESS_CONFIG_FILE}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
envs = payload.get("zone_process_setting", {}).get("process_info", {}).get("envs", [])
for item in envs:
    if isinstance(item, list) and len(item) == 2 and item[0] == "FLEXLB_CONFIG":
        print(item[1], end="")
        break
PY
)"
fi
FLEXLB_CONFIG="${FLEXLB_CONFIG:-${PROCESS_FLEXLB_CONFIG:-${DEFAULT_FLEXLB_CONFIG}}}"

RUNTIME_OVERRIDE_ENV_ARGS=()
OVERRIDE_ENV_KEYS=(
  FLEXLB_GRPC_EXECUTOR_CORE_SIZE
  FLEXLB_GRPC_EXECUTOR_MAX_SIZE
  FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE
  FLEXLB_JVM_HEAP_SIZE
  FLEXLB_MONITOR_ENABLED
  FLEXLB_MONITOR_MODE
  GRADIENT
  GRADIENT_MAX_SPEED
  GRADIENT_START_SPEED
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
    env "${FLEXLB_ENV_ARGS[@]}" "${PROCESS_ENV_ARGS[@]}" "${RUNTIME_OVERRIDE_ENV_ARGS[@]}" \
      "${MASTER_DISCOVERY_ENV[@]}" \
      "FLEXLB_CONFIG=${FLEXLB_CONFIG}" \
      "OTEL_TRACE_SKIP_PATTERN=${OTEL_TRACE_SKIP_PATTERN}" \
      "OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT}" \
      "HIPPO_ROLE=${HIPPO_ROLE}" \
      "FLEXLB_LOG_PATH=${FLEXLB_LOG_PATH}" \
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
    env "${FLEXLB_ENV_ARGS[@]}" "${PROCESS_ENV_ARGS[@]}" "${RUNTIME_OVERRIDE_ENV_ARGS[@]}" \
      "${MASTER_DISCOVERY_ENV[@]}" \
      "FLEXLB_CONFIG=${FLEXLB_CONFIG}" \
      "OTEL_TRACE_SKIP_PATTERN=${OTEL_TRACE_SKIP_PATTERN}" \
      "OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT}" \
      "HIPPO_ROLE=${HIPPO_ROLE}" \
      "FLEXLB_LOG_PATH=${FLEXLB_LOG_PATH}" \
      java -XX:StartFlightRecording=filename=${JFR_FILE},settings=profile,duration=${JFR_DURATION},disk=true,maxsize=256m,dumponexit=true "${JAVA_HEAP_OPTS[@]}" "${JAVA_MODULE_OPTS[@]}" "${JVM_SYSTEM_PROPS[@]}" -jar "${FLEXLB_JAR}" \
      --server.port="${FLEXLB_HTTP_PORT}" \
      --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
      --spring.profiles.active="${SPRING_PROFILE:-default}" \
      --flexlb.log.path="${FLEXLB_LOG_PATH}" \
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
    echo "Warming up FlexLB for ${FLEXLB_WARMUP_SECONDS}s before starting load..."
    sleep "${FLEXLB_WARMUP_SECONDS}"
  fi
  assert_mock_engine_healthy
  # Discovery can be healthy once and then degrade during warmup. Revalidate the
  # complete engine set immediately before applying load.
  wait_for_endpoints_ready "${FLEXLB_HTTP_PORT}" "${N_PREFILL}" "${N_DECODE}"
  save_master_info "${RUN_DIR}/master_info_before.json"
fi

CLIENT_START_EPOCH_MS="$(python3 - "${LOAD_CLIENT_START_DELAY_SECONDS}" <<'PY'
import sys
import time
print(int(time.time() * 1000 + float(sys.argv[1]) * 1000))
PY
)"
echo "Load clients will start at epoch_ms=${CLIENT_START_EPOCH_MS}"
echo "Send mode: ${SEND_MODE:-replay} (SEND_MODE_QPS=${SEND_MODE_QPS:-0})"

# Capture the master arrival/completion counter time series for the whole load
# window (stopped right after all clients finish; also killed by cleanup).
start_master_counter_poller

# Secondary 1s collectors (mock per-engine prometheus, master prometheus /
# inflight time series, process CPU/RSS) run over the same load window and
# stop at the same point as the counter poller above; consolidation merges
# their files afterwards.
start_secondary_pollers

# JavaLoadClient reads its configuration exclusively from environment
# variables (no CLI flags); lib_load_client.sh's run_java_load_client is
# the single source of truth for that mapping — every JavaLoadClient env
# var is exported explicitly there (unpassed ones blanked), so no ambient
# environment can leak in. PRIORITY is deliberately not passed: priority
# comes from the trace records only, and records without one stay unset on
# the wire (no default 50); the lib blanks ambient PRIORITY for us.
# M9: archive the JavaLoadClient env effective values at the client launch
# point. Receives the exact KEY=VALUE argv launch_java_load_client forwards
# (PRIORITY is recorded empty — this script deliberately never passes it);
# writes run_root/client_env.json once (first shard wins: the values are
# identical across shards except OUTPUT_DIR / SHARD_INDEX /
# SKIP_SERVER_LATENCY, and the worker layout itself is captured by
# LOAD_CLIENT_WORKERS). consolidate_run_outputs.py embeds it into
# run_meta.json as client_env, sibling of flexlb_env.
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
  # M9: one client_env.json per run, written at the first launch.
  if [[ ! -f "${RUN_DIR}/client_env.json" ]]; then
    write_client_env_snapshot \
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
      "ZERO_OUTPUT_POLICY=${ZERO_OUTPUT_POLICY}" \
      "FETCH_OUTPUT_STREAM=${FETCH_OUTPUT_STREAM}" \
      "LOOP=${LOOP}" \
      "SEND_MODE=${SEND_MODE}" \
      "SEND_MODE_QPS=${SEND_MODE_QPS}" \
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
      "ENABLE_FALLBACK=${ENABLE_FALLBACK:-0}" \
      "ENDPOINTS_FILE=${ENDPOINTS_FILE:-}" \
      "DRY_RUN=${DRY_RUN:-0}"
  fi
  run_java_load_client \
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
    "ZERO_OUTPUT_POLICY=${ZERO_OUTPUT_POLICY}" \
    "FETCH_OUTPUT_STREAM=${FETCH_OUTPUT_STREAM}" \
    "LOOP=${LOOP}" \
    "SEND_MODE=${SEND_MODE}" \
    "SEND_MODE_QPS=${SEND_MODE_QPS}" \
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
    "ENABLE_FALLBACK=${ENABLE_FALLBACK:-0}" \
    "ENDPOINTS_FILE=${ENDPOINTS_FILE:-}" \
    "DRY_RUN=${DRY_RUN:-0}"
}

if [[ "${LOAD_CLIENT_WORKERS}" -le 1 ]]; then
  launch_java_load_client "${RUN_DIR}/load_client" 1 0 "${MAX_CONCURRENCY}" 0 \
    | tee "${RUN_DIR}/client.stdout"
else
  mkdir -p "${RUN_DIR}/load_client"
  curl -fsS -X POST "http://${FLEXLB_HTTP_ADDR}/rtp_llm/server_latency/reset" >/dev/null
  SHARD_MAX_CONCURRENCY=$(( (MAX_CONCURRENCY + LOAD_CLIENT_WORKERS - 1) / LOAD_CLIENT_WORKERS ))
  for ((shard = 0; shard < LOAD_CLIENT_WORKERS; shard++)); do
    shard_dir="${RUN_DIR}/load_client/shard_${shard}"
    launch_java_load_client "${shard_dir}" "${LOAD_CLIENT_WORKERS}" "${shard}" \
      "${SHARD_MAX_CONCURRENCY}" 1 \
      >"${RUN_DIR}/client_shard_${shard}.stdout" 2>&1 &
    CLIENT_PIDS+=("$!")
    append_process_poll_pid "$!" "client_${shard}"
  done

  CLIENT_EXIT=0
  for pid in "${CLIENT_PIDS[@]}"; do
    wait "${pid}" || CLIENT_EXIT=$?
  done

  curl -fsS "http://${FLEXLB_HTTP_ADDR}/rtp_llm/server_latency" \
    >"${RUN_DIR}/load_client/server_latency.json"
  python3 - "${RUN_DIR}/load_client" "${LOAD_CLIENT_WORKERS}" \
    "${CLIENT_PACING_LAG_P99_LIMIT_MS}" <<'PY'
import collections
import json
import math
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
worker_count = int(sys.argv[2])
pacing_limit_ms = float(sys.argv[3])
shards = [
    json.loads((output_dir / f"shard_{index}" / "summary.json").read_text())
    for index in range(worker_count)
]
server = json.loads((output_dir / "server_latency.json").read_text())

rpc_start_ms = []
send_due_ms = []
pacing_lag_ms = []
priority_rows = []
for index in range(worker_count):
    request_path = output_dir / f"shard_{index}" / "per_request.jsonl"
    with request_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            # Java load client emits per-record priority (0 = unset); the
            # legacy Python client does not, so the key may be absent.
            if "priority" in record:
                priority_rows.append((
                    int(record.get("priority") or 0),
                    str(record.get("status", "")),
                    float(record.get("schedule_ms", 0.0) or 0.0),
                ))
            start_ms = float(record.get("send_start_epoch_ms", 0.0) or 0.0)
            if start_ms <= 0:
                continue
            rpc_start_ms.append(start_ms)
            send_due_ms.append(float(record.get("send_due_epoch_ms", 0.0) or 0.0))
            pacing_lag_ms.append(float(record.get("pacing_lag_ms", 0.0) or 0.0))

def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * quantile) - 1)], 3)

def distribution(values):
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": round(max(values), 3),
    }

def rate(values):
    if len(values) < 2:
        return 0.0
    first, last = min(values), max(values)
    return round((len(values) - 1) * 1000.0 / (last - first), 3) if last > first else 0.0

def peak_qps(values, window_ms):
    buckets = collections.Counter(int(value // window_ms) for value in values)
    return round(max(buckets.values(), default=0) * 1000.0 / window_ms, 3)

actual_rpc_start_count = sum(item.get("actual_sent_count", 0) for item in shards)
recorded_result_count = sum(item.get("recorded_result_count", item.get("total_requests", 0)) for item in shards)
sent_task_count = sum(item.get("sent_count", 0) for item in shards)
pacing = distribution(pacing_lag_ms)
success_count = sum(item.get("success_count", 0) for item in shards)
error_count = sum(item.get("error_count", 0) for item in shards)
validity_checks = {
    "zero_errors": error_count == 0,
    "all_scheduled_tasks_started": sent_task_count == actual_rpc_start_count,
    "all_started_rpcs_recorded": actual_rpc_start_count == recorded_result_count,
    "master_arrival_matches_success": server.get("arrival_count", 0) == success_count,
    "master_completion_matches_success": server.get("completion_count", 0) == success_count,
    "client_pacing_p99_within_limit": pacing["p99"] <= pacing_limit_ms,
}
# Uniform send-mode fields are propagated from the shard summaries so the
# aggregated report shows the arrival process (fields absent in replay mode).
send_mode_fields = {}
if shards and shards[0].get("send_mode") == "uniform":
    send_mode_fields = {
        key: shards[0].get(key)
        for key in ("send_mode", "target_qps", "per_shard_qps", "uniform_interval_ms")
    }
summary = {
    "load_client_workers": worker_count,
    **send_mode_fields,
    "sent_task_count": sent_task_count,
    "actual_rpc_start_count": actual_rpc_start_count,
    "recorded_result_count": recorded_result_count,
    "total_requests": recorded_result_count,
    "success_count": success_count,
    "error_count": error_count,
    "actual_send_qps": rate(rpc_start_ms),
    "client_pacing_lag_ms": pacing,
    "client_send_peak_qps": {
        f"{window_ms}ms": peak_qps(rpc_start_ms, window_ms)
        for window_ms in (1, 10, 100, 1000)
    },
    "trace_due_peak_qps": {
        f"{window_ms}ms": peak_qps(send_due_ms, window_ms)
        for window_ms in (1, 10, 100, 1000)
    },
    "server_arrival_qps": server.get("arrival_qps", 0.0),
    "server_completion_qps": server.get("completion_qps", 0.0),
    "schedule_latency_source": "server",
    "schedule_latency_ms": server.get("server_total_ms", {}),
    "server_stage_latency_ms": {
        key: server.get(key, {})
        for key in ("grpc_queue_ms", "route_submit_ms", "batch_wait_ms", "dispatch_ack_ms", "ack_response_ms")
    },
    "shard_summaries": [f"shard_{index}/summary.json" for index in range(worker_count)],
    "validity_checks": validity_checks,
    "test_valid": all(validity_checks.values()),
}
summary["error_rate"] = round(
    summary["error_count"] / summary["total_requests"], 6
) if summary["total_requests"] else 0.0
if priority_rows:
    # Same layout as the Java client shard-level priority_stats:
    # {"<priority>": {total, completed, rejected, avg_schedule_ms}}.
    groups = {}
    for prio, status, schedule_ms in priority_rows:
        group = groups.setdefault(prio, {"total": 0, "completed": 0, "rejected": 0, "sum": 0.0, "n": 0})
        group["total"] += 1
        if status in ("ok", "scheduled"):
            group["completed"] += 1
            if schedule_ms > 0:
                group["sum"] += schedule_ms
                group["n"] += 1
        else:
            group["rejected"] += 1
    summary["priority_stats"] = {
        str(prio): {
            "total": group["total"],
            "completed": group["completed"],
            "rejected": group["rejected"],
            "avg_schedule_ms": round(group["sum"] / group["n"], 3) if group["n"] else 0.0,
        }
        for prio, group in sorted(groups.items())
    }
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
(output_dir / "report.md").write_text(
    "# FlexLB multi-client performance\n\n"
    f"- Load client workers: {worker_count}\n"
    f"- Actual send QPS: {summary['actual_send_qps']}\n"
    f"- Client pacing P99 (ms): {summary['client_pacing_lag_ms']['p99']}\n"
    f"- Server arrival QPS: {summary['server_arrival_qps']}\n"
    f"- Server completion QPS: {summary['server_completion_qps']}\n"
    f"- Total requests: {summary['total_requests']}\n"
    f"- Error count: {summary['error_count']}\n"
    f"- Error rate: {summary['error_rate']}\n"
    f"- Test valid: {summary['test_valid']} ({json.dumps(summary['validity_checks'])})\n"
    f"- Server latency: {json.dumps(summary['schedule_latency_ms'])}\n"
)
print(json.dumps(summary, indent=2))
PY
  if [[ "${CLIENT_EXIT}" -ne 0 ]]; then
    # R12: best-effort consolidation on the load-client failure path so a
    # failed client still leaves a consolidated (analyzable) directory; the
    # CONSOLIDATED sentinel keeps this to one pass per script invocation.
    consolidate_run_outputs_now
    exit "${CLIENT_EXIT}"
  fi
fi

stop_master_counter_poller
stop_secondary_pollers
assert_mock_engine_healthy

if [[ "${SLO_BATCH_DRAIN_SECONDS}" -gt 0 ]]; then
  echo "Waiting ${SLO_BATCH_DRAIN_SECONDS}s for mock task status to drain..."
  sleep "${SLO_BATCH_DRAIN_SECONDS}"
fi

assert_mock_engine_healthy
if [[ "${START_FLEXLB}" == "1" ]]; then
  wait_for_endpoints_ready "${FLEXLB_HTTP_PORT}" "${N_PREFILL}" "${N_DECODE}"
  save_master_info "${RUN_DIR}/master_info_after.json"
  save_master_prometheus "${RUN_DIR}/master_prometheus_after.prom" || true
fi

SLO_ANALYSIS_FILE="${RUN_DIR}/load_client/slo_batch_analysis.json"
if [[ "${SLO_BATCH_ANALYSIS}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/analyze_slo_batch.py" \
    --run-dir "${RUN_DIR}" \
    --master-config "${PROCESS_CONFIG_FILE}" \
    --output "${SLO_ANALYSIS_FILE}" \
    >"${RUN_DIR}/slo_batch_analysis.stdout" || {
      echo "WARNING: failed to analyze SLO batch decisions" >&2
    }
fi

# Consolidate the run directory into the per-component JSON+log layout
# (run_meta/mock/master/client .json + .log, merged per_request.jsonl[.gz];
# see consolidate_run_outputs.py's docstring for the full keep/delete list).
# Runs while the mock cluster and master are still alive (cleanup kills them
# on EXIT), so the final cluster snapshot is captured from the control plane.
# Kept in place on purpose: endpoints.json, flexlb_env.txt, flexlb_profile.jfr,
# load_client/summary.json (the flexlb-online-eval skill reads that exact
# path) and load_client/report.md.

echo "summary=${RUN_DIR}/load_client/summary.json"
echo "run_meta=${RUN_DIR}/run_meta.json"
echo "mock=${RUN_DIR}/mock.json (${RUN_DIR}/mock.log)"
echo "master=${RUN_DIR}/master.json (${RUN_DIR}/master.log)"
echo "client=${RUN_DIR}/client.json (${RUN_DIR}/client.log)"
if [[ -f "${RUN_DIR}/per_request.jsonl" ]]; then
  echo "per_request=${RUN_DIR}/per_request.jsonl"
else
  echo "per_request=${RUN_DIR}/per_request.jsonl.gz"
fi
echo "report=${RUN_DIR}/load_client/report.md"
echo "jfr=${JFR_FILE}"

# K1: consolidate AFTER the summary= / artifact echo above and BEFORE the
# test_valid verdict below. The consolidation's per_request gzip pass can
# take tens of seconds on large runs while the flexlb-online-eval skill's
# timeout window is only DURATION+180s — printing the summary line first
# guarantees the correct exit code and artifact paths are already visible
# even if consolidation is slow or interrupted.
consolidate_run_outputs_now

SUMMARY_FILE="${RUN_DIR}/load_client/summary.json"
if [[ -f "${SUMMARY_FILE}" ]]; then
  TEST_VALID="$(python3 - "${SUMMARY_FILE}" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8")).get("test_valid")
print("unknown" if value is None else str(bool(value)).lower())
PY
)"
  if [[ "${TEST_VALID}" == "false" ]]; then
    echo "INVALID PERFORMANCE RUN: see validity_checks in ${SUMMARY_FILE}" >&2
    exit 1
  fi
fi

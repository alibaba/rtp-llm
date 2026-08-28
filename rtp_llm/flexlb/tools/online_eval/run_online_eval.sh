#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${FLEXLB_DIR}/../.." && pwd)"

FLEXLB_NETWORK_ISOLATED="${FLEXLB_NETWORK_ISOLATED:-0}"
if [[ "${FLEXLB_NETWORK_ISOLATED}" == "1" \
      && "${FLEXLB_NETWORK_NAMESPACE_ACTIVE:-0}" != "1" ]]; then
  exec unshare -Urn bash -c \
    'ip link set lo up; export FLEXLB_NETWORK_NAMESPACE_ACTIVE=1; exec "$@"' \
    bash bash "$0" "$@"
fi
FLEXLB_FAIL_ON_CONCURRENT_TEST="${FLEXLB_FAIL_ON_CONCURRENT_TEST:-1}"

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
MOCK_ENGINE_IMPL="${MOCK_ENGINE_IMPL:-java}"
JAVA_MOCK_ENGINE_JAR="${JAVA_MOCK_ENGINE_JAR:-${FLEXLB_DIR}/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar}"
# Load client implementation switch: java (JavaLoadClient, carries trace
# priority onto the wire) or python (legacy flexlb_load_client.py fallback,
# no priority passthrough). Single env var, no other override layer.
LOAD_CLIENT_IMPL="${LOAD_CLIENT_IMPL:-java}"
JAVA_LOAD_CLIENT_JAR="${JAVA_LOAD_CLIENT_JAR:-${FLEXLB_DIR}/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar}"
JAVA_LOAD_CLIENT_HEAP_SIZE="${JAVA_LOAD_CLIENT_HEAP_SIZE:-16g}"
JAVA_MOCK_EVENT_LOOP_THREADS="${JAVA_MOCK_EVENT_LOOP_THREADS:-32}"
JAVA_MOCK_COMPLETION_THREADS="${JAVA_MOCK_COMPLETION_THREADS:-16}"
# java_mock_stats sampling interval, passed straight to --stats-interval-ms
# (single env, no renaming). Default matches the historical 5s cadence; lower
# to 1000 for fine-grained pressure-test timelines.
JAVA_MOCK_STATS_INTERVAL_MS="${JAVA_MOCK_STATS_INTERVAL_MS:-5000}"
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
N_SHARDS="${N_SHARDS:-64}"  # mock engine 分片数，默认 64（多进程模式）
# HTTP proxy port for the shard launcher.
# Placed above the gRPC engine range to avoid ephemeral port collisions.
MOCK_PROXY_PORT=$((MOCK_BASE_GRPC_PORT + N_PREFILL + N_DECODE + 100 + N_SHARDS))

FLEXLB_HTTP_ADDR="${FLEXLB_HTTP_ADDR:-127.0.0.1:7001}"
FLEXLB_HTTP_PORT="${FLEXLB_HTTP_ADDR##*:}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-7002}"
FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"
START_FLEXLB="${START_FLEXLB:-1}"
START_MOCK="${START_MOCK:-1}"
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"

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
SCHEDULE_ONLY="${SCHEDULE_ONLY:-0}"
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
JFR_FILE="${JFR_FILE:-${RUN_DIR}/flexlb_profile.jfr}"
JFR_DURATION="${JFR_DURATION:-300s}"
FLEXLB_MONITOR_ENABLED="${FLEXLB_MONITOR_ENABLED:-true}"
FLEXLB_MONITOR_MODE="${FLEXLB_MONITOR_MODE:-critical-only}"
HIPPO_ROLE="${HIPPO_ROLE:-test}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${HOME}/.venvs/flexlb-eval/bin/python3" ]]; then
    PYTHON_BIN="${HOME}/.venvs/flexlb-eval/bin/python3"
  else
    PYTHON_BIN="$(command -v python3 || true)"
  fi
fi
# The aiohttp/grpc venv is only needed by the Python load client / mock engine.
if [[ "${LOAD_CLIENT_IMPL}" == "python" || "${MOCK_ENGINE_IMPL}" == "python" ]]; then
  if [[ -z "${PYTHON_BIN}" ]] \
      || ! "${PYTHON_BIN}" -c 'import aiohttp, grpc' >/dev/null 2>&1; then
    echo "Python with aiohttp and grpc is required; set PYTHON_BIN to the eval venv" >&2
    exit 1
  fi
fi

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
        "candidateChoice": {"type": "RANDOM_WITHIN_TOLERANCE"}
      },
      "decode": {
        "availability": {"maxKvUsagePercent": 90, "maxEngineRequests": 132},
        "kvReservation": {"maxOutputTokensForEstimate": 1000}
      }
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

# These are independent transport settings, not FLEXLB_CONFIG fields.
export FLEXLB_GRPC_EXECUTOR_CORE_SIZE="${FLEXLB_GRPC_EXECUTOR_CORE_SIZE:-128}"
export FLEXLB_GRPC_EXECUTOR_MAX_SIZE="${FLEXLB_GRPC_EXECUTOR_MAX_SIZE:-128}"
# FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE: no script default — code default (1000) applies
# unless the caller exports it explicitly (still forwarded via the environment).

MOCK_PID=""
FLEXLB_PID=""
MASTER_COUNTER_POLLER_PID=""
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

java_major() {
  local java_bin="${1:-java}"
  "${java_bin}" -version 2>&1 | awk -F'[\".]' '/version/ {print ($2 == "1" ? $3 : $2); exit}'
}

detect_java21_home() {
  if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    if [[ "$(java_major "${JAVA_HOME}/bin/java")" -ge 21 ]]; then
      echo "${JAVA_HOME}"
      return 0
    fi
  fi
  if [[ -n "${JAVA21_HOME:-}" && -x "${JAVA21_HOME}/bin/java" ]]; then
    echo "${JAVA21_HOME}"
    return 0
  fi
  if [[ -x "${HOME}/java21/bin/java" \
        && "$(java_major "${HOME}/java21/bin/java")" -ge 21 ]]; then
    echo "${HOME}/java21"
    return 0
  fi
  local java_bin
  while IFS= read -r java_bin; do
    if [[ -x "${java_bin}" && "$(java_major "${java_bin}")" -ge 21 ]]; then
      dirname "$(dirname "${java_bin}")"
      return 0
    fi
  done < <(
    {
      alternatives --display java 2>/dev/null || true
      update-alternatives --display java 2>/dev/null || true
    } | awk '/bin\/java/ {print $1}' | sort -u
  )
  return 1
}

JAVA21_HOME_DETECTED="$(detect_java21_home || true)"
if [[ -n "${JAVA21_HOME_DETECTED}" ]]; then
  export JAVA_HOME="${JAVA21_HOME_DETECTED}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

cleanup() {
  stop_master_counter_poller
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
  matches="$(pgrep -af 'flexlb_load_client\.py|mock_engine_shard_launcher\.py|flexlb-api-[^ ]*\.jar|flexlb-mock-engine-[^ ]*\.jar' || true)"
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
echo "LOAD_CLIENT_IMPL=${LOAD_CLIENT_IMPL} (java=JavaLoadClient with trace priority passthrough, python=legacy flexlb_load_client.py)"
if [[ "${LOAD_CLIENT_IMPL}" != "java" && "${LOAD_CLIENT_IMPL}" != "python" ]]; then
  echo "Unsupported LOAD_CLIENT_IMPL=${LOAD_CLIENT_IMPL}; expected java or python" >&2
  exit 1
fi
if [[ "${LOAD_CLIENT_IMPL}" == "java" ]]; then
  if [[ ! -f "${JAVA_LOAD_CLIENT_JAR}" ]]; then
    echo "Java load client jar not found: ${JAVA_LOAD_CLIENT_JAR}" >&2
    echo "Build it with: ./mvnw package -DskipTests -P '!internal'" >&2
    exit 1
  fi
  if [[ "$(java_major java)" -lt 21 ]]; then
    echo "Java 21 is required to run JavaLoadClient. Set JAVA21_HOME or JAVA_HOME." >&2
    exit 1
  fi
fi

if [[ "${FLEXLB_FAIL_ON_CONCURRENT_TEST}" == "1" ]]; then
  assert_no_concurrent_flexlb_test
fi

ENDPOINT_FILE="${RUN_DIR}/endpoints.json"
FLEXLB_ENV_FILE="${RUN_DIR}/flexlb_env.txt"

if [[ "${START_MOCK}" == "1" ]]; then
  if [[ "${MOCK_ENGINE_IMPL}" == "java" ]]; then
    mapfile -t JAVA_MOCK_PORTS < <(seq "${MOCK_BASE_GRPC_PORT}" \
      "$((MOCK_BASE_GRPC_PORT + N_PREFILL + N_DECODE - 1))")
    assert_ports_free "${JAVA_MOCK_PORTS[@]}"
    if [[ ! -f "${JAVA_MOCK_ENGINE_JAR}" ]]; then
      echo "Java mock engine jar not found: ${JAVA_MOCK_ENGINE_JAR}" >&2
      echo "Build it with: ./mvnw package -DskipTests -P '!internal'" >&2
      exit 1
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
      >"${RUN_DIR}/mock_engine.log" 2>&1 &
    MOCK_PID="$!"
    echo "Java mock engine heap: Xms=${JAVA_MOCK_JVM_XMS}, Xmx=${JAVA_MOCK_JVM_XMX}"
    echo "Java mock engine stats interval: ${JAVA_MOCK_STATS_INTERVAL_MS}ms"
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
  elif [[ "${MOCK_ENGINE_IMPL}" == "python" ]]; then
    MOCK_ENGINE_SCRIPT="${SCRIPT_DIR}/mock_engine_cluster.py"
    MOCK_ENGINE_EXTRA_ARGS=()
    if [[ "${N_SHARDS}" -gt 1 ]]; then
      MOCK_ENGINE_SCRIPT="${SCRIPT_DIR}/mock_engine_shard_launcher.py"
      MOCK_ENGINE_EXTRA_ARGS=(--n-shards "${N_SHARDS}")
    fi
    PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${MOCK_ENGINE_SCRIPT}" \
      --n-prefill "${N_PREFILL}" \
      --n-decode "${N_DECODE}" \
      --base-grpc-port "${MOCK_BASE_GRPC_PORT}" \
      --performance "${PERFORMANCE_FILE}" \
      --master-config "${PROCESS_CONFIG_FILE}" \
      --prefill-cache-blocks "${PREFILL_CACHE_BLOCKS}" \
      --decode-cache-blocks "${DECODE_CACHE_BLOCKS}" \
      --endpoint-file "${ENDPOINT_FILE}" \
      --env-file "${FLEXLB_ENV_FILE}" \
      "${MOCK_ENGINE_EXTRA_ARGS[@]}" \
      >"${RUN_DIR}/mock_engine.log" 2>&1 &
    MOCK_PID="$!"
    if [[ "${N_SHARDS}" -gt 1 ]]; then
      wait_for_port "127.0.0.1" "${MOCK_PROXY_PORT}" 180
    else
      wait_for_port "127.0.0.1" "${MOCK_BASE_GRPC_PORT}" 20
    fi
  else
    echo "Unsupported MOCK_ENGINE_IMPL=${MOCK_ENGINE_IMPL}; expected java or python" >&2
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
  while IFS= read -r -d '' process_env; do
    PROCESS_ENV_ARGS+=("${process_env}")
  done < <(python3 - "${PROCESS_CONFIG_FILE}" <<'PY'
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
    env "${FLEXLB_ENV_ARGS[@]}" "${PROCESS_ENV_ARGS[@]}" "${RUNTIME_OVERRIDE_ENV_ARGS[@]}" \
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
  if [[ "${FLEXLB_WARMUP_SECONDS:-0}" -gt 0 ]]; then
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

CLIENT_ARGS=(
  "${TRACE_FILE}"
  --flexlb-http-addr "${FLEXLB_HTTP_ADDR}"
  --replay-speed "${REPLAY_SPEED}"
  --duration-s "${DURATION_S}"
  --limit "${LIMIT}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --timeout-ms "${TIMEOUT_MS}"
  --sla-ttft-ms "${SLA_TTFT_MS}"
  --zero-output-policy "${ZERO_OUTPUT_POLICY}"
  --output-dir "${RUN_DIR}/load_client"
  --start-at-epoch-ms "${CLIENT_START_EPOCH_MS}"
)
if [[ "${SCHEDULE_ONLY}" == "1" ]]; then
  CLIENT_ARGS+=(--schedule-only)
fi
if [[ "${LOOP}" == "1" ]]; then
  CLIENT_ARGS+=(--loop)
fi
if [[ -n "${RESPONSE_TIMEOUT:-}" ]]; then
  CLIENT_ARGS+=(--response-timeout "${RESPONSE_TIMEOUT}")
fi
if [[ -n "${PUSHGATEWAY_URL}" ]]; then
  CLIENT_ARGS+=(--pushgateway-url "${PUSHGATEWAY_URL}")
fi
if [[ -n "${MAX_INPUT_LEN}" && "${MAX_INPUT_LEN}" != "0" ]]; then
  CLIENT_ARGS+=(--max-input-len "${MAX_INPUT_LEN}")
fi
if [[ -n "${MAX_OUTPUT_LEN}" && "${MAX_OUTPUT_LEN}" != "0" ]]; then
  CLIENT_ARGS+=(--max-output-len "${MAX_OUTPUT_LEN}")
fi
if [[ "${GRADIENT}" == "1" ]]; then
  CLIENT_ARGS+=(--gradient --gradient-max-speed "${GRADIENT_MAX_SPEED}" --gradient-start-speed "${GRADIENT_START_SPEED}")
fi

# JavaLoadClient reads its configuration exclusively from environment
# variables (no CLI flags); mirror CLIENT_ARGS one-to-one per shard.
# PRIORITY is deliberately not set: priority comes from the trace records
# only, and records without one stay unset on the wire (no default 50).
launch_java_load_client() {
  local output_dir="$1"
  local num_shards="$2"
  local shard_index="$3"
  local max_concurrency="$4"
  local skip_server_latency="$5"
  env \
    TRACE_FILE="${TRACE_FILE}" \
    TARGET_ADDR="${TARGET_ADDR:-${FLEXLB_HTTP_ADDR}}" \
    OUTPUT_DIR="${output_dir}" \
    NUM_SHARDS="${num_shards}" \
    SHARD_INDEX="${shard_index}" \
    MAX_CONCURRENCY="${max_concurrency}" \
    SKIP_SERVER_LATENCY="${skip_server_latency}" \
    REPLAY_SPEED="${REPLAY_SPEED}" \
    DURATION_S="${DURATION_S}" \
    LIMIT="${LIMIT}" \
    TIMEOUT_MS="${TIMEOUT_MS}" \
    SLA_TTFT_MS="${SLA_TTFT_MS}" \
    ZERO_OUTPUT_POLICY="${ZERO_OUTPUT_POLICY}" \
    SCHEDULE_ONLY="${SCHEDULE_ONLY}" \
    LOOP="${LOOP}" \
    SEND_MODE="${SEND_MODE}" \
    SEND_MODE_QPS="${SEND_MODE_QPS}" \
    GRADIENT="${GRADIENT}" \
    GRADIENT_START_SPEED="${GRADIENT_START_SPEED}" \
    GRADIENT_MAX_SPEED="${GRADIENT_MAX_SPEED}" \
    MAX_INPUT_LEN="${MAX_INPUT_LEN}" \
    MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN}" \
    PUSHGATEWAY_URL="${PUSHGATEWAY_URL}" \
    RESPONSE_TIMEOUT="${RESPONSE_TIMEOUT:-}" \
    START_AT_EPOCH_MS="${CLIENT_START_EPOCH_MS}" \
    LOAD_CLIENT_WORKERS="${LOAD_CLIENT_WORKERS}" \
    java -Xmx"${JAVA_LOAD_CLIENT_HEAP_SIZE}" -XX:+ExitOnOutOfMemoryError \
      -cp "${JAVA_LOAD_CLIENT_JAR}" org.flexlb.mockengine.JavaLoadClient
}

if [[ "${LOAD_CLIENT_WORKERS}" -le 1 ]]; then
  if [[ "${LOAD_CLIENT_IMPL}" == "java" ]]; then
    launch_java_load_client "${RUN_DIR}/load_client" 1 0 "${MAX_CONCURRENCY}" 0 \
      | tee "${RUN_DIR}/client.stdout"
  else
    PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/flexlb_load_client.py" "${CLIENT_ARGS[@]}" | tee "${RUN_DIR}/client.stdout"
  fi
else
  mkdir -p "${RUN_DIR}/load_client"
  curl -fsS -X POST "http://${FLEXLB_HTTP_ADDR}/rtp_llm/server_latency/reset" >/dev/null
  SHARD_MAX_CONCURRENCY=$(( (MAX_CONCURRENCY + LOAD_CLIENT_WORKERS - 1) / LOAD_CLIENT_WORKERS ))
  for ((shard = 0; shard < LOAD_CLIENT_WORKERS; shard++)); do
    shard_dir="${RUN_DIR}/load_client/shard_${shard}"
    if [[ "${LOAD_CLIENT_IMPL}" == "java" ]]; then
      launch_java_load_client "${shard_dir}" "${LOAD_CLIENT_WORKERS}" "${shard}" \
        "${SHARD_MAX_CONCURRENCY}" 1 \
        >"${RUN_DIR}/client_shard_${shard}.stdout" 2>&1 &
    else
      PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/flexlb_load_client.py" \
        "${CLIENT_ARGS[@]}" \
        --output-dir "${shard_dir}" \
        --num-shards "${LOAD_CLIENT_WORKERS}" \
        --shard-index "${shard}" \
        --max-concurrency "${SHARD_MAX_CONCURRENCY}" \
        --skip-server-latency \
        >"${RUN_DIR}/client_shard_${shard}.stdout" 2>&1 &
    fi
    CLIENT_PIDS+=("$!")
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
summary = {
    "load_client_workers": worker_count,
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
    exit "${CLIENT_EXIT}"
  fi
fi

stop_master_counter_poller
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

echo "summary=${RUN_DIR}/load_client/summary.json"
if [[ "${LOAD_CLIENT_WORKERS}" -le 1 ]]; then
  echo "per_request=${RUN_DIR}/load_client/per_request.jsonl"
else
  echo "per_request_shards=${RUN_DIR}/load_client/shard_*/per_request.jsonl"
fi
echo "report=${RUN_DIR}/load_client/report.md"
echo "server_latency=${RUN_DIR}/load_client/server_latency.json"
echo "slo_batch_analysis=${SLO_ANALYSIS_FILE}"
echo "flexlb_file_log=${FLEXLB_LOG_PATH}/flexlb.log"
echo "master_counters_timeseries=${MASTER_COUNTERS_FILE}"
echo "jfr=${JFR_FILE}"

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

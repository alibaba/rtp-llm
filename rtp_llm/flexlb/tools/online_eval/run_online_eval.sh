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
JAVA_MOCK_EVENT_LOOP_THREADS="${JAVA_MOCK_EVENT_LOOP_THREADS:-32}"
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
SCHEDULE_MODE="${SCHEDULE_MODE:-batch}"
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
if [[ -z "${PYTHON_BIN}" ]] \
    || ! "${PYTHON_BIN}" -c 'import aiohttp, grpc' >/dev/null 2>&1; then
  echo "Python with aiohttp and grpc is required; set PYTHON_BIN to the eval venv" >&2
  exit 1
fi

DEFAULT_FLEXLB_CONFIG='{"loadBalanceStrategy":"COST_BASED_PREFILL","decodeLoadBalanceStrategy":"COST_BASED_DECODE","cacheHitMaxCacheKeys":10000000,"cacheHitMetricReportEnabled":true,"cacheHitTimeWindowMs":1800000,"cacheHitTraceLogEnabled":false,"cacheHitWindowWriteEnabled":true,"decodeConcurrencyLimit":132,"flexlbBatchAlgorithm":"fixed_window","flexlbBatchFixedWaitMs":10,"flexlbBatchPredictThresholdMs":550,"flexlbBatchSizeMax":32,"hysteresisBiasPercent":30,"maxQueueSize":1000000,"flexlbBatchMaxInflight":1000000,"flexlbBatchDispatchPoolSize":500,"flexlbBatchDispatchQueueSize":10000,"prefillQueueSizeThreshold":100000,"defaultScheduleMode":"BATCH","flexlbBatchFixedMaxInflightBatches":-1,"costSloMs":1000,"flexlbBatchMinSize":8,"prefillLbTimeoutMs":5000}'
DEFAULT_STRATEGY_CONFIGS='{"shortestTtft":{"candidatePool":{"mode":"FIXED","size":2}}}'
FLEXLB_CONFIG="${FLEXLB_CONFIG:-${DEFAULT_FLEXLB_CONFIG}}"
STRATEGY_CONFIGS="${STRATEGY_CONFIGS:-${DEFAULT_STRATEGY_CONFIGS}}"
OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN:-.*}"
OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-none}"
HIPPO_ROLE="${HIPPO_ROLE:-flexlb_eval_master}"

# ========== Thread Pool Size Configuration ==========
# These defaults keep total threads <1000 on high-core machines.
export GRPC_CLIENT_EXECUTOR_CORE_SIZE="${GRPC_CLIENT_EXECUTOR_CORE_SIZE:-32}"
export GRPC_CLIENT_EXECUTOR_MAX_SIZE="${GRPC_CLIENT_EXECUTOR_MAX_SIZE:-32}"
export GRPC_CLIENT_EXECUTOR_QUEUE_SIZE="${GRPC_CLIENT_EXECUTOR_QUEUE_SIZE:-10000}"
export GRPC_CLIENT_EVENT_LOOP_THREADS="${GRPC_CLIENT_EVENT_LOOP_THREADS:-8}"
export GRPC_SERVER_WORKER_EVENT_LOOP_THREADS="${GRPC_SERVER_WORKER_EVENT_LOOP_THREADS:-4}"
export FLEXLB_N_CHANNELS="${FLEXLB_N_CHANNELS:-16}"
export HTTP_NETTY_EVENT_LOOP_THREADS="${HTTP_NETTY_EVENT_LOOP_THREADS:-4}"
export HTTP_NETTY_EVENT_EXECUTOR_THREADS="${HTTP_NETTY_EVENT_EXECUTOR_THREADS:-16}"
export HTTP_NETTY_EVENT_EXECUTOR_QUEUE_SIZE="${HTTP_NETTY_EVENT_EXECUTOR_QUEUE_SIZE:-1000}"
export HTTP_REQUEST_EXECUTOR_CORE_SIZE="${HTTP_REQUEST_EXECUTOR_CORE_SIZE:-32}"
export HTTP_REQUEST_EXECUTOR_MAX_SIZE="${HTTP_REQUEST_EXECUTOR_MAX_SIZE:-32}"
export HTTP_REQUEST_EXECUTOR_QUEUE_SIZE="${HTTP_REQUEST_EXECUTOR_QUEUE_SIZE:-10000}"
export ENGINE_SYNC_EXECUTOR_CORE_SIZE="${ENGINE_SYNC_EXECUTOR_CORE_SIZE:-32}"
export ENGINE_SYNC_EXECUTOR_MAX_SIZE="${ENGINE_SYNC_EXECUTOR_MAX_SIZE:-64}"
export STATUS_CHECK_EXECUTOR_CORE_SIZE="${STATUS_CHECK_EXECUTOR_CORE_SIZE:-32}"
export STATUS_CHECK_EXECUTOR_MAX_SIZE="${STATUS_CHECK_EXECUTOR_MAX_SIZE:-64}"
export SERVICE_DISCOVERY_MAX_SIZE="${SERVICE_DISCOVERY_MAX_SIZE:-32}"
export NETTY_SELECT_THREAD_MULTIPLIER="${NETTY_SELECT_THREAD_MULTIPLIER:-1}"
export NETTY_WORKER_THREAD_MULTIPLIER="${NETTY_WORKER_THREAD_MULTIPLIER:-1}"
export FLEXLB_GRPC_EXECUTOR_CORE_SIZE="${FLEXLB_GRPC_EXECUTOR_CORE_SIZE:-128}"
export FLEXLB_GRPC_EXECUTOR_MAX_SIZE="${FLEXLB_GRPC_EXECUTOR_MAX_SIZE:-128}"
export FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE="${FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE:-50000}"
export SCHEDULE_WORKER_SIZE="${SCHEDULE_WORKER_SIZE:-16}"

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
  python3 - "$@" <<'PY'
import socket
import sys

sockets = []
try:
    for raw_port in sys.argv[1:]:
        port = int(raw_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            print(f"required port {port} is not available: {exc}", file=sys.stderr)
            sys.exit(1)
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
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
      --performance "${PERFORMANCE_FILE}" \
      --master-config "${PROCESS_CONFIG_FILE}" \
      --prefill-cache-blocks "${PREFILL_CACHE_BLOCKS}" \
      --decode-cache-blocks "${DECODE_CACHE_BLOCKS}" \
      --endpoint-file "${ENDPOINT_FILE}" \
      --env-file "${FLEXLB_ENV_FILE}" \
      >"${RUN_DIR}/mock_engine.log" 2>&1 &
    MOCK_PID="$!"
    echo "Java mock engine heap: Xms=${JAVA_MOCK_JVM_XMS}, Xmx=${JAVA_MOCK_JVM_XMX}"
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

RUNTIME_OVERRIDE_ENV_ARGS=()
OVERRIDE_ENV_KEYS=(
  CACHE_HIT_MAX_CACHE_KEYS
  CACHE_HIT_METRIC_REPORT_ENABLED
  CACHE_HIT_TIME_WINDOW_MS
  CACHE_HIT_TRACE_LOG_ENABLED
  CACHE_HIT_WINDOW_WRITE_ENABLED
  COST_ALPHA0
  COST_ALPHA1
  COST_ALPHA2
  COST_ALPHA3
  COST_ALPHA4
  COST_ALPHA5
  COST_SLO_MS
  DECODE_CONCURRENCY_LIMIT
  DECODE_LOAD_BALANCE_STRATEGY
  DEFAULT_SCHEDULE_MODE
  ENGINE_SYNC_EXECUTOR_CORE_SIZE
  ENGINE_SYNC_EXECUTOR_MAX_SIZE
  FLEXLB_BATCH_ALGORITHM
  FLEXLB_BATCH_DISPATCH_POOL_SIZE
  FLEXLB_BATCH_DISPATCH_QUEUE_SIZE
  FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES
  FLEXLB_BATCH_FIXED_WAIT_MS
  FLEXLB_BATCH_MAX_INFLIGHT
  FLEXLB_BATCH_MIN_SIZE
  FLEXLB_BATCH_PREDICT_THRESHOLD_MS
  FLEXLB_BATCH_SIZE_MAX
  FLEXLB_GRPC_EXECUTOR_CORE_SIZE
  FLEXLB_GRPC_EXECUTOR_MAX_SIZE
  FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE
  FLEXLB_JVM_HEAP_SIZE
  FLEXLB_MONITOR_ENABLED
  FLEXLB_MONITOR_MODE
  GRADIENT
  GRADIENT_MAX_SPEED
  GRADIENT_START_SPEED
  GRPC_CLIENT_EVENT_LOOP_THREADS
  GRPC_CLIENT_EXECUTOR_CORE_SIZE
  GRPC_CLIENT_EXECUTOR_MAX_SIZE
  GRPC_CLIENT_EXECUTOR_QUEUE_SIZE
  GRPC_SERVER_WORKER_EVENT_LOOP_THREADS
  HTTP_NETTY_EVENT_EXECUTOR_QUEUE_SIZE
  HTTP_NETTY_EVENT_EXECUTOR_THREADS
  HTTP_NETTY_EVENT_LOOP_THREADS
  HTTP_REQUEST_EXECUTOR_CORE_SIZE
  HTTP_REQUEST_EXECUTOR_MAX_SIZE
  HTTP_REQUEST_EXECUTOR_QUEUE_SIZE
  HYSTERESIS_BIAS_PERCENT
  LOAD_BALANCE_STRATEGY
  MAX_QUEUE_SIZE
  NETTY_SELECT_THREAD_MULTIPLIER
  NETTY_WORKER_THREAD_MULTIPLIER
  PREFILL_QUEUE_SIZE_THRESHOLD
  PREFILL_TIME_FORMULA
  SCHEDULE_WORKER_SIZE
  SERVICE_DISCOVERY_MAX_SIZE
  STATUS_CHECK_EXECUTOR_CORE_SIZE
  STATUS_CHECK_EXECUTOR_MAX_SIZE
  SYNC_REQUEST_TIMEOUT_MS
  SYNC_STATUS_INTERVAL
)
for key in "${OVERRIDE_ENV_KEYS[@]}"; do
  if [[ -v "${key}" ]]; then
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
      "STRATEGY_CONFIGS=${STRATEGY_CONFIGS}" \
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
      "STRATEGY_CONFIGS=${STRATEGY_CONFIGS}" \
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

CLIENT_ARGS=(
  "${TRACE_FILE}"
  --flexlb-http-addr "${FLEXLB_HTTP_ADDR}"
  --schedule-mode "${SCHEDULE_MODE}"
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

if [[ "${LOAD_CLIENT_WORKERS}" -le 1 ]]; then
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/flexlb_load_client.py" "${CLIENT_ARGS[@]}" | tee "${RUN_DIR}/client.stdout"
else
  mkdir -p "${RUN_DIR}/load_client"
  curl -fsS -X POST "http://${FLEXLB_HTTP_ADDR}/rtp_llm/server_latency/reset" >/dev/null
  SHARD_MAX_CONCURRENCY=$(( (MAX_CONCURRENCY + LOAD_CLIENT_WORKERS - 1) / LOAD_CLIENT_WORKERS ))
  for ((shard = 0; shard < LOAD_CLIENT_WORKERS; shard++)); do
    shard_dir="${RUN_DIR}/load_client/shard_${shard}"
    PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/flexlb_load_client.py" \
      "${CLIENT_ARGS[@]}" \
      --output-dir "${shard_dir}" \
      --num-shards "${LOAD_CLIENT_WORKERS}" \
      --shard-index "${shard}" \
      --max-concurrency "${SHARD_MAX_CONCURRENCY}" \
      --skip-server-latency \
      >"${RUN_DIR}/client_shard_${shard}.stdout" 2>&1 &
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
for index in range(worker_count):
    request_path = output_dir / f"shard_{index}" / "per_request.jsonl"
    with request_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
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

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${FLEXLB_DIR}/../.." && pwd)"

TRACE_FILE="${TRACE_FILE:-${SCRIPT_DIR}/data/online_logs/trace_30min.jsonl}"
PERFORMANCE_FILE="${PERFORMANCE_FILE:-${SCRIPT_DIR}/data/performance/dsv4_flash_performance.sample.json}"
PROCESS_CONFIG_FILE="${PROCESS_CONFIG_FILE:-${SCRIPT_DIR}/data/config/master_fixed_window_220ms.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"

N_PREFILL="${N_PREFILL:-2}"
N_DECODE="${N_DECODE:-4}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-61000}"
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
JFR_FILE="${JFR_FILE:-${RUN_DIR}/flexlb_profile.jfr}"
JFR_DURATION="${JFR_DURATION:-300s}"

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
export FLEXLB_GRPC_EXECUTOR_CORE_SIZE="${FLEXLB_GRPC_EXECUTOR_CORE_SIZE:-64}"
export FLEXLB_GRPC_EXECUTOR_MAX_SIZE="${FLEXLB_GRPC_EXECUTOR_MAX_SIZE:-64}"
export FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE="${FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE:-10000}"
export SCHEDULE_WORKER_SIZE="${SCHEDULE_WORKER_SIZE:-16}"

MOCK_PID=""
FLEXLB_PID=""
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
  if [[ -n "${FLEXLB_PID}" ]]; then
    kill "${FLEXLB_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${MOCK_PID}" ]]; then
    kill "${MOCK_PID}" >/dev/null 2>&1 || true
  fi
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

wait_for_endpoints_ready() {
  local master_port=$1
  local expected_prefill=$2
  local expected_decode=$3
  local max_wait=120
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

  echo "[wait_for_endpoints_ready] WARNING: Timeout after ${max_wait}s waiting for endpoints. Proceeding anyway."
  return 0
}

mkdir -p "${RUN_DIR}"
echo "run_dir=${RUN_DIR}"

ENDPOINT_FILE="${RUN_DIR}/endpoints.json"
FLEXLB_ENV_FILE="${RUN_DIR}/flexlb_env.txt"

if [[ "${START_MOCK}" == "1" ]]; then
  MOCK_ENGINE_SCRIPT="${SCRIPT_DIR}/mock_engine_cluster.py"
  MOCK_ENGINE_EXTRA_ARGS=()
  if [[ "${N_SHARDS}" -gt 1 ]]; then
    MOCK_ENGINE_SCRIPT="${SCRIPT_DIR}/mock_engine_shard_launcher.py"
    MOCK_ENGINE_EXTRA_ARGS=(--n-shards "${N_SHARDS}")
  fi
  PYTHONDONTWRITEBYTECODE=1 python3 "${MOCK_ENGINE_SCRIPT}" \
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
    # 多进程模式：等待 launcher HTTP 代理端口（endpoints.json 合并后才启动）
    wait_for_port "127.0.0.1" "${MOCK_PROXY_PORT}" 180
  else
    # 单进程模式：等待第一个 gRPC 端口
    wait_for_port "127.0.0.1" "${MOCK_BASE_GRPC_PORT}" 20
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
  if [[ -n "${JVM_HEAP_SIZE}" ]]; then
    JAVA_HEAP_OPTS=(-Xms"${JVM_HEAP_SIZE}" -Xmx"${JVM_HEAP_SIZE}")
  fi
fi

if [[ "${START_FLEXLB}" == "1" ]]; then
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
      java -XX:StartFlightRecording=filename=${JFR_FILE},settings=profile,duration=${JFR_DURATION},disk=true,maxsize=256m,dumponexit=true "${JAVA_HEAP_OPTS[@]}" "${JAVA_MODULE_OPTS[@]}" "${JVM_SYSTEM_PROPS[@]}" -jar "${FLEXLB_JAR}" \
      --server.port="${FLEXLB_HTTP_PORT}" \
      --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
      --spring.profiles.active="${SPRING_PROFILE:-default}" \
      >"${RUN_DIR}/flexlb.log" 2>&1 &
  fi
  FLEXLB_PID="$!"
  wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60
  wait_for_endpoints_ready "${FLEXLB_HTTP_PORT}" "${N_PREFILL}" "${N_DECODE}"
fi

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

PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/flexlb_load_client.py" "${CLIENT_ARGS[@]}" | tee "${RUN_DIR}/client.stdout"

echo "summary=${RUN_DIR}/load_client/summary.json"
echo "per_request=${RUN_DIR}/load_client/per_request.jsonl"
echo "report=${RUN_DIR}/load_client/report.md"
echo "jfr=${JFR_FILE}"

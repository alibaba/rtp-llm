#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# run_cancel_smoke.sh — One-shot orchestration for FlexLB cancel smoke tests.
#
# 1. Start MockEngineCluster (2 prefill + 2 decode, with delays for cancel)
# 2. Generate service-discovery env vars
# 3. Build + start flexlb-api master (Java)
# 4. Run cancel_smoke.py
# 5. Collect results
# 6. Cleanup (trap EXIT)
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${FLEXLB_DIR}/../.." && pwd)"

# -- Configurable parameters ----------------------------------------------

RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run}"
RUN_ID="${RUN_ID:-cancel_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"

N_PREFILL="${N_PREFILL:-2}"
N_DECODE="${N_DECODE:-4}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-55151}"
PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS:-6000}"
DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS:-3000}"

FLEXLB_HTTP_PORT="${FLEXLB_HTTP_PORT:-18080}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-18081}"
FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"

START_FLEXLB="${START_FLEXLB:-1}"
START_MOCK="${START_MOCK:-1}"
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"

SCHEDULE_MODE="${SCHEDULE_MODE:-batch}"

# Performance config for cancel tests: enough delay for cancel window.
# prefill=100ms fixed, decode=20ms/step × 10 steps = 200ms total decode.
PERF_CONFIG_DIR="${RUN_DIR}/perf"
PERF_CONFIG_FILE="${PERF_CONFIG_DIR}/cancel_smoke_perf.json"

# FlexLB master config — flattened individual env vars (override via environment)
LOAD_BALANCE_STRATEGY="${LOAD_BALANCE_STRATEGY:-COST_BASED_PREFILL}"
DECODE_LOAD_BALANCE_STRATEGY="${DECODE_LOAD_BALANCE_STRATEGY:-COST_BASED_DECODE}"
DECODE_CONCURRENCY_LIMIT="${DECODE_CONCURRENCY_LIMIT:-132}"
FLEXLB_BATCH_ALGORITHM="${FLEXLB_BATCH_ALGORITHM:-fixed_window}"
FLEXLB_BATCH_FIXED_WAIT_MS="${FLEXLB_BATCH_FIXED_WAIT_MS:-10}"
FLEXLB_BATCH_PREDICT_THRESHOLD_MS="${FLEXLB_BATCH_PREDICT_THRESHOLD_MS:-550}"
FLEXLB_BATCH_SIZE_MAX="${FLEXLB_BATCH_SIZE_MAX:-32}"
FLEXLB_BATCH_MIN_SIZE="${FLEXLB_BATCH_MIN_SIZE:-1}"
FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES:-4}"
HYSTERESIS_BIAS_PERCENT="${HYSTERESIS_BIAS_PERCENT:-30}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-5000}"
PREFILL_QUEUE_SIZE_THRESHOLD="${PREFILL_QUEUE_SIZE_THRESHOLD:-100000}"
DEFAULT_SCHEDULE_MODE="${DEFAULT_SCHEDULE_MODE:-BATCH}"
COST_SLO_MS="${COST_SLO_MS:-30000}"
STRATEGY_CONFIGS='{}'

OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN:-.*}"
OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-none}"
HIPPO_ROLE="${HIPPO_ROLE:-flexlb_cancel_smoke_master}"

# -- Internal state --------------------------------------------------------

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

# -- Helpers ---------------------------------------------------------------

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
  echo ""
  echo "[cleanup] stopping processes ..."
  if [[ -n "${FLEXLB_PID}" ]]; then
    kill "${FLEXLB_PID}" >/dev/null 2>&1 || true
    wait "${FLEXLB_PID}" 2>/dev/null || true
    FLEXLB_PID=""
  fi
  if [[ -n "${MOCK_PID}" ]]; then
    kill "${MOCK_PID}" >/dev/null 2>&1 || true
    wait "${MOCK_PID}" 2>/dev/null || true
    MOCK_PID=""
  fi
  echo "[cleanup] done."
}
trap cleanup EXIT

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  python3 - "$host" "$port" "$timeout_s" <<'PY'
import socket, sys, time
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

# -- Setup -----------------------------------------------------------------

mkdir -p "${RUN_DIR}"
echo "run_dir=${RUN_DIR}"

ENDPOINT_FILE="${RUN_DIR}/endpoints.json"
FLEXLB_ENV_FILE="${RUN_DIR}/flexlb_env.txt"

# Generate performance config for cancel tests
mkdir -p "${PERF_CONFIG_DIR}"
cat > "${PERF_CONFIG_FILE}" <<'JSON'
{
  "block_size": 1024,
  "sleep_scale": 1.0,
  "prefill": {
    "fixed_ms": 100.0,
    "scale": 1.0
  },
  "decode": {
    "scale": 1.0,
    "step_ms_by_batch": [
      [1, 20.0],
      [2, 22.0],
      [4, 25.0],
      [8, 28.0],
      [16, 30.0],
      [32, 35.0],
      [64, 40.0],
      [128, 45.0],
      [256, 50.0]
    ]
  }
}
JSON
echo "perf_config=${PERF_CONFIG_FILE}"

# -- Start mock engine cluster --------------------------------------------

if [[ "${START_MOCK}" == "1" ]]; then
  echo ""
  echo "[1/4] Starting mock engine cluster (${N_PREFILL} prefill, ${N_DECODE} decode) ..."
  PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/mock_engine_cluster.py" \
    --n-prefill "${N_PREFILL}" \
    --n-decode "${N_DECODE}" \
    --base-grpc-port "${MOCK_BASE_GRPC_PORT}" \
    --performance "${PERF_CONFIG_FILE}" \
    --prefill-cache-blocks "${PREFILL_CACHE_BLOCKS}" \
    --decode-cache-blocks "${DECODE_CACHE_BLOCKS}" \
    --endpoint-file "${ENDPOINT_FILE}" \
    --env-file "${FLEXLB_ENV_FILE}" \
    >"${RUN_DIR}/mock_engine.log" 2>&1 &
  MOCK_PID="$!"
  wait_for_port "127.0.0.1" "${MOCK_BASE_GRPC_PORT}" 20
  echo "  mock cluster started (pid=${MOCK_PID})"
else
  if [[ ! -f "${ENDPOINT_FILE}" ]]; then
    echo "START_MOCK=0 requires ENDPOINT_FILE at ${ENDPOINT_FILE}" >&2
    exit 1
  fi
  echo "  [skipped] mock cluster already running"
fi

# -- Parse service discovery env vars from endpoint file ------------------

FLEXLB_ENV_ARGS=()
while IFS= read -r line; do
  FLEXLB_ENV_ARGS+=("${line}")
done < <(python3 - "${ENDPOINT_FILE}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for key, value in payload["env"].items():
    print(f"{key}={value}")
PY
)

# -- Start flexlb-api master -----------------------------------------------

if [[ "${START_FLEXLB}" == "1" ]]; then
  echo ""
  echo "[2/4] Starting flexlb-api master ..."

  if [[ "$(java_major java)" -lt 21 ]]; then
    echo "Java 21 is required to build/start flexlb-api. Set JAVA21_HOME or JAVA_HOME." >&2
    exit 1
  fi

  if [[ ! -f "${FLEXLB_JAR}" ]]; then
    echo "  Building flexlb-api (mvnw) ..."
    (cd "${FLEXLB_DIR}" && ./mvnw -P"${MAVEN_PROFILES}" -pl flexlb-api -am package -DskipTests)
  fi

  env "${FLEXLB_ENV_ARGS[@]}" \
    "LOAD_BALANCE_STRATEGY=${LOAD_BALANCE_STRATEGY}" \
    "DECODE_LOAD_BALANCE_STRATEGY=${DECODE_LOAD_BALANCE_STRATEGY}" \
    "DECODE_CONCURRENCY_LIMIT=${DECODE_CONCURRENCY_LIMIT}" \
    "FLEXLB_BATCH_ALGORITHM=${FLEXLB_BATCH_ALGORITHM}" \
    "FLEXLB_BATCH_FIXED_WAIT_MS=${FLEXLB_BATCH_FIXED_WAIT_MS}" \
    "FLEXLB_BATCH_PREDICT_THRESHOLD_MS=${FLEXLB_BATCH_PREDICT_THRESHOLD_MS}" \
    "FLEXLB_BATCH_SIZE_MAX=${FLEXLB_BATCH_SIZE_MAX}" \
    "FLEXLB_BATCH_MIN_SIZE=${FLEXLB_BATCH_MIN_SIZE}" \
    "FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES=${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES}" \
    "HYSTERESIS_BIAS_PERCENT=${HYSTERESIS_BIAS_PERCENT}" \
    "MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE}" \
    "PREFILL_QUEUE_SIZE_THRESHOLD=${PREFILL_QUEUE_SIZE_THRESHOLD}" \
    "DEFAULT_SCHEDULE_MODE=${DEFAULT_SCHEDULE_MODE}" \
    "COST_SLO_MS=${COST_SLO_MS}" \
    "STRATEGY_CONFIGS=${STRATEGY_CONFIGS}" \
    "OTEL_TRACE_SKIP_PATTERN=${OTEL_TRACE_SKIP_PATTERN}" \
    "OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT}" \
    "HIPPO_ROLE=${HIPPO_ROLE}" \
    "FLEXLB_EXPECT_FETCH_RESPONSE=true" \
    "FLEXLB_BATCH_ENABLED=${FLEXLB_BATCH_ENABLED:-true}" \
    "ENABLE_QUEUEING=${ENABLE_QUEUEING:-false}" \
    java "${JAVA_MODULE_OPTS[@]}" -jar "${FLEXLB_JAR}" \
    --server.port="${FLEXLB_HTTP_PORT}" \
    --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
    --spring.profiles.active="${SPRING_PROFILE:-default}" \
    >"${RUN_DIR}/flexlb.log" 2>&1 &
  FLEXLB_PID="$!"
  echo "  master starting (pid=${FLEXLB_PID}), waiting for port ${FLEXLB_HTTP_PORT} ..."
  wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60
  echo "  master started"
else
  echo "  [skipped] flexlb-api master already running"
fi

# -- Run cancel smoke tests ------------------------------------------------

echo ""
echo "[3/4] Running cancel smoke tests ..."
echo ""

PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/cancel_smoke.py" \
  --master-ip 127.0.0.1 \
  --master-http-port "${FLEXLB_HTTP_PORT}" \
  --schedule-mode "${SCHEDULE_MODE}" \
  2>&1 | tee "${RUN_DIR}/cancel_smoke.stdout"

SMOKE_EXIT="${PIPESTATUS[0]}"

# -- Collect results -------------------------------------------------------

echo ""
echo "[4/4] Results:"
echo "  exit_code=${SMOKE_EXIT}"
echo "  stdout=${RUN_DIR}/cancel_smoke.stdout"
echo "  mock_log=${RUN_DIR}/mock_engine.log"
echo "  flexlb_log=${RUN_DIR}/flexlb.log"
echo ""

exit "${SMOKE_EXIT}"

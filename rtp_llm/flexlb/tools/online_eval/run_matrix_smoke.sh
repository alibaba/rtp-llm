#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# run_matrix_smoke.sh — Matrix orchestration for FlexLB smoke tests.
#
# Runs three test suites (cancel, scheduling, anomaly) across three
# path/algorithm configurations (batch+fixed_window, direct, queue)
# against a single mock engine cluster (2P + 4D).
#
# Flow:
#   1. Start mock_engine_cluster once (reused across all groups)
#   2. For each group: set env → start master → run 3 suites → stop master
#   3. Summarise pass/fail per group
#   4. cleanup (stop mock cluster)
#
# Usage:
#   bash run_matrix_smoke.sh
#   START_MOCK=0 ENDPOINT_FILE=... bash run_matrix_smoke.sh  # reuse cluster
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${FLEXLB_DIR}/../.." && pwd)"

# -- Configurable parameters ------------------------------------------------

RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run}"
RUN_ID="${RUN_ID:-matrix_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
N_PREFILL="${N_PREFILL:-2}"
N_DECODE="${N_DECODE:-4}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-55151}"
MOCK_HTTP_PORT=$((MOCK_BASE_GRPC_PORT - 1))
PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS:-6000}"
DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS:-3000}"
FLEXLB_HTTP_PORT="${FLEXLB_HTTP_PORT:-18080}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-18081}"
FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"
START_MOCK="${START_MOCK:-1}"

# -- Common FlexLB config (constant across all groups) ---------------------

DECODE_LOAD_BALANCE_STRATEGY="${DECODE_LOAD_BALANCE_STRATEGY:-COST_BASED_DECODE}"
DECODE_CONCURRENCY_LIMIT="${DECODE_CONCURRENCY_LIMIT:-132}"
FLEXLB_BATCH_ALGORITHM="${FLEXLB_BATCH_ALGORITHM:-fixed_window}"
FLEXLB_BATCH_FIXED_WAIT_MS="${FLEXLB_BATCH_FIXED_WAIT_MS:-10}"
FLEXLB_BATCH_PREDICT_THRESHOLD_MS="${FLEXLB_BATCH_PREDICT_THRESHOLD_MS:-550}"
FLEXLB_BATCH_SIZE_MAX="${FLEXLB_BATCH_SIZE_MAX:-32}"
FLEXLB_BATCH_MIN_SIZE="${FLEXLB_BATCH_MIN_SIZE:-1}"
FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES:-4}"
# Hysteresis bias disabled for scheduling tests — ensures deterministic
# routing so distribution assertions are reliable.
HYSTERESIS_BIAS_PERCENT="${HYSTERESIS_BIAS_PERCENT:-0}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-5000}"
PREFILL_QUEUE_SIZE_THRESHOLD="${PREFILL_QUEUE_SIZE_THRESHOLD:-100000}"
COST_SLO_MS="${COST_SLO_MS:-30000}"
COST_HOTSPOT_MULTIPLIER="${COST_HOTSPOT_MULTIPLIER:-1.5}"
STRATEGY_CONFIGS='{}'
OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN:-.*}"
OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-none}"
HIPPO_ROLE="${HIPPO_ROLE:-flexlb_matrix_smoke_master}"

# -- Internal state --------------------------------------------------------

MOCK_PID=""
FLEXLB_PID=""
FLEXLB_ENV_ARGS=()

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

# -- Setup -----------------------------------------------------------------

mkdir -p "${RUN_DIR}"
echo "run_dir=${RUN_DIR}"

ENDPOINT_FILE="${RUN_DIR}/endpoints.json"
PERF_CONFIG_FILE="${RUN_DIR}/perf.json"
cat > "${PERF_CONFIG_FILE}" <<'JSON'
{
  "block_size": 1024,
  "sleep_scale": 1.0,
  "prefill": { "fixed_ms": 100.0, "scale": 1.0 },
  "decode": {
    "scale": 1.0,
    "step_ms_by_batch": [
      [1, 20.0], [2, 22.0], [4, 25.0], [8, 28.0],
      [16, 30.0], [32, 35.0], [64, 40.0], [128, 45.0], [256, 50.0]
    ]
  }
}
JSON

# -- Start mock engine cluster (once, reused across all groups) ------------

if [[ "${START_MOCK}" == "1" ]]; then
  echo ""
  echo "[1/3] Starting mock engine cluster (${N_PREFILL}P + ${N_DECODE}D) ..."
  PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/mock_engine_cluster.py" \
    --n-prefill "${N_PREFILL}" \
    --n-decode "${N_DECODE}" \
    --base-grpc-port "${MOCK_BASE_GRPC_PORT}" \
    --performance "${PERF_CONFIG_FILE}" \
    --prefill-cache-blocks "${PREFILL_CACHE_BLOCKS}" \
    --decode-cache-blocks "${DECODE_CACHE_BLOCKS}" \
    --endpoint-file "${ENDPOINT_FILE}" \
    --env-file "${RUN_DIR}/flexlb_env.txt" \
    >"${RUN_DIR}/mock_engine.log" 2>&1 &
  MOCK_PID="$!"
  wait_for_port "127.0.0.1" "${MOCK_BASE_GRPC_PORT}" 20
  echo "  mock cluster started (pid=${MOCK_PID}, http=${MOCK_HTTP_PORT})"
else
  if [[ ! -f "${ENDPOINT_FILE}" ]]; then
    echo "START_MOCK=0 requires ENDPOINT_FILE at ${ENDPOINT_FILE}" >&2
    exit 1
  fi
  echo "  [skipped] mock cluster already running (using ${ENDPOINT_FILE})"
fi

# Parse service-discovery env vars from endpoint file
while IFS= read -r line; do
  FLEXLB_ENV_ARGS+=("${line}")
done < <(python3 - "${ENDPOINT_FILE}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for key, value in payload["env"].items():
    print(f"{key}={value}")
PY
)

# Build flexlb-api if needed
if [[ ! -f "${FLEXLB_JAR}" ]]; then
  echo "  Building flexlb-api (mvnw) ..."
  (cd "${FLEXLB_DIR}" && ./mvnw -P"${MAVEN_PROFILES}" -pl flexlb-api -am package -DskipTests)
fi

# -- Group configuration ----------------------------------------------------

# Sets group-specific variables: LOAD_BALANCE_STRATEGY, FLEXLB_BATCH_ENABLED,
# ENABLE_QUEUEING, SCHEDULE_MODE, DEFAULT_SCHEDULE_MODE.
set_group_config() {
  case "$1" in
    batch)
      LOAD_BALANCE_STRATEGY="COST_BASED_PREFILL"
      FLEXLB_BATCH_ENABLED="true"
      ENABLE_QUEUEING="false"
      SCHEDULE_MODE="batch"
      DEFAULT_SCHEDULE_MODE="BATCH"
      TEST_RID_BASES=(10000 20000 30000)
      ;;
    direct)
      LOAD_BALANCE_STRATEGY="SHORTEST_TTFT"
      FLEXLB_BATCH_ENABLED="false"
      ENABLE_QUEUEING="false"
      SCHEDULE_MODE="direct"
      DEFAULT_SCHEDULE_MODE="DIRECT"
      TEST_RID_BASES=(40000 50000 60000)
      ;;
    queue)
      LOAD_BALANCE_STRATEGY="SHORTEST_TTFT"
      FLEXLB_BATCH_ENABLED="false"
      ENABLE_QUEUEING="true"
      SCHEDULE_MODE="queue"
      DEFAULT_SCHEDULE_MODE="QUEUE"
      TEST_RID_BASES=(70000 80000 90000)
      ;;
    *)
      echo "Unknown group: $1" >&2
      exit 1
      ;;
  esac
}

start_master() {
  local group="$1"
  local group_dir="${RUN_DIR}/${group}"
  mkdir -p "${group_dir}"
  echo "  starting master (group=${group}, mode=${SCHEDULE_MODE}) ..."
  env ${FLEXLB_ENV_ARGS[@]+"${FLEXLB_ENV_ARGS[@]}"} \
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
    "COST_HOTSPOT_MULTIPLIER=${COST_HOTSPOT_MULTIPLIER}" \
    "STRATEGY_CONFIGS=${STRATEGY_CONFIGS}" \
    "OTEL_TRACE_SKIP_PATTERN=${OTEL_TRACE_SKIP_PATTERN}" \
    "OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT}" \
    "HIPPO_ROLE=${HIPPO_ROLE}" \
    "FLEXLB_EXPECT_FETCH_RESPONSE=true" \
    "FLEXLB_BATCH_ENABLED=${FLEXLB_BATCH_ENABLED}" \
    "ENABLE_QUEUEING=${ENABLE_QUEUEING}" \
    java "${JAVA_MODULE_OPTS[@]}" -jar "${FLEXLB_JAR}" \
    --server.port="${FLEXLB_HTTP_PORT}" \
    --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
    --spring.profiles.active="${SPRING_PROFILE:-default}" \
    >"${group_dir}/flexlb.log" 2>&1 &
  FLEXLB_PID="$!"
  wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60
  echo "  master started (pid=${FLEXLB_PID})"
}

stop_master() {
  if [[ -n "${FLEXLB_PID}" ]]; then
    echo "  stopping master (pid=${FLEXLB_PID}) ..."
    kill "${FLEXLB_PID}" >/dev/null 2>&1 || true
    wait "${FLEXLB_PID}" 2>/dev/null || true
    FLEXLB_PID=""
    sleep 2
  fi
}

# run_test_suite <name> <script> <request_id_base> <group>
run_test_suite() {
  local name="$1" script="$2" rid_base="$3" group="$4"
  local group_dir="${RUN_DIR}/${group}"
  echo ""
  echo "  --- ${name} (rid_base=${rid_base}) ---"
  local cmd_args=(
    --master-ip 127.0.0.1
    --master-http-port "${FLEXLB_HTTP_PORT}"
    --flexlb-http-port "${FLEXLB_HTTP_PORT}"
    --schedule-mode "${SCHEDULE_MODE}"
    --request-id-base "${rid_base}"
  )
  # scheduling_smoke.py and anomaly_smoke.py need --mock-http-port;
  # cancel_smoke.py does not accept it.
  if [[ "${script}" != "cancel_smoke.py" ]]; then
    cmd_args+=(--mock-http-port "${MOCK_HTTP_PORT}")
  fi
  set +e
  PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/${script}" \
    "${cmd_args[@]}" 2>&1 | tee "${group_dir}/${name}.stdout"
  exit_code=${PIPESTATUS[0]}
  set -e
  if [[ "${exit_code}" -eq 0 ]]; then
    echo "  ${name}: PASS"
  else
    echo "  ${name}: FAIL (exit=${exit_code})"
  fi
  return "${exit_code}"
}

# -- Main loop: 3 groups x 3 test suites ------------------------------------

GROUP_NAMES=("batch" "direct" "queue")
TEST_NAMES=("cancel_smoke" "scheduling_smoke" "anomaly_smoke")
TEST_SCRIPTS=("cancel_smoke.py" "scheduling_smoke.py" "anomaly_smoke.py")
TOTAL_PASS=0
TOTAL_FAIL=0
GROUP_RESULTS=()

echo ""
echo "[2/3] Running matrix smoke tests ..."

for group in "${GROUP_NAMES[@]}"; do
  echo ""
  echo "=========================================="
  echo "  Group: ${group}"
  echo "=========================================="

  set_group_config "${group}"
  start_master "${group}"

  group_pass=0
  group_fail=0
  for i in "${!TEST_NAMES[@]}"; do
    if run_test_suite "${TEST_NAMES[$i]}" "${TEST_SCRIPTS[$i]}" "${TEST_RID_BASES[$i]}" "${group}"; then
      group_pass=$((group_pass + 1))
    else
      group_fail=$((group_fail + 1))
    fi
  done

  TOTAL_PASS=$((TOTAL_PASS + group_pass))
  TOTAL_FAIL=$((TOTAL_FAIL + group_fail))
  GROUP_RESULTS+=("${group}: ${group_pass}/3 passed")

  stop_master
done

# -- Summary ---------------------------------------------------------------

echo ""
echo "[3/3] Matrix Summary:"
echo "=========================================="
for result in ${GROUP_RESULTS[@]+"${GROUP_RESULTS[@]}"}; do
  echo "  ${result}"
done
echo ""
echo "  Total: ${TOTAL_PASS} passed, ${TOTAL_FAIL} failed (out of $((TOTAL_PASS + TOTAL_FAIL)) suites)"
echo "  logs: ${RUN_DIR}/<group>/<test>.stdout"
echo "=========================================="

if [[ "${TOTAL_FAIL}" -gt 0 ]]; then
  exit 1
fi

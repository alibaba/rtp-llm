#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# run_smoke.sh — Unified smoke test entry point.
#
# Replaces: run_matrix_smoke.sh + run_cancel_smoke.sh
#
# Runs functional smoke tests (cancel, scheduling, anomaly, resilience)
# across three path configurations (batch, direct, queue) against a
# single mock engine cluster.
#
# Usage:
#   bash run_smoke.sh                              # all groups, all suites
#   bash run_smoke.sh --group batch                 # batch path only
#   bash run_smoke.sh --suite cancel                # cancel suite only
#   bash run_smoke.sh --group direct --suite all    # direct path, all suites
#
# Structural selectors (--group, --suite) do NOT override env vars.
# All configuration is via environment variables (one set, no multi-layer).
# ===========================================================================

# -- Source common functions ------------------------------------------------

flexlb_init_paths
source "${SCRIPT_DIR}/common.sh"
assert_not_root
setup_java21
detect_python

# -- Parse structural selectors ---------------------------------------------

SMOKE_GROUP="all"
SMOKE_SUITE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) SMOKE_GROUP="$2"; shift 2;;
    --suite) SMOKE_SUITE="$2"; shift 2;;
    -h|--help)
      echo "Usage: bash run_smoke.sh [--group batch|direct|queue|all] [--suite cancel|scheduling|anomaly|resilience|all]"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1;;
  esac
done

# -- Configuration (single layer: env var with fallback) --------------------

RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run}"
RUN_ID="${RUN_ID:-smoke_$(date +%Y%m%d_%H%M%S)}"
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
FLEXLB_FAIL_ON_CONCURRENT_TEST="${FLEXLB_FAIL_ON_CONCURRENT_TEST:-1}"

# Smoke-specific scheduling config (shared across all groups)
DECODE_LOAD_BALANCE_STRATEGY="${DECODE_LOAD_BALANCE_STRATEGY:-COST_BASED_DECODE}"
DECODE_CONCURRENCY_LIMIT="${DECODE_CONCURRENCY_LIMIT:-132}"
FLEXLB_BATCH_ALGORITHM="${FLEXLB_BATCH_ALGORITHM:-fixed_window}"
FLEXLB_BATCH_FIXED_WAIT_MS="${FLEXLB_BATCH_FIXED_WAIT_MS:-10}"
FLEXLB_BATCH_PREDICT_THRESHOLD_MS="${FLEXLB_BATCH_PREDICT_THRESHOLD_MS:-550}"
FLEXLB_BATCH_SIZE_MAX="${FLEXLB_BATCH_SIZE_MAX:-32}"
FLEXLB_BATCH_MIN_SIZE="${FLEXLB_BATCH_MIN_SIZE:-1}"
FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES:-4}"
HYSTERESIS_BIAS_PERCENT="${HYSTERESIS_BIAS_PERCENT:-0}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-5000}"
PREFILL_QUEUE_SIZE_THRESHOLD="${PREFILL_QUEUE_SIZE_THRESHOLD:-100000}"
COST_SLO_MS="${COST_SLO_MS:-30000}"
COST_HOTSPOT_MULTIPLIER="${COST_HOTSPOT_MULTIPLIER:-1.5}"
STRATEGY_CONFIGS="${STRATEGY_CONFIGS:-{}}"
OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN:-.*}"
OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-none}"
HIPPO_ROLE="${HIPPO_ROLE:-flexlb_smoke_master}"

# -- Print effective configuration ------------------------------------------

print_env_summary \
  RUN_DIR N_PREFILL N_DECODE MOCK_BASE_GRPC_PORT MOCK_HTTP_PORT \
  FLEXLB_HTTP_PORT FLEXLB_MANAGEMENT_PORT FLEXLB_JAR START_MOCK \
  SMOKE_GROUP SMOKE_SUITE \
  DECODE_LOAD_BALANCE_STRATEGY FLEXLB_BATCH_ALGORITHM \
  FLEXLB_BATCH_FIXED_WAIT_MS FLEXLB_BATCH_PREDICT_THRESHOLD_MS \
  FLEXLB_BATCH_SIZE_MAX FLEXLB_BATCH_MIN_SIZE \
  FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES HYSTERESIS_BIAS_PERCENT \
  MAX_QUEUE_SIZE COST_SLO_MS COST_HOTSPOT_MULTIPLIER

# -- Internal state ---------------------------------------------------------

MOCK_PID=""
FLEXLB_PID=""
FLEXLB_ENV_ARGS=()

# -- Trap -------------------------------------------------------------------

trap cleanup_all EXIT

# -- Setup ------------------------------------------------------------------

mkdir -p "${RUN_DIR}"
echo "run_dir=${RUN_DIR}"

if [[ "${FLEXLB_FAIL_ON_CONCURRENT_TEST}" == "1" ]]; then
  assert_no_concurrent_flexlb_test
fi

# -- Perf config (smoke standard) ------------------------------------------

PERFORMANCE_FILE="${RUN_DIR}/perf.json"
generate_perf_config "${PERFORMANCE_FILE}" 100.0 20.0

# -- Start mock engine cluster (once, reused across all groups) -------------

ENDPOINT_FILE="${RUN_DIR}/endpoints.json"
FLEXLB_ENV_FILE="${RUN_DIR}/flexlb_env.txt"

echo ""
echo "[1/3] Starting mock engine cluster (${N_PREFILL}P + ${N_DECODE}D) ..."
start_mock_cluster "${ENDPOINT_FILE}" "${FLEXLB_ENV_FILE}" "${RUN_DIR}/mock_engine.log"
parse_endpoint_env "${ENDPOINT_FILE}"

# -- Build JAR if needed ----------------------------------------------------

build_flexlb_jar

# -- Group configuration ----------------------------------------------------

set_group_config() {
  case "$1" in
    batch)
      LOAD_BALANCE_STRATEGY="COST_BASED_PREFILL"
      FLEXLB_BATCH_ENABLED="true"
      ENABLE_QUEUEING="false"
      SCHEDULE_MODE="batch"
      DEFAULT_SCHEDULE_MODE="BATCH"
      TEST_RID_BASES=(10000 20000 30000 31000)
      ;;
    direct)
      LOAD_BALANCE_STRATEGY="SHORTEST_TTFT"
      FLEXLB_BATCH_ENABLED="false"
      ENABLE_QUEUEING="false"
      SCHEDULE_MODE="direct"
      DEFAULT_SCHEDULE_MODE="DIRECT"
      TEST_RID_BASES=(40000 50000 60000 61000)
      ;;
    queue)
      LOAD_BALANCE_STRATEGY="SHORTEST_TTFT"
      FLEXLB_BATCH_ENABLED="false"
      ENABLE_QUEUEING="true"
      SCHEDULE_MODE="queue"
      DEFAULT_SCHEDULE_MODE="QUEUE"
      TEST_RID_BASES=(70000 80000 90000 91000)
      ;;
    *)
      echo "Unknown group: $1" >&2
      exit 1
      ;;
  esac
}

# -- Master start/stop (group-specific env vars) ----------------------------

start_smoke_master() {
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

# -- Test suite runner ------------------------------------------------------

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

# -- Resolve group and suite lists ------------------------------------------

if [[ "${SMOKE_GROUP}" == "all" ]]; then
  GROUP_NAMES=("batch" "direct" "queue")
else
  GROUP_NAMES=("${SMOKE_GROUP}")
fi

# Suite name → script mapping
declare -A SUITE_SCRIPTS=(
  [cancel]="cancel_smoke.py"
  [scheduling]="scheduling_smoke.py"
  [anomaly]="anomaly_smoke.py"
  [resilience]="resilience_smoke.py"
)

if [[ "${SMOKE_SUITE}" == "all" ]]; then
  SUITE_NAMES=("cancel" "scheduling" "anomaly" "resilience")
else
  SUITE_NAMES=("${SMOKE_SUITE}")
fi

# -- Main loop: groups × suites --------------------------------------------

TOTAL_PASS=0
TOTAL_FAIL=0
GROUP_RESULTS=()

echo ""
echo "[2/3] Running smoke tests ..."
echo "  Groups: ${GROUP_NAMES[*]}"
echo "  Suites: ${SUITE_NAMES[*]}"

for group in "${GROUP_NAMES[@]}"; do
  echo ""
  echo "=========================================="
  echo "  Group: ${group}"
  echo "=========================================="

  set_group_config "${group}"
  start_smoke_master "${group}"

  group_pass=0
  group_fail=0
  suite_idx=0
  for suite in "${SUITE_NAMES[@]}"; do
    script="${SUITE_SCRIPTS[${suite}]}"
    if run_test_suite "${suite}_smoke" "${script}" "${TEST_RID_BASES[${suite_idx}]}" "${group}"; then
      group_pass=$((group_pass + 1))
    else
      group_fail=$((group_fail + 1))
    fi
    suite_idx=$((suite_idx + 1))
  done

  TOTAL_PASS=$((TOTAL_PASS + group_pass))
  TOTAL_FAIL=$((TOTAL_FAIL + group_fail))
  GROUP_RESULTS+=("${group}: ${group_pass}/${#SUITE_NAMES[@]} passed")

  stop_master
done

# -- Summary ---------------------------------------------------------------

echo ""
echo "[3/3] Smoke Test Summary:"
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

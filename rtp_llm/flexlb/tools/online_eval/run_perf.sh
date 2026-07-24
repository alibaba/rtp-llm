#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# run_perf.sh — Unified performance test entry point.
#
# Replaces: run_online_eval.sh + run_burst_test.sh + run_stability_test.sh
#           + run_batcher_experiments.sh
#
# Modes:
#   online     — single performance run (delegates to run_online_eval.sh)
#   burst      — multi-speed burst traffic stress test
#   stability  — multi-round stability test with GC/HeapDump monitoring
#   batcher    — batcher parameter optimization experiments
#
# Usage:
#   bash run_perf.sh                              # default: online mode
#   bash run_perf.sh --mode online                # single run
#   bash run_perf.sh --mode burst                 # burst traffic test
#   bash run_perf.sh --mode stability             # stability test
#   bash run_perf.sh --mode batcher               # batcher experiments
#
# Structural selector (--mode) does NOT override env vars.
# All configuration is via environment variables (one set, no multi-layer).
#
# Skill compatibility: `run_perf.sh --mode online` delegates to
# run_online_eval.sh with the current environment, maintaining backward
# compatibility with flexlb-online-eval skill.
# ===========================================================================

# -- Source common functions ------------------------------------------------

flexlb_init_paths
source "${SCRIPT_DIR}/common.sh"
assert_not_root
setup_java21
detect_python

# -- Network isolation (perf tests may need it) -----------------------------

flexlb_maybe_enter_namespace

# -- Parse structural selector ----------------------------------------------

PERF_MODE="online"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) PERF_MODE="$2"; shift 2;;
    -h|--help)
      echo "Usage: bash run_perf.sh [--mode online|burst|stability|batcher]"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1;;
  esac
done

# -- Configuration (single layer: env var with fallback) --------------------
# These env vars pass through directly to run_online_eval.sh.
# No multi-layer override — one set of env vars, printed at startup.

RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run}"
RUN_ID="${RUN_ID:-perf_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
FLEXLB_LOG_PATH="${FLEXLB_LOG_PATH:-${RUN_DIR}/flexlb_logs}"

N_PREFILL="${N_PREFILL:-2}"
N_DECODE="${N_DECODE:-4}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-61000}"
MOCK_ENGINE_IMPL="${MOCK_ENGINE_IMPL:-java}"
N_SHARDS="${N_SHARDS:-64}"
FLEXLB_HTTP_ADDR="${FLEXLB_HTTP_ADDR:-127.0.0.1:7001}"
FLEXLB_HTTP_PORT="${FLEXLB_HTTP_ADDR##*:}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-7002}"
FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"
START_FLEXLB="${START_FLEXLB:-1}"
START_MOCK="${START_MOCK:-1}"
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"
FLEXLB_FAIL_ON_CONCURRENT_TEST="${FLEXLB_FAIL_ON_CONCURRENT_TEST:-1}"

# Load client parameters
LIMIT="${LIMIT:-1000}"
DURATION_S="${DURATION_S:-0}"
REPLAY_SPEED="${REPLAY_SPEED:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-999999999}"
SCHEDULE_MODE="${SCHEDULE_MODE:-batch}"
TIMEOUT_MS="${TIMEOUT_MS:-3600000}"
SLA_TTFT_MS="${SLA_TTFT_MS:-500}"
ZERO_OUTPUT_POLICY="${ZERO_OUTPUT_POLICY:-skip}"
LOAD_CLIENT_WORKERS="${LOAD_CLIENT_WORKERS:-8}"
LOAD_CLIENT_START_DELAY_SECONDS="${LOAD_CLIENT_START_DELAY_SECONDS:-10}"

# FlexLB scheduling config
LOAD_BALANCE_STRATEGY="${LOAD_BALANCE_STRATEGY:-COST_BASED_PREFILL}"
DECODE_LOAD_BALANCE_STRATEGY="${DECODE_LOAD_BALANCE_STRATEGY:-COST_BASED_DECODE}"
DECODE_CONCURRENCY_LIMIT="${DECODE_CONCURRENCY_LIMIT:-132}"
FLEXLB_BATCH_ALGORITHM="${FLEXLB_BATCH_ALGORITHM:-fixed_window}"
FLEXLB_BATCH_FIXED_WAIT_MS="${FLEXLB_BATCH_FIXED_WAIT_MS:-10}"
FLEXLB_BATCH_PREDICT_THRESHOLD_MS="${FLEXLB_BATCH_PREDICT_THRESHOLD_MS:-550}"
FLEXLB_BATCH_SIZE_MAX="${FLEXLB_BATCH_SIZE_MAX:-32}"
FLEXLB_BATCH_MIN_SIZE="${FLEXLB_BATCH_MIN_SIZE:-8}"
FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES:--1}"
HYSTERESIS_BIAS_PERCENT="${HYSTERESIS_BIAS_PERCENT:-30}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-1000000}"
PREFILL_QUEUE_SIZE_THRESHOLD="${PREFILL_QUEUE_SIZE_THRESHOLD:-100000}"
COST_SLO_MS="${COST_SLO_MS:-1000}"
FLEXLB_BATCH_MAX_INFLIGHT="${FLEXLB_BATCH_MAX_INFLIGHT:-1000000}"
FLEXLB_BATCH_DISPATCH_POOL_SIZE="${FLEXLB_BATCH_DISPATCH_POOL_SIZE:-500}"
FLEXLB_BATCH_DISPATCH_QUEUE_SIZE="${FLEXLB_BATCH_DISPATCH_QUEUE_SIZE:-10000}"

# Observability
FLEXLB_MONITOR_ENABLED="${FLEXLB_MONITOR_ENABLED:-true}"
FLEXLB_MONITOR_MODE="${FLEXLB_MONITOR_MODE:-critical-only}"
OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN:-.*}"
OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-none}"
HIPPO_ROLE="${HIPPO_ROLE:-flexlb_eval_master}"

# JFR
JFR_FILE="${JFR_FILE:-${RUN_DIR}/flexlb_profile.jfr}"
JFR_DURATION="${JFR_DURATION:-300s}"

# Burst/stability mode defaults
SPEEDS="${SPEEDS:-}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"

# Batcher experiment defaults
BATCHER_EXPERIMENTS="${BATCHER_EXPERIMENTS:-}"

# -- Print effective configuration ------------------------------------------

print_env_summary \
  PERF_MODE RUN_DIR N_PREFILL N_DECODE MOCK_BASE_GRPC_PORT \
  FLEXLB_HTTP_ADDR FLEXLB_MANAGEMENT_PORT FLEXLB_JAR \
  START_FLEXLB START_MOCK MOCK_ENGINE_IMPL N_SHARDS \
  LIMIT DURATION_S REPLAY_SPEED MAX_CONCURRENCY SCHEDULE_MODE \
  TIMEOUT_MS SLA_TTFT_MS ZERO_OUTPUT_POLICY LOAD_CLIENT_WORKERS \
  LOAD_BALANCE_STRATEGY DECODE_LOAD_BALANCE_STRATEGY \
  FLEXLB_BATCH_ALGORITHM FLEXLB_BATCH_FIXED_WAIT_MS \
  FLEXLB_BATCH_PREDICT_THRESHOLD_MS FLEXLB_BATCH_SIZE_MAX \
  FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES DECODE_CONCURRENCY_LIMIT \
  HYSTERESIS_BIAS_PERCENT MAX_QUEUE_SIZE COST_SLO_MS \
  FLEXLB_MONITOR_ENABLED FLEXLB_MONITOR_MODE \
  OTEL_TRACE_SKIP_PATTERN OTEL_EXPORTER_OTLP_ENDPOINT HIPPO_ROLE \
  JFR_FILE JFR_DURATION SPEEDS MONITOR_INTERVAL

# -- Concurrency guard ------------------------------------------------------

if [[ "${FLEXLB_FAIL_ON_CONCURRENT_TEST}" == "1" ]]; then
  assert_no_concurrent_flexlb_test
fi

# ===========================================================================
# Core executor: delegates to run_online_eval.sh
# All env vars pass through directly (no override, no multi-layer).
# ===========================================================================

run_online_eval_core() {
  local eval_run_id="${1:-${RUN_ID}}"
  echo ""
  echo ">>> Running online eval (RUN_ID=${eval_run_id}) ..."
  set +e
  env \
    RUN_ROOT="${RUN_ROOT}" \
    RUN_ID="${eval_run_id}" \
    N_PREFILL="${N_PREFILL}" \
    N_DECODE="${N_DECODE}" \
    MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT}" \
    MOCK_ENGINE_IMPL="${MOCK_ENGINE_IMPL}" \
    N_SHARDS="${N_SHARDS}" \
    FLEXLB_HTTP_ADDR="${FLEXLB_HTTP_ADDR}" \
    FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT}" \
    FLEXLB_JAR="${FLEXLB_JAR}" \
    START_FLEXLB="${START_FLEXLB}" \
    START_MOCK="${START_MOCK}" \
    MAVEN_PROFILES="${MAVEN_PROFILES}" \
    LIMIT="${LIMIT}" \
    DURATION_S="${DURATION_S}" \
    REPLAY_SPEED="${REPLAY_SPEED}" \
    MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
    SCHEDULE_MODE="${SCHEDULE_MODE}" \
    TIMEOUT_MS="${TIMEOUT_MS}" \
    SLA_TTFT_MS="${SLA_TTFT_MS}" \
    ZERO_OUTPUT_POLICY="${ZERO_OUTPUT_POLICY}" \
    LOAD_CLIENT_WORKERS="${LOAD_CLIENT_WORKERS}" \
    LOAD_CLIENT_START_DELAY_SECONDS="${LOAD_CLIENT_START_DELAY_SECONDS}" \
    LOAD_BALANCE_STRATEGY="${LOAD_BALANCE_STRATEGY}" \
    DECODE_LOAD_BALANCE_STRATEGY="${DECODE_LOAD_BALANCE_STRATEGY}" \
    DECODE_CONCURRENCY_LIMIT="${DECODE_CONCURRENCY_LIMIT}" \
    FLEXLB_BATCH_ALGORITHM="${FLEXLB_BATCH_ALGORITHM}" \
    FLEXLB_BATCH_FIXED_WAIT_MS="${FLEXLB_BATCH_FIXED_WAIT_MS}" \
    FLEXLB_BATCH_PREDICT_THRESHOLD_MS="${FLEXLB_BATCH_PREDICT_THRESHOLD_MS}" \
    FLEXLB_BATCH_SIZE_MAX="${FLEXLB_BATCH_SIZE_MAX}" \
    FLEXLB_BATCH_MIN_SIZE="${FLEXLB_BATCH_MIN_SIZE}" \
    FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES}" \
    FLEXLB_BATCH_MAX_INFLIGHT="${FLEXLB_BATCH_MAX_INFLIGHT}" \
    FLEXLB_BATCH_DISPATCH_POOL_SIZE="${FLEXLB_BATCH_DISPATCH_POOL_SIZE}" \
    FLEXLB_BATCH_DISPATCH_QUEUE_SIZE="${FLEXLB_BATCH_DISPATCH_QUEUE_SIZE}" \
    HYSTERESIS_BIAS_PERCENT="${HYSTERESIS_BIAS_PERCENT}" \
    MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE}" \
    PREFILL_QUEUE_SIZE_THRESHOLD="${PREFILL_QUEUE_SIZE_THRESHOLD}" \
    COST_SLO_MS="${COST_SLO_MS}" \
    FLEXLB_MONITOR_ENABLED="${FLEXLB_MONITOR_ENABLED}" \
    FLEXLB_MONITOR_MODE="${FLEXLB_MONITOR_MODE}" \
    OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN}" \
    OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT}" \
    HIPPO_ROLE="${HIPPO_ROLE}" \
    JFR_FILE="${JFR_FILE}" \
    JFR_DURATION="${JFR_DURATION}" \
    FLEXLB_LOG_PATH="${FLEXLB_LOG_PATH}" \
    ${JAVA_TOOL_OPTIONS:+JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS}"} \
    bash "${SCRIPT_DIR}/run_online_eval.sh" 2>&1 | tee "${RUN_ROOT}/${eval_run_id}.stdout"
  local eval_exit=$?
  set -e
  if [[ ${eval_exit} -ne 0 ]]; then
    echo "    WARNING: run_online_eval.sh exited with code ${eval_exit}"
  fi
  return ${eval_exit}
}

# ===========================================================================
# Mode: online — single performance run
# ===========================================================================

mode_online() {
  echo "============================================"
  echo "  Mode: online (single performance run)"
  echo "============================================"
  mkdir -p "${RUN_DIR}"
  run_online_eval_core "${RUN_ID}"
  echo ""
  echo "  summary=${RUN_DIR}/load_client/summary.json"
  echo "  report=${RUN_DIR}/load_client/report.md"
}

# ===========================================================================
# Mode: burst — multi-speed burst traffic stress test
# ===========================================================================

mode_burst() {
  local speeds=()
  if [[ -n "${SPEEDS}" ]]; then
    read -r -a speeds <<< "${SPEEDS}"
  else
    speeds=(5 10 15 20 30 50)
  fi

  local burst_dir="${RUN_ROOT}/burst_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${burst_dir}"
  local round_dirs=()
  local monitor_pid=""

  cleanup_burst_monitor() {
    if [[ -n "${monitor_pid}" ]]; then
      kill "${monitor_pid}" 2>/dev/null || true
      wait "${monitor_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup_burst_monitor EXIT

  echo "============================================"
  echo "  Mode: burst (multi-speed stress test)"
  echo "  Speeds: ${speeds[*]}"
  echo "============================================"

  for speed in "${speeds[@]}"; do
    local round_id="burst_${speed}x_$(date +%Y%m%d_%H%M%S)"
    local round_dir="${RUN_ROOT}/${round_id}"
    mkdir -p "${round_dir}"
    round_dirs+=("${round_dir}")

    echo ">>> Speed ${speed}x starting (output: ${round_dir})"

    # Start stability monitor
    python3 "${SCRIPT_DIR}/stability_monitor.py" \
      --flexlb-http-addr "${FLEXLB_HTTP_ADDR}" \
      --management-port "${FLEXLB_MANAGEMENT_PORT}" \
      --mock-http-port "$((MOCK_BASE_GRPC_PORT - 1))" \
      --interval "${MONITOR_INTERVAL}" \
      --output "${round_dir}/monitor.jsonl" &
    monitor_pid=$!

    # Run online eval with this speed
    REPLAY_SPEED="${speed}" run_online_eval_core "${round_id}"

    # Stop monitor
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""

    echo "<<< Speed ${speed}x done"
    if [[ "${speed}" != "${speeds[-1]}" ]]; then
      echo "    Waiting 3s for port release..."
      sleep 3
    fi
  done

  # Generate analysis report
  echo "============================================"
  echo "  Generating burst traffic analysis report..."
  echo "============================================"
  local report_path="${burst_dir}/burst_traffic_report.md"
  python3 "${SCRIPT_DIR}/analyze_burst_results.py" \
    --run-dirs "${round_dirs[@]}" \
    --speeds ${speeds[*]} \
    --sla-ttft-ms "${SLA_TTFT_MS}" \
    --output "${report_path}" || echo "  WARNING: analysis failed"

  echo ""
  echo "  Report: ${report_path}"
  echo "  Per-speed outputs:"
  for dir in "${round_dirs[@]}"; do
    echo "    - ${dir}/"
  done
}

# ===========================================================================
# Mode: stability — multi-round stability test with GC/HeapDump
# ===========================================================================

mode_stability() {
  local speeds=()
  if [[ -n "${SPEEDS}" ]]; then
    read -r -a speeds <<< "${SPEEDS}"
  else
    speeds=(10 20 50)
  fi

  local stability_dir="${RUN_ROOT}/stability_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${stability_dir}"
  local round_dirs=()
  local monitor_pid=""

  cleanup_stability_monitor() {
    if [[ -n "${monitor_pid}" ]]; then
      kill "${monitor_pid}" 2>/dev/null || true
      wait "${monitor_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup_stability_monitor EXIT

  echo "============================================"
  echo "  Mode: stability (multi-round stability test)"
  echo "  Speeds: ${speeds[*]}"
  echo "============================================"

  for speed in "${speeds[@]}"; do
    local round_id="stability_${speed}x_$(date +%Y%m%d_%H%M%S)"
    local round_dir="${RUN_ROOT}/${round_id}"
    mkdir -p "${round_dir}"
    round_dirs+=("${round_dir}")

    echo ">>> Round ${speed}x starting (output: ${round_dir})"

    # Start stability monitor
    python3 "${SCRIPT_DIR}/stability_monitor.py" \
      --flexlb-http-addr "${FLEXLB_HTTP_ADDR}" \
      --management-port "${FLEXLB_MANAGEMENT_PORT}" \
      --mock-http-port "$((MOCK_BASE_GRPC_PORT + N_PREFILL + N_DECODE + 100 + N_SHARDS))" \
      --interval "${MONITOR_INTERVAL}" \
      --output "${round_dir}/monitor.jsonl" &
    monitor_pid=$!

    # Inject GC logging + HeapDump via JAVA_TOOL_OPTIONS
    export JAVA_TOOL_OPTIONS="-Xlog:gc*:${round_dir}/gc.log:time,uptime,level,tags:filecount=1,filesize=50m -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=${round_dir}/"

    # Run online eval with this speed
    REPLAY_SPEED="${speed}" run_online_eval_core "${round_id}"

    # Stop monitor
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""

    unset JAVA_TOOL_OPTIONS

    echo "<<< Round ${speed}x done"
    if [[ "${speed}" != "${speeds[-1]}" ]]; then
      echo "    Waiting 3s for port release..."
      sleep 3
    fi
  done

  # Generate stability report
  echo "============================================"
  echo "  Generating stability comparison report..."
  echo "============================================"
  local report_path="${stability_dir}/stability_report.md"
  python3 "${SCRIPT_DIR}/generate_stability_report.py" \
    --run-dirs "${round_dirs[@]}" \
    --sla-ttft-ms "${SLA_TTFT_MS}" \
    --output "${report_path}" || echo "  WARNING: report generation failed"

  echo ""
  echo "  Report: ${report_path}"
  echo "  Per-round outputs:"
  for dir in "${round_dirs[@]}"; do
    echo "    - ${dir}/"
  done
}

# ===========================================================================
# Mode: batcher — batcher parameter optimization experiments
# ===========================================================================

mode_batcher() {
  # Default experiments: name:max_inflight:wait_ms:speed
  local experiments=()
  if [[ -n "${BATCHER_EXPERIMENTS}" ]]; then
    read -r -a experiments <<< "${BATCHER_EXPERIMENTS}"
  else
    experiments=(
      "exp_a1_b4_w220_s50:4:220:50"
      "exp_a2_b8_w220_s50:8:220:50"
      "exp_b1_b2_w100_s50:2:100:50"
      "exp_b2_b2_w50_s50:2:50:50"
      "exp_c1_b4_w100_s50:4:100:50"
      "exp_c2_b8_w100_s50:8:100:50"
      "exp_d1_b4_w100_s10:4:100:10"
      "exp_d2_b4_w100_s20:4:100:20"
    )
  fi

  echo "============================================"
  echo "  Mode: batcher (parameter optimization)"
  echo "  Experiments: ${#experiments[@]}"
  echo "============================================"

  for exp_spec in "${experiments[@]}"; do
    local name max_inflight wait_ms speed
    IFS=':' read -r name max_inflight wait_ms speed <<< "${exp_spec}"

    echo ""
    echo "=============================================="
    echo "  EXPERIMENT: ${name}"
    echo "    MAX_INFLIGHT_BATCHES=${max_inflight}"
    echo "    WAIT_MS=${wait_ms}"
    echo "    REPLAY_SPEED=${speed}"
    echo "=============================================="

    FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${max_inflight}" \
    FLEXLB_BATCH_FIXED_WAIT_MS="${wait_ms}" \
    REPLAY_SPEED="${speed}" \
    LIMIT="${LIMIT:-0}" \
    ZERO_OUTPUT_POLICY="${ZERO_OUTPUT_POLICY:-one}" \
    MAX_CONCURRENCY="${MAX_CONCURRENCY:-16384}" \
      run_online_eval_core "${name}"

    echo ""
    echo "  EXPERIMENT ${name} COMPLETE"
    echo "  Summary: ${RUN_ROOT}/${name}/load_client/summary.json"

    # Clean up ports between experiments
    echo "  Cleaning up ports..."
    sleep 5
  done

  echo ""
  echo "=============================================="
  echo "  ALL BATCHER EXPERIMENTS COMPLETE"
  echo "=============================================="
  echo ""
  echo "Summary files:"
  for exp_spec in "${experiments[@]}"; do
    local name
    IFS=':' read -r name _ _ _ <<< "${exp_spec}"
    local summary="${RUN_ROOT}/${name}/load_client/summary.json"
    if [[ -f "${summary}" ]]; then
      echo "  ${summary}"
    else
      echo "  MISSING: ${summary}"
    fi
  done
}

# ===========================================================================
# Dispatch
# ===========================================================================

case "${PERF_MODE}" in
  online)     mode_online ;;
  burst)      mode_burst ;;
  stability)  mode_stability ;;
  batcher)    mode_batcher ;;
  *)
    echo "ERROR: unknown mode '${PERF_MODE}' (expected: online|burst|stability|batcher)" >&2
    exit 1
    ;;
esac

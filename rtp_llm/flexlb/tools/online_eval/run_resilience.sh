#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# run_resilience.sh — Unified resilience test entry point.
#
# Replaces: flexlb_behavior_test.sh + engine_disconnect_ttft_test.sh
#           + engine_kill_restart_test.sh + master_kill_restart_test.sh
#           + master_recovery_ttft_test.sh
#
# Scenarios:
#   behavior            — 4 sub-scenarios: TTL cleanup, gRPC recovery,
#                         quota blocking, calibrate cleanup
#   disconnect_ttft     — engine stop/restart via HTTP API, TTFT observation
#   engine_kill_restart — kill -9 single victim engine, restart, verify
#   master_kill_restart — kill -9 master, restart, verify (5 assertions)
#   master_recovery     — kill master + fallback + restart + TTFT recovery
#
# Usage:
#   bash run_resilience.sh                                    # default: behavior
#   bash run_resilience.sh --scenario behavior                # all 4 sub-scenarios
#   bash run_resilience.sh --scenario disconnect_ttft         # disconnect TTFT
#   bash run_resilience.sh --scenario engine_kill_restart     # engine kill/restart
#   bash run_resilience.sh --scenario master_kill_restart     # master kill/restart
#   bash run_resilience.sh --scenario master_recovery         # master recovery TTFT
#
# Structural selector (--scenario) does NOT override env vars.
# All configuration is via environment variables (one set, no multi-layer).
#
# Delegation: each scenario handler delegates to the corresponding original
# script (which still exists in this directory). Env vars pass through
# directly via process inheritance — no multi-layer override.
# ===========================================================================

# -- Source common functions ------------------------------------------------

flexlb_init_paths
source "${SCRIPT_DIR}/common.sh"
assert_not_root
setup_java21
detect_python

# -- Parse structural selector ----------------------------------------------

RESILIENCE_SCENARIO="behavior"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) RESILIENCE_SCENARIO="$2"; shift 2;;
    -h|--help)
      echo "Usage: bash run_resilience.sh [--scenario behavior|disconnect_ttft|engine_kill_restart|master_kill_restart|master_recovery]"
      echo ""
      echo "Scenarios:"
      echo "  behavior            — TTL cleanup, gRPC recovery, quota blocking, calibrate"
      echo "  disconnect_ttft     — engine stop/restart via HTTP API, TTFT observation"
      echo "  engine_kill_restart — kill -9 victim engine, restart, verify recovery"
      echo "  master_kill_restart — kill -9 master, restart, 5 hard assertions"
      echo "  master_recovery     — kill master + fallback + restart + TTFT recovery"
      echo ""
      echo "Sub-scenario selection (behavior only):"
      echo "  SCENARIOS=1,2 bash run_resilience.sh --scenario behavior"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1;;
  esac
done

# -- Configuration (single layer: env var with fallback) --------------------
# All env vars pass through to delegated scripts via process inheritance.
# No multi-layer override — one set of env vars, printed at startup.

# Cluster topology (all scenarios default to 2P+2D)
N_PREFILL="${N_PREFILL:-2}"
N_DECODE="${N_DECODE:-2}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-55151}"
MOCK_HTTP_PORT=$((MOCK_BASE_GRPC_PORT - 1))
FLEXLB_HTTP_PORT="${FLEXLB_HTTP_PORT:-18080}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-18081}"
FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"
PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS:-6000}"
DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS:-3000}"
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"
FLEXLB_FAIL_ON_CONCURRENT_TEST="${FLEXLB_FAIL_ON_CONCURRENT_TEST:-1}"

# Behavior test specific
FLEXLB_INFLIGHT_TTL_MS="${FLEXLB_INFLIGHT_TTL_MS:-30000}"
FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES:-4}"
SYNC_REQUEST_TIMEOUT_MS="${SYNC_REQUEST_TIMEOUT_MS:-1000}"
SCENARIOS="${SCENARIOS:-1,2,3,4}"

# Load client parameters (shared by kill/restart and TTFT scenarios)
LOAD_CLIENT_LIMIT="${LOAD_CLIENT_LIMIT:-0}"
LOAD_CLIENT_CONCURRENCY="${LOAD_CLIENT_CONCURRENCY:-20}"
LOAD_CLIENT_TIMEOUT_MS="${LOAD_CLIENT_TIMEOUT_MS:-10000}"
LOAD_CLIENT_REPLAY_SPEED="${LOAD_CLIENT_REPLAY_SPEED:-20}"
LOAD_CLIENT_TRACE_FILTER_OL="${LOAD_CLIENT_TRACE_FILTER_OL:-200}"

# TTFT scenarios use higher mock latency for observable TTFT
MOCK_PREFILL_MS="${MOCK_PREFILL_MS:-1000.0}"
MOCK_DECODE_STEP_MS="${MOCK_DECODE_STEP_MS:-100.0}"

# Timing parameters — kill/restart scenarios
STEADY_STATE_WAIT="${STEADY_STATE_WAIT:-8}"
KILL_WAIT="${KILL_WAIT:-8}"
RECOVERY_WAIT="${RECOVERY_WAIT:-15}"
KILL_TIMING="${KILL_TIMING:-steady}"

# Timing parameters — TTFT scenarios
BASELINE_DURATION="${BASELINE_DURATION:-60}"
DISCONNECT_WAIT="${DISCONNECT_WAIT:-30}"
DISCONNECT_TARGET="${DISCONNECT_TARGET:-prefill-0}"
LOAD_CLIENT_PAUSE_DURING_KILL="${LOAD_CLIENT_PAUSE_DURING_KILL:-0}"

# Engine kill scenario
ENGINE_MODE="${ENGINE_MODE:-multi}"
KILL_TARGET="${KILL_TARGET:-prefill}"

# Stability monitor
MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"

# -- Print effective configuration ------------------------------------------

print_env_summary \
  RESILIENCE_SCENARIO N_PREFILL N_DECODE MOCK_BASE_GRPC_PORT \
  FLEXLB_HTTP_PORT FLEXLB_MANAGEMENT_PORT FLEXLB_JAR \
  PREFILL_CACHE_BLOCKS DECODE_CACHE_BLOCKS \
  FLEXLB_INFLIGHT_TTL_MS FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES \
  SYNC_REQUEST_TIMEOUT_MS SCENARIOS \
  LOAD_CLIENT_LIMIT LOAD_CLIENT_CONCURRENCY LOAD_CLIENT_TIMEOUT_MS \
  LOAD_CLIENT_REPLAY_SPEED LOAD_CLIENT_TRACE_FILTER_OL \
  MOCK_PREFILL_MS MOCK_DECODE_STEP_MS \
  STEADY_STATE_WAIT KILL_WAIT RECOVERY_WAIT KILL_TIMING \
  BASELINE_DURATION DISCONNECT_WAIT DISCONNECT_TARGET \
  LOAD_CLIENT_PAUSE_DURING_KILL ENGINE_MODE KILL_TARGET \
  MONITOR_INTERVAL

# -- Concurrency guard ------------------------------------------------------

if [[ "${FLEXLB_FAIL_ON_CONCURRENT_TEST}" == "1" ]]; then
  assert_no_concurrent_flexlb_test
fi

# -- Trace file check -------------------------------------------------------

TRACE_FILE="${SCRIPT_DIR}/data/online_logs/trace_30min.jsonl"
if [[ ! -f "${TRACE_FILE}" ]]; then
  echo "WARNING: trace file not found: ${TRACE_FILE}" >&2
  echo "  Most resilience scenarios require this trace file." >&2
fi

# ===========================================================================
# Scenario handlers — each delegates to the corresponding original script.
# Env vars pass through via process inheritance (single layer, no override).
# ===========================================================================

scenario_behavior() {
  echo "============================================"
  echo "  Scenario: behavior (sub-scenarios: ${SCENARIOS})"
  echo "  Delegates to: flexlb_behavior_test.sh"
  echo "============================================"
  echo ""
  set +e
  bash "${SCRIPT_DIR}/flexlb_behavior_test.sh"
  local rc=$?
  set -e
  return ${rc}
}

scenario_disconnect_ttft() {
  echo "============================================"
  echo "  Scenario: disconnect_ttft (target: ${DISCONNECT_TARGET})"
  echo "  Delegates to: engine_disconnect_ttft_test.sh"
  echo "============================================"
  echo ""
  set +e
  bash "${SCRIPT_DIR}/engine_disconnect_ttft_test.sh"
  local rc=$?
  set -e
  return ${rc}
}

scenario_engine_kill_restart() {
  echo "============================================"
  echo "  Scenario: engine_kill_restart (mode: ${ENGINE_MODE}, target: ${KILL_TARGET})"
  echo "  Delegates to: engine_kill_restart_test.sh"
  echo "============================================"
  echo ""
  set +e
  bash "${SCRIPT_DIR}/engine_kill_restart_test.sh"
  local rc=$?
  set -e
  return ${rc}
}

scenario_master_kill_restart() {
  echo "============================================"
  echo "  Scenario: master_kill_restart (timing: ${KILL_TIMING})"
  echo "  Delegates to: master_kill_restart_test.sh"
  echo "============================================"
  echo ""
  set +e
  bash "${SCRIPT_DIR}/master_kill_restart_test.sh"
  local rc=$?
  set -e
  return ${rc}
}

scenario_master_recovery() {
  echo "============================================"
  echo "  Scenario: master_recovery (pause_during_kill: ${LOAD_CLIENT_PAUSE_DURING_KILL})"
  echo "  Delegates to: master_recovery_ttft_test.sh"
  echo "============================================"
  echo ""
  set +e
  bash "${SCRIPT_DIR}/master_recovery_ttft_test.sh"
  local rc=$?
  set -e
  return ${rc}
}

# ===========================================================================
# Main dispatch
# ===========================================================================

echo ""
echo "Starting resilience test: ${RESILIENCE_SCENARIO}"
echo ""

set +e
case "${RESILIENCE_SCENARIO}" in
  behavior)            scenario_behavior ;;
  disconnect_ttft)     scenario_disconnect_ttft ;;
  engine_kill_restart) scenario_engine_kill_restart ;;
  master_kill_restart) scenario_master_kill_restart ;;
  master_recovery)     scenario_master_recovery ;;
  *)
    echo "ERROR: unknown scenario '${RESILIENCE_SCENARIO}'" >&2
    echo "Available: behavior, disconnect_ttft, engine_kill_restart, master_kill_restart, master_recovery" >&2
    exit 1
    ;;
esac
SCENARIO_RC=$?
set -e

# ===========================================================================
# Summary
# ===========================================================================

echo ""
echo "=========================================="
echo "  Resilience Test Complete"
echo "=========================================="
echo "  Scenario:  ${RESILIENCE_SCENARIO}"
if [[ ${SCENARIO_RC} -eq 0 ]]; then
  echo "  Status:    PASS (exit=0)"
else
  echo "  Status:    FAIL (exit=${SCENARIO_RC})"
fi
echo "=========================================="

exit ${SCENARIO_RC}

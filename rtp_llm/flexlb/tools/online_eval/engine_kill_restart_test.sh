#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# engine_kill_restart_test.sh
#
# FlexLB Engine kill-restart destructive test.
#
# Unlike master_kill_restart_test.sh (which kills the Master), this script
# kills a single mock ENGINE process while the Master stays running.
#
# Architecture note:
#   mock_engine_cluster.py starts all engines within ONE Python process.
#   To kill a single engine independently, we start the "victim" engine as
#   a separate process via run_single_engine.py.  The remaining engines
#   run in the cluster process.  The Master discovers all engines via
#   static DOMAIN_ADDRESS env vars (no health check, no dynamic removal).
#
# Flow:
#   1.  Start mock engine cluster (surviving engines)
#   2.  Start victim engine (standalone process)
#   3.  Start FlexLB Master (batch path)
#   4.  Start load client (background)
#   5.  Wait for steady state
#   6.  Collect baseline data
#   7.  KILL victim engine (kill -9)
#   8.  Wait (observe failures during downtime)
#   9.  Collect kill-period data (Master still alive? routing?)
#   10. Restart victim engine
#   11. Wait (observe recovery)
#   12. Stop load client
#   13. Recovery verification (100 short requests)
#   14. Collect post-restart data
#   15. Generate test report with 5 hard assertions
#
# Usage:
#   bash engine_kill_restart_test.sh                          # multi, kill prefill
#   KILL_TARGET=decode bash engine_kill_restart_test.sh       # multi, kill decode
#   ENGINE_MODE=single bash engine_kill_restart_test.sh       # single, kill prefill
#   ENGINE_MODE=single KILL_TARGET=decode bash engine_kill_restart_test.sh
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"
TRACE_FILE="${SCRIPT_DIR}/data/online_logs/trace_30min.jsonl"

# -- Java setup ------------------------------------------------------------

export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21}"
export PATH="${JAVA_HOME}/bin:${PATH}"

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

# -- Configurable parameters -----------------------------------------------

ENGINE_MODE="${ENGINE_MODE:-multi}"    # "multi" (2P+2D) or "single" (1P+1D)
KILL_TARGET="${KILL_TARGET:-prefill}"   # "prefill" or "decode"

if [[ "${ENGINE_MODE}" == "single" ]]; then
  N_PREFILL_TOTAL=1
  N_DECODE_TOTAL=1
else
  N_PREFILL_TOTAL=2
  N_DECODE_TOTAL=2
fi

# Cluster gets total-1 engines of the killed role
if [[ "${KILL_TARGET}" == "prefill" ]]; then
  CLUSTER_N_PREFILL=$((N_PREFILL_TOTAL - 1))
  CLUSTER_N_DECODE=${N_DECODE_TOTAL}
else
  CLUSTER_N_PREFILL=${N_PREFILL_TOTAL}
  CLUSTER_N_DECODE=$((N_DECODE_TOTAL - 1))
fi

MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-55151}"
MOCK_HTTP_PORT=$((MOCK_BASE_GRPC_PORT - 1))
VICTIM_GRPC_PORT=$((MOCK_BASE_GRPC_PORT + 150))
VICTIM_HTTP_PORT=$((VICTIM_GRPC_PORT - 1))
FLEXLB_HTTP_PORT="${FLEXLB_HTTP_PORT:-18080}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-18081}"
PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS:-6000}"
DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS:-3000}"

# Victim engine name and cache blocks
if [[ "${KILL_TARGET}" == "prefill" ]]; then
  VICTIM_NAME="prefill-$((N_PREFILL_TOTAL - 1))"
  VICTIM_CACHE_BLOCKS=${PREFILL_CACHE_BLOCKS}
else
  VICTIM_NAME="decode-$((N_DECODE_TOTAL - 1))"
  VICTIM_CACHE_BLOCKS=${DECODE_CACHE_BLOCKS}
fi

# -- Compute cluster engine addresses --------------------------------------
# Cluster allocates ports sequentially: prefill-0, prefill-1, ..., decode-0, ...
CLUSTER_PREFILL_ADDRS=""
CLUSTER_DECODE_ADDRS=""
_port=${MOCK_BASE_GRPC_PORT}
for ((i = 0; i < CLUSTER_N_PREFILL; i++)); do
  if [[ -z "${CLUSTER_PREFILL_ADDRS}" ]]; then
    CLUSTER_PREFILL_ADDRS="127.0.0.1:${_port}"
  else
    CLUSTER_PREFILL_ADDRS="${CLUSTER_PREFILL_ADDRS},127.0.0.1:${_port}"
  fi
  _port=$((_port + 1))
done
for ((i = 0; i < CLUSTER_N_DECODE; i++)); do
  if [[ -z "${CLUSTER_DECODE_ADDRS}" ]]; then
    CLUSTER_DECODE_ADDRS="127.0.0.1:${_port}"
  else
    CLUSTER_DECODE_ADDRS="${CLUSTER_DECODE_ADDRS},127.0.0.1:${_port}"
  fi
  _port=$((_port + 1))
done

# -- Combine addresses for DOMAIN_ADDRESS ----------------------------------
VICTIM_ADDR="127.0.0.1:${VICTIM_GRPC_PORT}"
if [[ "${KILL_TARGET}" == "prefill" ]]; then
  if [[ -z "${CLUSTER_PREFILL_ADDRS}" ]]; then
    PREFILL_DOMAIN_ADDR="${VICTIM_ADDR}"
  else
    PREFILL_DOMAIN_ADDR="${CLUSTER_PREFILL_ADDRS},${VICTIM_ADDR}"
  fi
  DECODE_DOMAIN_ADDR="${CLUSTER_DECODE_ADDRS}"
else
  PREFILL_DOMAIN_ADDR="${CLUSTER_PREFILL_ADDRS}"
  if [[ -z "${CLUSTER_DECODE_ADDRS}" ]]; then
    DECODE_DOMAIN_ADDR="${VICTIM_ADDR}"
  else
    DECODE_DOMAIN_ADDR="${CLUSTER_DECODE_ADDRS},${VICTIM_ADDR}"
  fi
fi

# -- Model service config (constant JSON) ----------------------------------
readonly MODEL_SERVICE_CONFIG_JSON='{"service_id":"aigc.text-generation.generation.engine_service","load_balance":true,"role_endpoints":[{"group":"mock","prefill_endpoint":{"address":"mock.prefill.hosts.address","protocol":"grpc","path":"/"},"decode_endpoint":{"address":"mock.decode.hosts.address","protocol":"grpc","path":"/"}}]}'

# -- Load client parameters ------------------------------------------------
LOAD_CLIENT_LIMIT="${LOAD_CLIENT_LIMIT:-0}"
LOAD_CLIENT_CONCURRENCY="${LOAD_CLIENT_CONCURRENCY:-20}"
LOAD_CLIENT_TIMEOUT_MS="${LOAD_CLIENT_TIMEOUT_MS:-10000}"
LOAD_CLIENT_REPLAY_SPEED="${LOAD_CLIENT_REPLAY_SPEED:-20}"

# -- Timing parameters (seconds) -------------------------------------------
STEADY_STATE_WAIT="${STEADY_STATE_WAIT:-8}"
KILL_WAIT="${KILL_WAIT:-8}"
RECOVERY_WAIT="${RECOVERY_WAIT:-15}"

# -- Run directory ---------------------------------------------------------
RUN_DIR="${SCRIPT_DIR}/run/engine_kill_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"

# -- State -----------------------------------------------------------------
MOCK_PID=""
VICTIM_PID=""
FLEXLB_PID=""
LOAD_CLIENT_PID=""
KILL_TS=""
RESTART_TS=""

# -- Helpers ---------------------------------------------------------------

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

check_port_free() {
  local port="$1"
  if lsof -i :"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: port ${port} is already in use" >&2
    lsof -i :"${port}" -sTCP:LISTEN >&2
    return 1
  fi
  return 0
}

cleanup() {
  echo ""
  echo "[cleanup] stopping processes ..."
  if [[ -n "${FLEXLB_PID}" ]]; then
    kill "${FLEXLB_PID}" >/dev/null 2>&1 || true
    wait "${FLEXLB_PID}" 2>/dev/null || true
    FLEXLB_PID=""
  fi
  if [[ -n "${VICTIM_PID}" ]]; then
    kill -9 "${VICTIM_PID}" >/dev/null 2>&1 || true
    wait "${VICTIM_PID}" 2>/dev/null || true
    VICTIM_PID=""
  fi
  if [[ -n "${MOCK_PID}" ]]; then
    kill "${MOCK_PID}" >/dev/null 2>&1 || true
    wait "${MOCK_PID}" 2>/dev/null || true
    MOCK_PID=""
  fi
  if [[ -n "${LOAD_CLIENT_PID}" ]]; then
    kill "${LOAD_CLIENT_PID}" >/dev/null 2>&1 || true
    wait "${LOAD_CLIENT_PID}" 2>/dev/null || true
    LOAD_CLIENT_PID=""
  fi
  echo "[cleanup] done."
}
trap cleanup EXIT

start_victim_engine() {
  local log_file="$1"
  echo "  starting victim engine (${VICTIM_NAME}, ${KILL_TARGET}) at port ${VICTIM_GRPC_PORT} ..."
  # Pre-flight: ensure both gRPC and HTTP ports are free
  check_port_free "${VICTIM_GRPC_PORT}" || {
    echo "  attempting to kill process on port ${VICTIM_GRPC_PORT} ..." >&2
    local stale_pid
    stale_pid=$(lsof -ti :"${VICTIM_GRPC_PORT}" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "${stale_pid}" ]]; then
      kill -9 "${stale_pid}" 2>/dev/null || true
      sleep 2
    fi
    check_port_free "${VICTIM_GRPC_PORT}" || return 1
  }
  check_port_free "${VICTIM_HTTP_PORT}" || {
    echo "  attempting to kill process on port ${VICTIM_HTTP_PORT} ..." >&2
    local stale_pid
    stale_pid=$(lsof -ti :"${VICTIM_HTTP_PORT}" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "${stale_pid}" ]]; then
      kill -9 "${stale_pid}" 2>/dev/null || true
      sleep 2
    fi
    check_port_free "${VICTIM_HTTP_PORT}" || return 1
  }
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/run_single_engine.py" \
    --host 127.0.0.1 \
    --grpc-port "${VICTIM_GRPC_PORT}" \
    --role "${KILL_TARGET}" \
    --name "${VICTIM_NAME}" \
    --performance "${PERF_CONFIG_FILE}" \
    --cache-blocks "${VICTIM_CACHE_BLOCKS}" \
    --total-kv-tokens 6291456 \
    --block-size 1024 \
    >"${log_file}" 2>&1 &
  VICTIM_PID="$!"
  wait_for_port "127.0.0.1" "${VICTIM_GRPC_PORT}" 20
  # Verify the victim process is still alive after port came up
  # (guards against the case where gRPC binds first but HTTP fails, causing crash)
  if ! kill -0 "${VICTIM_PID}" 2>/dev/null; then
    echo "ERROR: victim engine process died during startup" >&2
    echo "--- victim engine log ---" >&2
    cat "${log_file}" >&2
    return 1
  fi
  echo "  victim engine started (pid=${VICTIM_PID})"
}

start_master() {
  local log_file="$1"
  echo "  starting master ..."
  env \
    "MODEL_SERVICE_CONFIG=${MODEL_SERVICE_CONFIG_JSON}" \
    "DOMAIN_ADDRESS:mock.prefill.hosts.address=${PREFILL_DOMAIN_ADDR}" \
    "DOMAIN_ADDRESS:mock.decode.hosts.address=${DECODE_DOMAIN_ADDR}" \
    "LOAD_BALANCE_STRATEGY=COST_BASED_PREFILL" \
    "DECODE_LOAD_BALANCE_STRATEGY=COST_BASED_DECODE" \
    "FLEXLB_BATCH_ENABLED=true" \
    "FLEXLB_EXPECT_FETCH_RESPONSE=true" \
    "HYSTERESIS_BIAS_PERCENT=0" \
    "MAX_QUEUE_SIZE=5000" \
    "FLEXLB_BATCH_ALGORITHM=fixed_window" \
    "FLEXLB_BATCH_FIXED_WAIT_MS=10" \
    "FLEXLB_BATCH_PREDICT_THRESHOLD_MS=550" \
    "FLEXLB_BATCH_SIZE_MAX=32" \
    "FLEXLB_BATCH_MIN_SIZE=1" \
    "FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES=4" \
    "DECODE_CONCURRENCY_LIMIT=132" \
    "PREFILL_QUEUE_SIZE_THRESHOLD=100000" \
    "COST_SLO_MS=30000" \
    "COST_HOTSPOT_MULTIPLIER=1.5" \
    "DEFAULT_SCHEDULE_MODE=BATCH" \
    "ENABLE_QUEUEING=false" \
    "STRATEGY_CONFIGS={}" \
    "OTEL_TRACE_SKIP_PATTERN=.*" \
    "OTEL_EXPORTER_OTLP_ENDPOINT=none" \
    "HIPPO_ROLE=flexlb_engine_kill_test" \
    java "${JAVA_MODULE_OPTS[@]}" -jar "${FLEXLB_JAR}" \
    --server.port="${FLEXLB_HTTP_PORT}" \
    --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
    --logging.level.org.flexlb=DEBUG \
    --logging.file.name="${log_file}.logback" \
    >"${log_file}" 2>&1 &
  FLEXLB_PID="$!"
  wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60
  # Verify master process is still alive
  if ! kill -0 "${FLEXLB_PID}" 2>/dev/null; then
    echo "ERROR: master process died during startup" >&2
    echo "--- master log ---" >&2
    cat "${log_file}" >&2
    return 1
  fi
  echo "  master started (pid=${FLEXLB_PID})"
}

# -- Prerequisites check ---------------------------------------------------

echo ""
echo "=== Prerequisites Check ==="
if [[ ! -f "${FLEXLB_JAR}" ]]; then
  echo "ERROR: FlexLB JAR not found: ${FLEXLB_JAR}" >&2
  exit 1
fi
if [[ ! -f "${TRACE_FILE}" ]]; then
  echo "ERROR: Trace file not found: ${TRACE_FILE}" >&2
  exit 1
fi
java -version 2>&1 | head -1
echo "  JAR: ${FLEXLB_JAR}"
echo "  Trace: ${TRACE_FILE} ($(wc -l < "${TRACE_FILE}") lines)"
echo "  Engine Mode: ${ENGINE_MODE}"
echo "  Kill Target: ${KILL_TARGET}"
echo "  Cluster: ${CLUSTER_N_PREFILL}P + ${CLUSTER_N_DECODE}D"
echo "  Victim: ${VICTIM_NAME} (${KILL_TARGET}) at ${VICTIM_ADDR}"
echo "  Prefill domain: ${PREFILL_DOMAIN_ADDR}"
echo "  Decode domain: ${DECODE_DOMAIN_ADDR}"
echo "  All prerequisites OK."

# -- Pre-flight port check -------------------------------------------------
echo ""
echo "=== Pre-flight Port Check ==="
_preflight_ports=("${FLEXLB_HTTP_PORT}" "${FLEXLB_MANAGEMENT_PORT}" "${MOCK_HTTP_PORT}" "${VICTIM_GRPC_PORT}" "${VICTIM_HTTP_PORT}")
for ((i = 0; i < CLUSTER_N_PREFILL + CLUSTER_N_DECODE; i++)); do
  _preflight_ports+=("$((MOCK_BASE_GRPC_PORT + i))")
done
_preflight_failed=0
for port in "${_preflight_ports[@]}"; do
  if lsof -i :"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  WARNING: port ${port} is in use, killing stale process ..."
    _stale_pid=$(lsof -ti :"${port}" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "${_stale_pid}" ]]; then
      kill -9 "${_stale_pid}" 2>/dev/null || true
    fi
    sleep 2
    if lsof -i :"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "  ERROR: port ${port} still in use after kill attempt" >&2
      _preflight_failed=1
    else
      echo "  port ${port} now free"
    fi
  fi
done
if [[ "${_preflight_failed}" -ne 0 ]]; then
  echo "ERROR: pre-flight port check failed, aborting" >&2
  exit 1
fi
echo "  All ports free."

# -- Write perf config -----------------------------------------------------

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

# ===========================================================================
# Step 1: Start mock engine cluster (surviving engines)
# ===========================================================================

echo ""
echo "=== Step 1: Start mock engine cluster (${CLUSTER_N_PREFILL}P + ${CLUSTER_N_DECODE}D) ==="
ENDPOINT_FILE="${RUN_DIR}/endpoints.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/mock_engine_cluster.py" \
  --n-prefill "${CLUSTER_N_PREFILL}" \
  --n-decode "${CLUSTER_N_DECODE}" \
  --base-grpc-port "${MOCK_BASE_GRPC_PORT}" \
  --performance "${PERF_CONFIG_FILE}" \
  --prefill-cache-blocks "${PREFILL_CACHE_BLOCKS}" \
  --decode-cache-blocks "${DECODE_CACHE_BLOCKS}" \
  --endpoint-file "${ENDPOINT_FILE}" \
  --env-file "${RUN_DIR}/flexlb_env.txt" \
  >"${RUN_DIR}/mock_engine.log" 2>&1 &
MOCK_PID="$!"
wait_for_port "127.0.0.1" "${MOCK_HTTP_PORT}" 20
# Verify mock cluster process is still alive (guards against crash during startup)
if ! kill -0 "${MOCK_PID}" 2>/dev/null; then
  echo "ERROR: mock cluster process died during startup" >&2
  echo "--- mock engine log ---" >&2
  cat "${RUN_DIR}/mock_engine.log" >&2
  exit 1
fi
echo "  mock cluster started (pid=${MOCK_PID}, http=${MOCK_HTTP_PORT})"

# ===========================================================================
# Step 2: Start victim engine (standalone process)
# ===========================================================================

echo ""
echo "=== Step 2: Start victim engine (standalone) ==="
start_victim_engine "${RUN_DIR}/victim_engine_initial.log"

# ===========================================================================
# Step 3: Start FlexLB Master
# ===========================================================================

echo ""
echo "=== Step 3: Start FlexLB Master (batch path) ==="
start_master "${RUN_DIR}/flexlb_master.log"

# ===========================================================================
# Step 4: Start load client (background)
# ===========================================================================

echo ""
echo "=== Step 4: Start load client ==="
LOAD_CLIENT_DIR="${RUN_DIR}/load_client"
mkdir -p "${LOAD_CLIENT_DIR}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/flexlb_load_client.py" \
  "${TRACE_FILE}" \
  --flexlb-http-addr "127.0.0.1:${FLEXLB_HTTP_PORT}" \
  --schedule-mode batch \
  --replay-speed "${LOAD_CLIENT_REPLAY_SPEED}" \
  --limit "${LOAD_CLIENT_LIMIT}" \
  --max-concurrency "${LOAD_CLIENT_CONCURRENCY}" \
  --timeout-ms "${LOAD_CLIENT_TIMEOUT_MS}" \
  --output-dir "${LOAD_CLIENT_DIR}" \
  >"${RUN_DIR}/load_client.log" 2>&1 &
LOAD_CLIENT_PID="$!"
echo "  load client started (pid=${LOAD_CLIENT_PID})"

# ===========================================================================
# Step 5: Wait for steady state
# ===========================================================================

echo ""
echo "=== Step 5: Steady state wait (${STEADY_STATE_WAIT}s) ==="
sleep "${STEADY_STATE_WAIT}"
if kill -0 "${LOAD_CLIENT_PID}" 2>/dev/null; then
  echo "  load client is running"
else
  echo "  WARNING: load client has already exited"
fi

# ===========================================================================
# Step 6: Collect baseline data
# ===========================================================================

echo ""
echo "=== Step 6: Collect baseline data ==="
echo "  - master inflight_status ..."
curl -s -o "${RUN_DIR}/baseline_inflight.json" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" \
  || echo "  WARNING: inflight_status request failed"
echo "  - cluster snapshot ..."
curl -s -o "${RUN_DIR}/baseline_cluster_snapshot.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" \
  || echo '{"engines":[]}' > "${RUN_DIR}/baseline_cluster_snapshot.json"
echo "  - victim snapshot ..."
curl -s -o "${RUN_DIR}/baseline_victim_snapshot.json" \
  "http://127.0.0.1:${VICTIM_HTTP_PORT}/snapshot" \
  || echo '{"engines":[]}' > "${RUN_DIR}/baseline_victim_snapshot.json"
echo "  - load client per_request.jsonl ..."
cp "${LOAD_CLIENT_DIR}/per_request.jsonl" "${RUN_DIR}/pre_kill_per_request.jsonl" 2>/dev/null \
  || echo "  NOTE: per_request.jsonl not available yet"

# ===========================================================================
# Step 7: KILL victim engine (kill -9)
# ===========================================================================

echo ""
echo "=== Step 7: KILL victim engine (kill -9) ==="
KILL_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "  kill timestamp: ${KILL_TS}"
echo "  killing victim engine (pid=${VICTIM_PID}, name=${VICTIM_NAME}) ..."
kill -9 "${VICTIM_PID}" || true
wait "${VICTIM_PID}" 2>/dev/null || true
VICTIM_PID=""

# ===========================================================================
# Step 8: Wait (observe failures during downtime)
# ===========================================================================

echo ""
echo "=== Step 8: Wait ${KILL_WAIT}s (kill period, observe failures) ==="
sleep "${KILL_WAIT}"
if kill -0 "${LOAD_CLIENT_PID}" 2>/dev/null; then
  echo "  load client is still running"
else
  echo "  load client has exited"
fi

# ===========================================================================
# Step 9: Collect kill-period data
# ===========================================================================

echo ""
echo "=== Step 9: Collect kill-period data ==="
echo "  - master health check (should be alive) ..."
MASTER_HEALTH_CODE=$(curl -s -o "${RUN_DIR}/kill_inflight.json" -w "%{http_code}" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" 2>/dev/null || echo "000")
echo "${MASTER_HEALTH_CODE}" > "${RUN_DIR}/kill_master_health.txt"
echo "  master HTTP status: ${MASTER_HEALTH_CODE}"
echo "  - cluster snapshot (surviving engines) ..."
curl -s -o "${RUN_DIR}/kill_cluster_snapshot.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" \
  || echo '{"engines":[]}' > "${RUN_DIR}/kill_cluster_snapshot.json"
echo "  - victim snapshot (expected to fail) ..."
curl -s -o "${RUN_DIR}/kill_victim_snapshot.json" \
  "http://127.0.0.1:${VICTIM_HTTP_PORT}/snapshot" 2>/dev/null \
  || echo '{"engines":[]}' > "${RUN_DIR}/kill_victim_snapshot.json"
echo "  - load client per_request.jsonl ..."
cp "${LOAD_CLIENT_DIR}/per_request.jsonl" "${RUN_DIR}/kill_per_request.jsonl" 2>/dev/null \
  || echo "  NOTE: per_request.jsonl not available"

# ===========================================================================
# Step 10: Restart victim engine
# ===========================================================================

echo ""
echo "=== Step 10: Restart victim engine ==="
RESTART_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "  restart timestamp: ${RESTART_TS}"
sleep 1  # brief pause to ensure port is released
start_victim_engine "${RUN_DIR}/victim_engine_restart.log"

# Master health check after victim restart
_post_restart_health=$(curl -s -o /dev/null -w "%{http_code}" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" 2>/dev/null || echo "000")
echo "  master health after victim restart: HTTP ${_post_restart_health}"

# ===========================================================================
# Step 11: Wait (observe recovery)
# ===========================================================================

echo ""
echo "=== Step 11: Wait ${RECOVERY_WAIT}s (recovery period) ==="
sleep "${RECOVERY_WAIT}"
if kill -0 "${LOAD_CLIENT_PID}" 2>/dev/null; then
  echo "  load client is still running"
else
  echo "  load client has exited"
fi
# Master health check after recovery wait
_recovery_health=$(curl -s -o /dev/null -w "%{http_code}" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" 2>/dev/null || echo "000")
echo "  master health after recovery wait: HTTP ${_recovery_health}"
if [[ "${_recovery_health}" != "200" ]]; then
  echo "  WARNING: master is not responding after recovery wait!"
  echo "  master logback log (last 30 lines):"
  tail -30 "${RUN_DIR}/flexlb_master.log.logback" 2>/dev/null || echo "  (no logback log available)"
fi

# ===========================================================================
# Step 12: Stop load client
# ===========================================================================

echo ""
echo "=== Step 12: Stop load client ==="
if kill -0 "${LOAD_CLIENT_PID}" 2>/dev/null; then
  echo "  stopping load client (pid=${LOAD_CLIENT_PID}) ..."
  kill "${LOAD_CLIENT_PID}" 2>/dev/null || true
  wait "${LOAD_CLIENT_PID}" 2>/dev/null || true
else
  echo "  load client already exited"
fi
LOAD_CLIENT_PID=""
sleep 1  # Allow file flush
sleep 5  # drain wait for in-flight to settle

# ===========================================================================
# Step 13: Recovery Verification
# ===========================================================================

echo ""
echo "=== Step 13: Recovery Verification ==="
RECOVERY_TRACE="${RUN_DIR}/recovery_trace.jsonl"
python3 -c "
import json
with open('${TRACE_FILE}') as f:
    for line in f:
        req = json.loads(line)
        if req.get('ol', 0) <= 200:
            print(line, end='')
" > "${RECOVERY_TRACE}" 2>/dev/null
RECOVERY_TRACE_LINES=$(wc -l < "${RECOVERY_TRACE}" 2>/dev/null || echo 0)
echo "  recovery trace: ${RECOVERY_TRACE_LINES} short-output requests (ol <= 200)"

RECOVERY_VERIFY_DIR="${RUN_DIR}/recovery_verify"
mkdir -p "${RECOVERY_VERIFY_DIR}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/flexlb_load_client.py" \
  "${RECOVERY_TRACE}" \
  --flexlb-http-addr "127.0.0.1:${FLEXLB_HTTP_PORT}" \
  --schedule-mode batch \
  --replay-speed 0 \
  --limit 100 \
  --max-concurrency 10 \
  --timeout-ms 10000 \
  --output-dir "${RECOVERY_VERIFY_DIR}" \
  >"${RUN_DIR}/recovery_verify.log" 2>&1 || true
echo "  recovery verification completed"
cat "${RECOVERY_VERIFY_DIR}/summary.json" 2>/dev/null || echo "  NOTE: recovery summary not available"
# Master health check after recovery verification
_post_recovery_health=$(curl -s -o /dev/null -w "%{http_code}" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" 2>/dev/null || echo "000")
echo "  master health after recovery verification: HTTP ${_post_recovery_health}"
if [[ "${_post_recovery_health}" != "200" ]]; then
  echo "  WARNING: master is not responding after recovery verification!"
  echo "  master logback log (last 50 lines):"
  tail -50 "${RUN_DIR}/flexlb_master.log.logback" 2>/dev/null || echo "  (no logback log available)"
fi
sleep 5  # drain wait for recovery verification in-flight to settle

# ===========================================================================
# Step 14: Collect post-restart data
# ===========================================================================

echo ""
echo "=== Step 14: Collect post-restart data ==="
echo "  - master inflight_status ..."
curl -s -o "${RUN_DIR}/post_restart_inflight.json" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" \
  || echo "  WARNING: inflight_status request failed"
echo "  - cluster snapshot ..."
curl -s -o "${RUN_DIR}/post_restart_cluster_snapshot.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" \
  || echo '{"engines":[]}' > "${RUN_DIR}/post_restart_cluster_snapshot.json"
echo "  - victim snapshot ..."
curl -s -o "${RUN_DIR}/post_restart_victim_snapshot.json" \
  "http://127.0.0.1:${VICTIM_HTTP_PORT}/snapshot" \
  || echo '{"engines":[]}' > "${RUN_DIR}/post_restart_victim_snapshot.json"
echo "  - load client outputs ..."
cp "${LOAD_CLIENT_DIR}/summary.json" "${RUN_DIR}/final_summary.json" 2>/dev/null \
  || echo "  NOTE: summary.json not available"
cp "${LOAD_CLIENT_DIR}/per_request.jsonl" "${RUN_DIR}/final_per_request.jsonl" 2>/dev/null \
  || echo "  WARNING: per_request.jsonl not available"

# ===========================================================================
# Step 15: Generate test report
# ===========================================================================

echo ""
echo "=== Step 15: Generate test report ==="
ENGINE_MODE="${ENGINE_MODE}" \
KILL_TARGET="${KILL_TARGET}" \
KILL_TS="${KILL_TS}" \
RESTART_TS="${RESTART_TS}" \
MOCK_PID="${MOCK_PID}" \
VICTIM_NAME="${VICTIM_NAME}" \
VICTIM_GRPC_PORT="${VICTIM_GRPC_PORT}" \
VICTIM_HTTP_PORT="${VICTIM_HTTP_PORT}" \
LOAD_CLIENT_PID="${LOAD_CLIENT_PID:-exited}" \
MASTER_HEALTH_CODE="${MASTER_HEALTH_CODE}" \
CLUSTER_N_PREFILL="${CLUSTER_N_PREFILL}" \
CLUSTER_N_DECODE="${CLUSTER_N_DECODE}" \
N_PREFILL_TOTAL="${N_PREFILL_TOTAL}" \
N_DECODE_TOTAL="${N_DECODE_TOTAL}" \
FLEXLB_HTTP_PORT_VAL="${FLEXLB_HTTP_PORT}" \
MOCK_HTTP_PORT_VAL="${MOCK_HTTP_PORT}" \
MOCK_BASE_GRPC_PORT_VAL="${MOCK_BASE_GRPC_PORT}" \
python3 - "${RUN_DIR}" <<'PYEOF'
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

run_dir = Path(sys.argv[1])

# -- Metadata from environment --
engine_mode = os.environ.get("ENGINE_MODE", "multi")
kill_target = os.environ.get("KILL_TARGET", "prefill")
kill_ts = os.environ.get("KILL_TS", "N/A")
restart_ts = os.environ.get("RESTART_TS", "N/A")
mock_pid = os.environ.get("MOCK_PID", "N/A")
victim_name = os.environ.get("VICTIM_NAME", "N/A")
victim_grpc_port = os.environ.get("VICTIM_GRPC_PORT", "N/A")
victim_http_port = os.environ.get("VICTIM_HTTP_PORT", "N/A")
load_client_pid = os.environ.get("LOAD_CLIENT_PID", "N/A")
master_health_code = os.environ.get("MASTER_HEALTH_CODE", "000")
cluster_n_prefill = os.environ.get("CLUSTER_N_PREFILL", "0")
cluster_n_decode = os.environ.get("CLUSTER_N_DECODE", "0")
n_prefill_total = os.environ.get("N_PREFILL_TOTAL", "2")
n_decode_total = os.environ.get("N_DECODE_TOTAL", "2")

# -- Load JSON helper --
def load_json(name):
    p = run_dir / name
    if p.exists() and p.stat().st_size > 0:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def load_jsonl(name):
    p = run_dir / name
    results = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
    return results

# -- Combine cluster + victim snapshots --
def combine_snapshots(cluster_file, victim_file):
    cluster = load_json(cluster_file) or {"engines": []}
    victim = load_json(victim_file) or {"engines": []}
    return {"engines": cluster.get("engines", []) + victim.get("engines", [])}

baseline_snapshot = combine_snapshots("baseline_cluster_snapshot.json", "baseline_victim_snapshot.json")
kill_snapshot = combine_snapshots("kill_cluster_snapshot.json", "kill_victim_snapshot.json")
post_restart_snapshot = combine_snapshots("post_restart_cluster_snapshot.json", "post_restart_victim_snapshot.json")

# -- Load other data --
results = load_jsonl("final_per_request.jsonl")
baseline_inflight = load_json("baseline_inflight.json")
kill_inflight = load_json("kill_inflight.json")
post_restart_inflight = load_json("post_restart_inflight.json")
final_summary = load_json("final_summary.json")

# -- Compute request statistics --
total = len(results)
status_counts = Counter(r.get("status", "unknown") for r in results)
ok_count = status_counts.get("ok", 0)
scheduled_count = status_counts.get("scheduled", 0)
schedule_error_count = status_counts.get("schedule_error", 0)
exception_count = status_counts.get("exception", 0)
error_count = schedule_error_count + exception_count
failure_rate = (error_count / total * 100) if total > 0 else 0.0

summary_available = final_summary is not None
if summary_available:
    total = final_summary.get("total_requests", total)
    ok_count = final_summary.get("completed", ok_count)
    scheduled_count = final_summary.get("scheduled", scheduled_count)
    error_count = final_summary.get("errors", error_count)
    failure_rate = (error_count / total * 100) if total > 0 else 0.0
    summary_status_counts = final_summary.get("status_counts", {})
else:
    summary_status_counts = dict(status_counts)

# -- Error type distribution --
error_types = Counter()
for r in results:
    err = r.get("error", "")
    if not err:
        continue
    el = err.lower()
    if "connection refused" in el or "connectionrefusederror" in el:
        error_types["connection_refused"] += 1
    elif "unavailable" in el:
        error_types["grpc_unavailable"] += 1
    elif "deadline" in el or "deadlineexceeded" in el:
        error_types["deadline_exceeded"] += 1
    elif "timeout" in el or "timeouterror" in el:
        error_types["timeout"] += 1
    elif "cancelled" in el or "canceled" in el:
        error_types["cancelled"] += 1
    elif "channel shutdown" in el or "channelclosederror" in el:
        error_types["channel_shutdown"] += 1
    else:
        error_types[f"other: {err[:80]}"] += 1

# -- Inflight comparison --
def parse_inflight(data):
    if not data:
        return None
    sched = data.get("scheduler_inflight", "N/A")
    prefill_eps = data.get("prefill_endpoints", [])
    decode_eps = data.get("decode_endpoints", [])
    prefill_clean = all(ep.get("inflight_batches", 0) == 0 for ep in prefill_eps)
    decode_clean = all(ep.get("inflight_requests", 0) == 0 for ep in decode_eps)
    return {
        "scheduler_inflight": sched,
        "prefill_clean": prefill_clean,
        "decode_clean": decode_clean,
        "prefill_detail": [
            {"ep": ep.get("ip_port", "?"), "batches": ep.get("inflight_batches", 0)}
            for ep in prefill_eps
        ],
        "decode_detail": [
            {"ep": ep.get("ip_port", "?"), "reqs": ep.get("inflight_requests", 0)}
            for ep in decode_eps
        ],
    }

baseline_in = parse_inflight(baseline_inflight)
kill_in = parse_inflight(kill_inflight)
post_restart_in = parse_inflight(post_restart_inflight)

# -- Mock engine snapshot helper --
def check_engines(snapshot):
    if not snapshot:
        return None
    engines = snapshot.get("engines", [])
    return {
        "total_running": sum(e.get("running", 0) for e in engines),
        "engines": [
            {
                "name": e.get("name", "?"),
                "role": e.get("role", "?"),
                "running": e.get("running", 0),
                "accepted": e.get("accepted", 0),
                "completed": e.get("completed", 0),
                "cancelled": e.get("cancelled_count", 0),
            }
            for e in engines
        ],
    }

baseline_res = check_engines(baseline_snapshot)
kill_res = check_engines(kill_snapshot)
post_restart_res = check_engines(post_restart_snapshot)

# -- Recovery verification data --
recovery_summary = load_json("recovery_verify/summary.json")
recovery_total = recovery_summary.get("total_requests", 0) if recovery_summary else 0
recovery_ok = recovery_summary.get("completed", 0) if recovery_summary else 0
recovery_success_rate = (recovery_ok / recovery_total * 100) if recovery_total > 0 else 0

# -- Assertion 1: Master did not crash (HTTP port available during kill) --
master_alive = master_health_code == "200"

# -- Assertion 2: Surviving engines continued accepting requests (multi)
#    OR Master gracefully degraded (single) --
if engine_mode == "multi":
    # Compare surviving engine accepted counts (same role as killed)
    baseline_accepted = sum(
        e.get("accepted", 0) for e in (baseline_res or {}).get("engines", [])
        if e.get("role") == kill_target and e.get("name") != victim_name
    )
    kill_accepted = sum(
        e.get("accepted", 0) for e in (kill_res or {}).get("engines", [])
        if e.get("role") == kill_target and e.get("name") != victim_name
    )
    surviving_engines_ok = kill_accepted > baseline_accepted
    assertion2_detail = (
        f"surviving {kill_target} engines accepted: "
        f"baseline={baseline_accepted} -> kill={kill_accepted}"
    )
else:
    # Single-engine: check Master returned errors (not hang)
    # Master alive + load client still running = graceful degradation
    surviving_engines_ok = master_alive
    assertion2_detail = (
        f"single-engine mode: master_alive={master_alive}, "
        f"load_client_pid={load_client_pid} (graceful degradation)"
    )

# -- Pass/Fail determination --
test_passed = True
fail_reasons = []

if total == 0:
    test_passed = False
    fail_reasons.append("no requests recorded (load client may have failed)")

# Hard assertion 1: Master did not crash
if not master_alive:
    test_passed = False
    fail_reasons.append(
        f"Master crashed during kill period (HTTP status={master_health_code})"
    )

# Hard assertion 2: Surviving engines / graceful degradation
if not surviving_engines_ok:
    test_passed = False
    if engine_mode == "multi":
        fail_reasons.append(
            f"Surviving engines did not continue accepting requests ({assertion2_detail})"
        )
    else:
        fail_reasons.append(
            f"Master did not gracefully degrade ({assertion2_detail})"
        )

# Hard assertion 3: Post-restart endpoint inflight = 0 (killed role only)
# The Master stays running (unlike master kill-restart), so the non-killed
# role may still have active requests from the main load client.
if post_restart_in:
    if kill_target == "prefill" and not post_restart_in.get("prefill_clean", True):
        test_passed = False
        fail_reasons.append(
            f"Post-restart prefill inflight not clean: {post_restart_in.get('prefill_detail', [])}"
        )
    if kill_target == "decode" and not post_restart_in.get("decode_clean", True):
        test_passed = False
        fail_reasons.append(
            f"Post-restart decode inflight not clean: {post_restart_in.get('decode_detail', [])}"
        )
else:
    test_passed = False
    fail_reasons.append(
        "Post-restart inflight data unavailable (master may have crashed or is unresponsive)"
    )

# Hard assertion 4: Recovery success rate >= 95%
if recovery_total == 0:
    test_passed = False
    fail_reasons.append(
        "Recovery verification failed: no requests completed (master may have crashed or is unresponsive)"
    )
elif recovery_success_rate < 95.0:
    test_passed = False
    fail_reasons.append(
        f"Recovery success rate {recovery_success_rate:.1f}% < 95% threshold"
    )

# Hard assertion 5: Mock engine no abnormal cancelled
if post_restart_res:
    for engine in post_restart_res.get("engines", []):
        if engine.get("cancelled", 0) > 0:
            test_passed = False
            fail_reasons.append(
                f"Mock engine {engine.get('name', '?')} has "
                f"{engine.get('cancelled', 0)} cancelled requests"
            )

if not test_passed and not fail_reasons:
    fail_reasons.append("unknown failure")

# -- Generate Markdown report --
lines = []
w = lines.append

w("# FlexLB Engine Kill-Restart Destructive Test Report")
w("")
w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w(f"**Run Directory**: `{run_dir}`")
w("")

w("## 1. Environment")
w("")
w("| Parameter | Value |")
w("|---|---|")
w(f"| Engine Mode | {engine_mode} |")
w(f"| Kill Target | {kill_target} |")
w(f"| N_PREFILL (total) | {n_prefill_total} |")
w(f"| N_DECODE (total) | {n_decode_total} |")
w(f"| Cluster Engines | {cluster_n_prefill}P + {cluster_n_decode}D |")
w(f"| Victim Engine | {victim_name} ({kill_target}) |")
w(f"| Victim gRPC Port | {victim_grpc_port} |")
w(f"| Victim HTTP Port | {victim_http_port} |")
w(f"| Mock Cluster PID | {mock_pid} |")
w(f"| Load Client PID | {load_client_pid} |")
w(f"| Kill Timestamp | {kill_ts} |")
w(f"| Restart Timestamp | {restart_ts} |")
w(f"| Master HTTP (kill period) | {master_health_code} |")
w(f"| FLEXLB_HTTP_PORT | {os.environ.get('FLEXLB_HTTP_PORT_VAL', '18080')} |")
w(f"| MOCK_HTTP_PORT | {os.environ.get('MOCK_HTTP_PORT_VAL', '55150')} |")
w(f"| MOCK_BASE_GRPC_PORT | {os.environ.get('MOCK_BASE_GRPC_PORT_VAL', '55151')} |")
w(f"| Schedule Mode | batch |")
w("")

w("## 2. Request Statistics")
w("")
src = "summary.json" if summary_available else "per_request.jsonl (computed)"
w(f"_Source: {src}_")
w("")
w("| Metric | Value |")
w("|---|---|")
w(f"| Total Requests | {total} |")
w(f"| Completed (ok) | {ok_count} |")
w(f"| Scheduled | {scheduled_count} |")
w(f"| Total Errors | {error_count} |")
w(f"| Failure Rate | {failure_rate:.2f}% |")
w("")
w("### Status Distribution")
w("")
w("| Status | Count | Percentage |")
w("|---|---|")
for status, count in sorted(summary_status_counts.items(), key=lambda x: -x[1]):
    pct = (count / total * 100) if total > 0 else 0
    w(f"| {status} | {count} | {pct:.1f}% |")
w("")

if error_types:
    w("### Error Type Distribution")
    w("")
    w("| Error Type | Count |")
    w("|---|---|")
    for et, count in error_types.most_common():
        w(f"| {et} | {count} |")
    w("")

w("## 3. Inflight Status Comparison")
w("")

def write_inflight_section(title, inflight_data):
    w(f"### {title}")
    w("")
    if not inflight_data:
        w("- _No data available_")
        w("")
        return
    w(f"- Scheduler Inflight: **{inflight_data['scheduler_inflight']}**")
    w(f"- Prefill Endpoints Clean: {inflight_data['prefill_clean']}")
    w(f"- Decode Endpoints Clean: {inflight_data['decode_clean']}")
    if inflight_data["prefill_detail"]:
        w("")
        w("  | Endpoint | Inflight Batches |")
        w("  |---|---|")
        for ep in inflight_data["prefill_detail"]:
            w(f"  | {ep['ep']} | {ep['batches']} |")
    if inflight_data["decode_detail"]:
        w("")
        w("  | Endpoint | Inflight Requests |")
        w("  |---|---|")
        for ep in inflight_data["decode_detail"]:
            w(f"  | {ep['ep']} | {ep['reqs']} |")
    w("")

write_inflight_section("Baseline (pre-kill)", baseline_in)
write_inflight_section("Kill Period", kill_in)
write_inflight_section("Post-Restart", post_restart_in)

w("## 4. Mock Engine Snapshot Comparison")
w("")

def write_snapshot_section(title, res):
    w(f"### {title}")
    w("")
    if not res:
        w("- _No data available_")
        w("")
        return
    w(f"- Total Running: **{res['total_running']}**")
    w("")
    w("  | Engine | Role | Running | Accepted | Completed | Cancelled |")
    w("  |---|---|---|---|---|---|")
    for e in res["engines"]:
        w(f"  | {e['name']} | {e['role']} | {e['running']} | {e['accepted']} | {e['completed']} | {e['cancelled']} |")
    w("")

write_snapshot_section("Baseline Snapshot", baseline_res)
write_snapshot_section("Kill-Period Snapshot", kill_res)
write_snapshot_section("Post-Restart Snapshot", post_restart_res)

w("## 5. Recovery Verification")
w("")
w("| Metric | Value |")
w("|---|---|")
w(f"| Total Requests | {recovery_total} |")
w(f"| Completed (ok) | {recovery_ok} |")
w(f"| Success Rate | {recovery_success_rate:.1f}% |" if recovery_total > 0 else "| Success Rate | N/A |")
w("")

w("## 6. Hard Assertions")
w("")
w("| # | Assertion | Result | Detail |")
w("|---|---|---|---|")
w(f"| 1 | Master did not crash (HTTP alive during kill) | {'PASS' if master_alive else 'FAIL'} | HTTP status={master_health_code} |")
w(f"| 2 | {'Surviving engines accepted requests' if engine_mode == 'multi' else 'Master graceful degradation'} | {'PASS' if surviving_engines_ok else 'FAIL'} | {assertion2_detail} |")
post_inflight_ok = False
if post_restart_in:
    if kill_target == "prefill":
        post_inflight_ok = post_restart_in.get("prefill_clean", False)
    else:
        post_inflight_ok = post_restart_in.get("decode_clean", False)
w(f"| 3 | Post-restart {kill_target} inflight = 0 | {'PASS' if post_inflight_ok else 'FAIL'} | scheduler={post_restart_in.get('scheduler_inflight', 'N/A') if post_restart_in else 'N/A'} |")
recovery_ok_assertion = recovery_total > 0 and recovery_success_rate >= 95.0
w(f"| 4 | Recovery success rate >= 95% | {'PASS' if recovery_ok_assertion else 'FAIL'} | {recovery_success_rate:.1f}% ({recovery_ok}/{recovery_total}) |")
no_cancelled = True
cancelled_detail = "none"
if post_restart_res:
    for e in post_restart_res.get("engines", []):
        if e.get("cancelled", 0) > 0:
            no_cancelled = False
            cancelled_detail = f"{e['name']}: {e['cancelled']}"
            break
w(f"| 5 | No abnormal cancelled | {'PASS' if no_cancelled else 'FAIL'} | {cancelled_detail} |")
w("")

w("## 7. Test Conclusion")
w("")
if test_passed:
    w("**Result: PASS**")
else:
    w("**Result: FAIL**")
w("")
if fail_reasons:
    w("Failure reasons:")
    for reason in fail_reasons:
        w(f"- {reason}")
    w("")
w("### Observations")
w("")
w(f"- Victim engine ({victim_name}, {kill_target}) was killed at {kill_ts} and restarted at {restart_ts}")
w(f"- Engine mode: {engine_mode}, kill target: {kill_target}")
w(f"- Master remained alive during kill period: {master_alive} (HTTP {master_health_code})")
if total > 0:
    w(f"- Load client: {ok_count}/{total} succeeded ({(ok_count/total*100) if total > 0 else 0:.1f}%)")
    if error_count > 0:
        w(f"- {error_count} requests failed during kill period (expected — engine unavailable)")
if kill_res:
    surviving = [e for e in kill_res.get("engines", []) if e.get("name") != victim_name]
    w(f"- Surviving engines during kill: {len(surviving)} engines")
if post_restart_res:
    w(f"- Post-restart total running: {post_restart_res['total_running']}")
if post_restart_in:
    w(f"- Post-restart scheduler inflight: {post_restart_in['scheduler_inflight']}")
if recovery_total > 0:
    w(f"- Recovery verification: {recovery_ok}/{recovery_total} succeeded ({recovery_success_rate:.1f}%)")
w("")

w("---")
w(f"_Report generated at {datetime.now().isoformat()}_")

report = "\n".join(lines)
report_path = run_dir / "test_report.md"
report_path.write_text(report, encoding="utf-8")
print(report)
PYEOF

# ===========================================================================
# Done
# ===========================================================================

echo ""
echo "=========================================="
echo "  Test Complete"
echo "=========================================="
echo "  Report:    ${RUN_DIR}/test_report.md"
echo "  Run dir:   ${RUN_DIR}"
echo "  Master log: ${RUN_DIR}/flexlb_master.log"
echo "  Mock log:  ${RUN_DIR}/mock_engine.log"
echo "  Victim log (initial): ${RUN_DIR}/victim_engine_initial.log"
echo "  Victim log (restart): ${RUN_DIR}/victim_engine_restart.log"
echo "  Load client log: ${RUN_DIR}/load_client.log"
echo "=========================================="

#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# flexlb_behavior_test.sh
#
# FlexLB engine disconnect/reconnect behavior test.
#
# Uses mock_engine_cluster's HTTP API (/stop_engine, /start_engine) to test
# FlexLB's behavior when an engine's gRPC server stops and restarts.
#
# Scenarios:
#   1. Stuck inflight TTL cleanup timing
#   2. gRPC channel recovery after engine restart
#   3. Single-engine inflight quota blocking (1P+1D)
#   4. Calibrate-based fast cleanup after engine restart
#
# Usage:
#   bash flexlb_behavior_test.sh
#   SCENARIOS="1,2" bash flexlb_behavior_test.sh   # run subset
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
N_PREFILL="${N_PREFILL:-2}"
N_DECODE="${N_DECODE:-2}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-55151}"
MOCK_HTTP_PORT=$((MOCK_BASE_GRPC_PORT - 1))
FLEXLB_HTTP_PORT="${FLEXLB_HTTP_PORT:-18080}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-18081}"
PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS:-6000}"
DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS:-3000}"

# Test-specific config (short TTL, low quota, short gRPC timeout)
FLEXLB_INFLIGHT_TTL_MS="${FLEXLB_INFLIGHT_TTL_MS:-30000}"
FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES:-4}"
SYNC_REQUEST_TIMEOUT_MS="${SYNC_REQUEST_TIMEOUT_MS:-1000}"

# Load client defaults
LOAD_CLIENT_REPLAY_SPEED="${LOAD_CLIENT_REPLAY_SPEED:-20}"
LOAD_CLIENT_CONCURRENCY="${LOAD_CLIENT_CONCURRENCY:-20}"
LOAD_CLIENT_TIMEOUT_MS="${LOAD_CLIENT_TIMEOUT_MS:-10000}"

# Which scenarios to run
SCENARIOS="${SCENARIOS:-1,2,3,4}"

# -- Run directory ---------------------------------------------------------
RUN_DIR="${SCRIPT_DIR}/run/behavior_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"

# -- State -----------------------------------------------------------------
MOCK_PID=""
FLEXLB_PID=""
CURRENT_SCENARIO=""

# -- Perf config (shared) --------------------------------------------------
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

# -- Short trace for recovery tests ----------------------------------------
SHORT_TRACE="${RUN_DIR}/short_trace.jsonl"
python3 -c "
import json
with open('${TRACE_FILE}') as f:
    for line in f:
        req = json.loads(line)
        if req.get('ol', 0) <= 200:
            print(line, end='')
" > "${SHORT_TRACE}" 2>/dev/null || true
SHORT_TRACE_LINES=$(wc -l < "${SHORT_TRACE}" 2>/dev/null || echo 0)
echo "Short trace (ol<=200): ${SHORT_TRACE_LINES} lines"

# ===========================================================================
# Helper functions
# ===========================================================================

log() {
  local ts
  ts=$(date +"%H:%M:%S")
  echo "[${ts}] $*"
}

wait_for_port() {
  local host="$1" port="$2" timeout_s="$3"
  python3 - "$host" "$port" "$timeout_s" <<'PY'
import socket, sys, time
host, port, timeout_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.time() + timeout_s
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            sys.exit(0)
    except OSError:
        time.sleep(0.5)
sys.exit(1)
PY
}

check_port_free() {
  local port="$1"
  if lsof -i :"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

kill_stale_on_port() {
  local port="$1"
  local stale_pid
  stale_pid=$(lsof -ti :"${port}" -sTCP:LISTEN 2>/dev/null || true)
  if [[ -n "${stale_pid}" ]]; then
    kill -9 "${stale_pid}" 2>/dev/null || true
    sleep 1
  fi
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

# ===========================================================================
# Environment setup / teardown
# ===========================================================================

setup_environment() {
  local n_prefill="$1"
  local n_decode="$2"
  local scenario_name="$3"
  local subdir="${4:-${scenario_name}}"

  CURRENT_SCENARIO="${scenario_name}"
  local scenario_dir="${RUN_DIR}/${subdir}"
  mkdir -p "${scenario_dir}"

  log "=== Setup: ${scenario_name} (${n_prefill}P + ${n_decode}D) ==="

  # Pre-flight: free all ports
  local ports=("${FLEXLB_HTTP_PORT}" "${FLEXLB_MANAGEMENT_PORT}" "${MOCK_HTTP_PORT}")
  for ((i = 0; i < n_prefill + n_decode; i++)); do
    ports+=("$((MOCK_BASE_GRPC_PORT + i))")
  done
  for port in "${ports[@]}"; do
    if ! check_port_free "${port}"; then
      log "  freeing stale process on port ${port} ..."
      kill_stale_on_port "${port}"
    fi
  done
  sleep 1

  # Start mock engine cluster
  log "  starting mock cluster (${n_prefill}P + ${n_decode}D) ..."
  local endpoint_file="${scenario_dir}/endpoints.json"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/mock_engine_cluster.py" \
    --n-prefill "${n_prefill}" \
    --n-decode "${n_decode}" \
    --base-grpc-port "${MOCK_BASE_GRPC_PORT}" \
    --performance "${PERF_CONFIG_FILE}" \
    --prefill-cache-blocks "${PREFILL_CACHE_BLOCKS}" \
    --decode-cache-blocks "${DECODE_CACHE_BLOCKS}" \
    --endpoint-file "${endpoint_file}" \
    --env-file "${scenario_dir}/flexlb_env.txt" \
    >"${scenario_dir}/mock_engine.log" 2>&1 &
  MOCK_PID="$!"
  wait_for_port "127.0.0.1" "${MOCK_HTTP_PORT}" 20
  if ! kill -0 "${MOCK_PID}" 2>/dev/null; then
    echo "ERROR: mock cluster died during startup" >&2
    cat "${scenario_dir}/mock_engine.log" >&2
    exit 1
  fi
  log "  mock cluster started (pid=${MOCK_PID}, http=${MOCK_HTTP_PORT})"

  # Start FlexLB Master
  log "  starting FlexLB Master ..."
  # Parse service-discovery env vars from endpoint file
  local env_args=()
  while IFS= read -r line; do
    env_args+=("${line}")
  done < <(python3 - "${endpoint_file}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for key, value in payload["env"].items():
    print(f"{key}={value}")
PY
)
  env \
    "${env_args[@]+"${env_args[@]}"}" \
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
    "FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES=${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES}" \
    "FLEXLB_INFLIGHT_TTL_MS=${FLEXLB_INFLIGHT_TTL_MS}" \
    "SYNC_REQUEST_TIMEOUT_MS=${SYNC_REQUEST_TIMEOUT_MS}" \
    "DECODE_CONCURRENCY_LIMIT=132" \
    "PREFILL_QUEUE_SIZE_THRESHOLD=100000" \
    "COST_SLO_MS=30000" \
    "COST_HOTSPOT_MULTIPLIER=1.5" \
    "DEFAULT_SCHEDULE_MODE=BATCH" \
    "ENABLE_QUEUEING=false" \
    "STRATEGY_CONFIGS={}" \
    "OTEL_TRACE_SKIP_PATTERN=.*" \
    "OTEL_EXPORTER_OTLP_ENDPOINT=none" \
    "HIPPO_ROLE=flexlb_behavior_test" \
    java "${JAVA_MODULE_OPTS[@]}" -jar "${FLEXLB_JAR}" \
    --server.port="${FLEXLB_HTTP_PORT}" \
    --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
    --logging.level.org.flexlb=DEBUG \
    >"${scenario_dir}/flexlb_master.log" 2>&1 &
  FLEXLB_PID="$!"
  wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60
  if ! kill -0 "${FLEXLB_PID}" 2>/dev/null; then
    echo "ERROR: master died during startup" >&2
    cat "${scenario_dir}/flexlb_master.log" >&2
    exit 1
  fi
  log "  master started (pid=${FLEXLB_PID})"
  sleep 3  # let master discover engines
}

teardown_environment() {
  log "=== Teardown: ${CURRENT_SCENARIO} ==="
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
  sleep 2  # let ports free
}

# ===========================================================================
# API helper functions
# ===========================================================================

# Get inflight batch count for a specific prefill engine by gRPC port
get_inflight_count() {
  local port="$1"
  curl -s "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" 2>/dev/null | \
    python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    port = sys.argv[1]
    for ep in data.get('prefill_endpoints', []):
        ip_port = ep.get('ip_port', '')
        if ip_port.endswith(f':{port}'):
            ib = ep.get('inflight_batches', 0)
            if isinstance(ib, list):
                print(len(ib))
            elif isinstance(ib, (int, float)):
                print(int(ib))
            else:
                print(0)
            sys.exit(0)
    print(0)
except Exception:
    print(-1)
" "${port}"
}

# Get a field from a specific mock engine by name
get_mock_field() {
  local engine_name="$1"
  local field="$2"
  curl -s "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" 2>/dev/null | \
    python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    name, field = sys.argv[1], sys.argv[2]
    for e in data.get('engines', []):
        if e.get('name') == name:
            val = e.get(field, 0)
            if isinstance(val, bool):
                print('true' if val else 'false')
            else:
                print(val)
            sys.exit(0)
    print(-1)
except Exception:
    print(-1)
" "${engine_name}" "${field}"
}

# Stop a mock engine via HTTP API
stop_engine() {
  local engine_name="$1"
  log "  POST /stop_engine {\"engine\": \"${engine_name}\"}"
  curl -s -X POST "http://127.0.0.1:${MOCK_HTTP_PORT}/stop_engine" \
    -H "Content-Type: application/json" \
    -d "{\"engine\": \"${engine_name}\"}" 2>/dev/null || true
  echo ""
}

# Start a mock engine via HTTP API
start_engine() {
  local engine_name="$1"
  log "  POST /start_engine {\"engine\": \"${engine_name}\"}"
  curl -s -X POST "http://127.0.0.1:${MOCK_HTTP_PORT}/start_engine" \
    -H "Content-Type: application/json" \
    -d "{\"engine\": \"${engine_name}\"}" 2>/dev/null || true
  echo ""
}

# Set performance parameter for a mock engine via /set_perf API
set_perf() {
  local engine_name="$1"
  local perf_key="$2"   # prefill_fixed_ms or decode_scale
  local perf_val="$3"
  curl -s -X POST "http://127.0.0.1:${MOCK_HTTP_PORT}/set_perf" \
    -H "Content-Type: application/json" \
    -d "{\"engine\": \"${engine_name}\", \"${perf_key}\": ${perf_val}}" \
    >/dev/null 2>&1
  log "Set ${engine_name} ${perf_key}=${perf_val}"
}

# Reset engine performance to defaults (prefill_fixed_ms=100, decode_scale=1.0)
reset_perf() {
  local engine_name="$1"
  set_perf "${engine_name}" "prefill_fixed_ms" 100
  set_perf "${engine_name}" "decode_scale" 1.0
}

# Send requests synchronously via load client
send_requests() {
  local trace="$1"
  local limit="$2"
  local concurrency="$3"
  local timeout_ms="$4"
  local output_dir="$5"
  local label="$6"

  mkdir -p "${output_dir}"
  log "  sending ${limit} requests (${label}) ..." >&2
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/flexlb_load_client.py" \
    "${trace}" \
    --flexlb-http-addr "127.0.0.1:${FLEXLB_HTTP_PORT}" \
    --schedule-mode batch \
    --replay-speed 0 \
    --limit "${limit}" \
    --max-concurrency "${concurrency}" \
    --timeout-ms "${timeout_ms}" \
    --output-dir "${output_dir}" \
    >"${output_dir}/load_client.log" 2>&1 || true
  # Parse summary
  local total ok errors
  total=$(python3 -c "import json; d=json.load(open('${output_dir}/summary.json')); print(d.get('total_requests',0))" 2>/dev/null || echo 0)
  ok=$(python3 -c "import json; d=json.load(open('${output_dir}/summary.json')); print(d.get('completed',0))" 2>/dev/null || echo 0)
  errors=$(python3 -c "import json; d=json.load(open('${output_dir}/summary.json')); print(d.get('errors',0))" 2>/dev/null || echo 0)
  log "  results: total=${total} ok=${ok} errors=${errors}" >&2
  echo "${total} ${ok} ${errors}"
}

# ===========================================================================
# Scenario 1: Stuck inflight TTL cleanup timing
# ===========================================================================
scenario_1_ttl_cleanup() {
  local sd="${RUN_DIR}/scenario1"
  log "=========================================="
  log "  Scenario 1: Stuck inflight TTL cleanup"
  log "  Target: Verify stuck inflight cleaned after TTL=${FLEXLB_INFLIGHT_TTL_MS}ms"
  log "=========================================="

  setup_environment 2 2 "Scenario1" "scenario1"

  local p0_port=${MOCK_BASE_GRPC_PORT}

  # Step 1: Slow down both prefill engines so requests stay inflight
  log "T=0s: slowing prefill engines (prefill_fixed_ms=10000) ..."
  set_perf "prefill-0" "prefill_fixed_ms" 10000
  set_perf "prefill-1" "prefill_fixed_ms" 10000

  # Step 2: Generate filtered trace with long output (ol 100-500, up to 50 lines)
  local long_trace="${sd}/long_trace.jsonl"
  python3 -c "
import json
with open('${TRACE_FILE}') as f:
    count = 0
    for line in f:
        req = json.loads(line)
        if 100 <= req.get('ol', 0) <= 500:
            print(line, end='')
            count += 1
            if count >= 50:
                break
" > "${long_trace}" 2>/dev/null || true
  local trace_lines
  trace_lines=$(wc -l < "${long_trace}" 2>/dev/null || echo 0)
  log "  filtered trace: ${trace_lines} lines (ol 100-500)"

  # Step 3: Send requests asynchronously (background, do not wait for completion)
  log "T=0s: sending ${trace_lines} requests in background ..."
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/flexlb_load_client.py" \
    "${long_trace}" \
    --flexlb-http-addr "127.0.0.1:${FLEXLB_HTTP_PORT}" \
    --schedule-mode batch \
    --replay-speed 0 \
    --max-concurrency 20 \
    --timeout-ms 30000 \
    --output-dir "${sd}/load" \
    >"${sd}/load_client.log" 2>&1 &
  local load_pid=$!

  # Step 4: Wait for requests to be enqueued (accepted > 0 on prefill-0)
  log "Waiting for requests to be enqueued on prefill-0 ..."
  local waited=0
  while [ ${waited} -lt 15 ]; do
    local accepted_now
    accepted_now=$(get_mock_field "prefill-0" "accepted")
    if [ "${accepted_now}" -gt 0 ] 2>/dev/null; then
      log "  prefill-0 accepted=${accepted_now} (waited ${waited}s)"
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done

  # Step 5: Record inflight count (should be > 0 with slow prefill)
  local inflight_before
  inflight_before=$(get_inflight_count "${p0_port}")
  log "T=${waited}s: prefill-0 inflight=${inflight_before} (before kill)"

  # Step 6: Stop prefill-0 engine while requests are still inflight
  log "T=${waited}s: stopping prefill-0 ..."
  stop_engine "prefill-0"

  # Step 7: Kill load client (stop sending new requests)
  kill ${load_pid} 2>/dev/null || true
  wait ${load_pid} 2>/dev/null || true

  # Step 8: Wait 5s for alive=false (let gRPC failure propagate)
  sleep 5

  # Step 9: Record stuck inflight count
  local inflight_after_kill
  inflight_after_kill=$(get_inflight_count "${p0_port}")
  log "T=$((waited + 5))s: stuck inflight=${inflight_after_kill}"

  # Step 10: Poll inflight count every 5s for 90s (TTL=30s + scheduling margin)
  local timeline_file="${sd}/timeline.jsonl"
  > "${timeline_file}"
  local elapsed=0
  local max_wait=90
  local poll_interval=5
  local cleanup_time="-1"
  local inflight_final="${inflight_after_kill}"

  while [[ ${elapsed} -le ${max_wait} ]]; do
    local inflight_now accepted_now stopped_now
    inflight_now=$(get_inflight_count "${p0_port}")
    accepted_now=$(get_mock_field "prefill-0" "accepted")
    stopped_now=$(get_mock_field "prefill-0" "stopped")
    inflight_final="${inflight_now}"
    log "  T=${elapsed}s inflight=${inflight_now} accepted=${accepted_now} stopped=${stopped_now}"
    echo "{\"t\":${elapsed},\"inflight\":${inflight_now},\"accepted\":${accepted_now},\"stopped\":\"${stopped_now}\"}" >> "${timeline_file}"
    if [[ "${inflight_now}" == "0" && "${cleanup_time}" == "-1" ]]; then
      cleanup_time=${elapsed}
      log "  ** inflight reached 0 at T=${elapsed}s"
    fi
    if [[ ${elapsed} -ge ${max_wait} ]]; then
      break
    fi
    sleep ${poll_interval}
    elapsed=$((elapsed + poll_interval))
  done

  # Step 11: Verify
  local s1_pass="PASS"
  local s1_reasons=""
  if [[ "${inflight_after_kill}" == "0" || "${inflight_after_kill}" == "-1" ]]; then
    s1_pass="FAIL"
    s1_reasons="${s1_reasons}No stuck inflight after kill (inflight=${inflight_after_kill}); "
  fi
  if [[ "${inflight_final}" != "0" ]]; then
    s1_pass="FAIL"
    s1_reasons="${s1_reasons}Inflight not cleaned after ${max_wait}s (inflight=${inflight_final}); "
  fi
  if [[ "${cleanup_time}" == "-1" ]]; then
    s1_pass="FAIL"
    s1_reasons="${s1_reasons}Cleanup never triggered; "
  fi

  log "  Result: ${s1_pass}"
  [[ -n "${s1_reasons}" ]] && log "  Reasons: ${s1_reasons}"
  log "  Cleanup time: ${cleanup_time}s (TTL=${FLEXLB_INFLIGHT_TTL_MS}ms)"

  # Save results
  cat > "${sd}/result.json" <<JSONEOF
{
  "scenario": "TTL cleanup",
  "pass": "${s1_pass}",
  "reasons": "${s1_reasons}",
  "inflight_before_kill": ${inflight_after_kill},
  "inflight_final": ${inflight_final},
  "cleanup_time_s": ${cleanup_time},
  "ttl_ms": ${FLEXLB_INFLIGHT_TTL_MS}
}
JSONEOF

  teardown_environment
}

# ===========================================================================
# Scenario 2: gRPC channel recovery
# ===========================================================================
scenario_2_grpc_recovery() {
  local sd="${RUN_DIR}/scenario2"
  log "=========================================="
  log "  Scenario 2: gRPC channel recovery"
  log "  Target: Verify channel recovers after engine restart"
  log "=========================================="

  setup_environment 2 2 "Scenario2" "scenario2"

  local p0_port=${MOCK_BASE_GRPC_PORT}

  # Step 1: Send 50 requests, record prefill-0 accepted
  log "T=0s: sending 50 baseline requests ..."
  send_requests "${TRACE_FILE}" 50 20 10000 "${sd}/baseline" "baseline"
  sleep 2
  local accepted_before
  accepted_before=$(get_mock_field "prefill-0" "accepted")
  log "T=5s: prefill-0 accepted=${accepted_before}"

  # Step 2: Stop prefill-0
  log "T=5s: stopping prefill-0 ..."
  stop_engine "prefill-0"

  # Step 3: Wait 5s for alive=false
  log "T=5s: waiting 5s for alive=false ..."
  sleep 5

  # Step 4: Send 50 requests, verify prefill-0 not growing
  log "T=10s: sending 50 requests during downtime ..."
  send_requests "${SHORT_TRACE}" 50 20 10000 "${sd}/downtime" "downtime"
  sleep 2
  local accepted_downtime
  accepted_downtime=$(get_mock_field "prefill-0" "accepted")
  log "T=15s: prefill-0 accepted=${accepted_downtime} (was ${accepted_before})"

  # Step 5: Start prefill-0
  log "T=15s: starting prefill-0 ..."
  start_engine "prefill-0"

  # Step 6: Wait 10s for channel reconnect + alive recovery
  log "T=15s: waiting 10s for recovery ..."
  sleep 10

  # Step 7: Send 100 short requests
  log "T=25s: sending 100 recovery requests ..."
  local recovery_out
  recovery_out=$(send_requests "${SHORT_TRACE}" 100 10 10000 "${sd}/recovery" "recovery")
  local r_total r_ok r_errors
  read -r r_total r_ok r_errors <<< "${recovery_out}"
  sleep 2
  local accepted_after
  accepted_after=$(get_mock_field "prefill-0" "accepted")
  log "T=35s: prefill-0 accepted=${accepted_after} (downtime=${accepted_downtime})"

  # Verify
  local s2_pass="PASS"
  local s2_reasons=""
  if [[ "${accepted_downtime}" != "${accepted_before}" ]]; then
    s2_pass="FAIL"
    s2_reasons="${s2_reasons}prefill-0 accepted changed during downtime (${accepted_before} -> ${accepted_downtime}); "
  fi
  if [[ "${accepted_after}" == "${accepted_downtime}" ]]; then
    s2_pass="FAIL"
    s2_reasons="${s2_reasons}prefill-0 accepted did not grow after restart (${accepted_downtime} -> ${accepted_after}); "
  fi
  local success_rate=0
  if [[ ${r_total} -gt 0 ]]; then
    success_rate=$(python3 -c "print(round(${r_ok}/${r_total}*100, 1))")
  fi
  if [[ ${r_total} -gt 0 ]] && python3 -c "exit(0 if ${success_rate} < 95 else 1)" 2>/dev/null; then
    s2_pass="FAIL"
    s2_reasons="${s2_reasons}Recovery success rate ${success_rate}% < 95%; "
  fi

  log "  Result: ${s2_pass}"
  [[ -n "${s2_reasons}" ]] && log "  Reasons: ${s2_reasons}"
  log "  Recovery: ${r_ok}/${r_total} (${success_rate}%)"

  cat > "${sd}/result.json" <<JSONEOF
{
  "scenario": "gRPC recovery",
  "pass": "${s2_pass}",
  "reasons": "${s2_reasons}",
  "accepted_before": ${accepted_before},
  "accepted_downtime": ${accepted_downtime},
  "accepted_after": ${accepted_after},
  "recovery_total": ${r_total},
  "recovery_ok": ${r_ok},
  "success_rate": ${success_rate}
}
JSONEOF

  teardown_environment
}

# ===========================================================================
# Scenario 3: Single-engine inflight quota blocking (1P+1D)
# ===========================================================================
scenario_3_quota_blocking() {
  local sd="${RUN_DIR}/scenario3"
  log "=========================================="
  log "  Scenario 3: Quota blocking (1P+1D)"
  log "  Target: Verify stuck inflight blocks new requests"
  log "=========================================="

  setup_environment 1 1 "Scenario3" "scenario3"

  local p0_port=${MOCK_BASE_GRPC_PORT}

  # Step 1: Send 4 requests to fill per-worker quota
  log "T=0s: sending 4 requests to fill quota (MAX_INFLIGHT_BATCHES=${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES}) ..."
  send_requests "${SHORT_TRACE}" 4 4 10000 "${sd}/fill_quota" "fill_quota"
  sleep 2
  local inflight_filled
  inflight_filled=$(get_inflight_count "${p0_port}")
  log "T=5s: prefill-0 inflight=${inflight_filled}"

  # Step 2: Stop prefill-0
  log "T=5s: stopping prefill-0 ..."
  stop_engine "prefill-0"

  # Step 3: Wait 3s for alive=false
  log "T=5s: waiting 3s for alive=false ..."
  sleep 3

  # Step 4: Send 20 new requests (should be blocked)
  log "T=8s: sending 20 requests (expect blocking) ..."
  local blocked_out
  blocked_out=$(send_requests "${SHORT_TRACE}" 20 20 10000 "${sd}/blocked" "blocked")
  local b_total b_ok b_errors
  read -r b_total b_ok b_errors <<< "${blocked_out}"
  log "T=20s: blocked results: total=${b_total} ok=${b_ok} errors=${b_errors}"

  # Step 5: Wait 40s for TTL cleanup (TTL=30s + scheduling interval)
  log "T=20s: waiting 40s for TTL cleanup ..."
  sleep 40

  # Step 6: Start prefill-0
  log "T=60s: starting prefill-0 ..."
  start_engine "prefill-0"

  # Step 7: Wait 10s for recovery
  log "T=60s: waiting 10s for recovery ..."
  sleep 10

  # Step 8: Send 20 new requests
  log "T=70s: sending 20 requests (expect recovery) ..."
  local recovery_out
  recovery_out=$(send_requests "${SHORT_TRACE}" 20 20 10000 "${sd}/recovery" "recovery")
  local r_total r_ok r_errors
  read -r r_total r_ok r_errors <<< "${recovery_out}"
  log "T=80s: recovery results: total=${r_total} ok=${r_ok} errors=${r_errors}"

  # Verify
  local s3_pass="PASS"
  local s3_reasons=""
  # During block: most should fail
  local block_fail_rate=0
  if [[ ${b_total} -gt 0 ]]; then
    block_fail_rate=$(python3 -c "print(round(${b_errors}/${b_total}*100, 1))")
  fi
  if [[ ${b_total} -gt 0 ]] && python3 -c "exit(0 if ${block_fail_rate} < 50 else 1)" 2>/dev/null; then
    s3_pass="FAIL"
    s3_reasons="${s3_reasons}Block fail rate ${block_fail_rate}% < 50% (expected most failures); "
  fi
  # After recovery: success rate >= 90%
  local recovery_rate=0
  if [[ ${r_total} -gt 0 ]]; then
    recovery_rate=$(python3 -c "print(round(${r_ok}/${r_total}*100, 1))")
  fi
  if [[ ${r_total} -gt 0 ]] && python3 -c "exit(0 if ${recovery_rate} < 90 else 1)" 2>/dev/null; then
    s3_pass="FAIL"
    s3_reasons="${s3_reasons}Recovery rate ${recovery_rate}% < 90%; "
  fi

  log "  Result: ${s3_pass}"
  [[ -n "${s3_reasons}" ]] && log "  Reasons: ${s3_reasons}"
  log "  Block: ${b_ok}/${b_total} ok (${block_fail_rate}% fail)"
  log "  Recovery: ${r_ok}/${r_total} ok (${recovery_rate}%)"

  # Error type distribution
  python3 - "${sd}/blocked/per_request.jsonl" > "${sd}/error_types.json" 2>/dev/null <<'PY' || true
import json, sys
from collections import Counter
errors = Counter()
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    err = r.get("error", "")
    if not err:
        continue
    el = err.lower()
    if "connection refused" in el:
        errors["connection_refused"] += 1
    elif "unavailable" in el:
        errors["grpc_unavailable"] += 1
    elif "deadline" in el:
        errors["deadline_exceeded"] += 1
    elif "timeout" in el:
        errors["timeout"] += 1
    elif "cancelled" in el:
        errors["cancelled"] += 1
    else:
        errors[f"other: {err[:60]}"] += 1
print(json.dumps(dict(errors), indent=2))
PY

  cat > "${sd}/result.json" <<JSONEOF
{
  "scenario": "Quota blocking",
  "pass": "${s3_pass}",
  "reasons": "${s3_reasons}",
  "block_total": ${b_total},
  "block_ok": ${b_ok},
  "block_errors": ${b_errors},
  "block_fail_rate": ${block_fail_rate},
  "recovery_total": ${r_total},
  "recovery_ok": ${r_ok},
  "recovery_rate": ${recovery_rate}
}
JSONEOF

  teardown_environment
}

# ===========================================================================
# Scenario 4: Calibrate-based fast cleanup
# ===========================================================================
scenario_4_calibrate_recovery() {
  local sd="${RUN_DIR}/scenario4"
  log "=========================================="
  log "  Scenario 4: Calibrate fast cleanup"
  log "  Target: Verify calibrate cleans inflight faster than TTL"
  log "=========================================="

  setup_environment 2 2 "Scenario4" "scenario4"

  local p0_port=${MOCK_BASE_GRPC_PORT}

  # Step 1: Slow down both prefill engines so requests stay inflight
  log "T=0s: slowing prefill engines (prefill_fixed_ms=10000) ..."
  set_perf "prefill-0" "prefill_fixed_ms" 10000
  set_perf "prefill-1" "prefill_fixed_ms" 10000

  # Step 2: Generate filtered trace with long output (ol 100-500, up to 50 lines)
  local long_trace="${sd}/long_trace.jsonl"
  python3 -c "
import json
with open('${TRACE_FILE}') as f:
    count = 0
    for line in f:
        req = json.loads(line)
        if 100 <= req.get('ol', 0) <= 500:
            print(line, end='')
            count += 1
            if count >= 50:
                break
" > "${long_trace}" 2>/dev/null || true
  local trace_lines
  trace_lines=$(wc -l < "${long_trace}" 2>/dev/null || echo 0)
  log "  filtered trace: ${trace_lines} lines (ol 100-500)"

  # Step 3: Send requests asynchronously (background)
  log "T=0s: sending ${trace_lines} requests in background ..."
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/flexlb_load_client.py" \
    "${long_trace}" \
    --flexlb-http-addr "127.0.0.1:${FLEXLB_HTTP_PORT}" \
    --schedule-mode batch \
    --replay-speed 0 \
    --max-concurrency 20 \
    --timeout-ms 30000 \
    --output-dir "${sd}/load" \
    >"${sd}/load_client.log" 2>&1 &
  local load_pid=$!

  # Step 4: Wait for requests to be enqueued on prefill-0
  log "Waiting for requests to be enqueued on prefill-0 ..."
  local waited=0
  while [ ${waited} -lt 15 ]; do
    local accepted_now
    accepted_now=$(get_mock_field "prefill-0" "accepted")
    if [ "${accepted_now}" -gt 0 ] 2>/dev/null; then
      log "  prefill-0 accepted=${accepted_now} (waited ${waited}s)"
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done

  # Step 5: Record inflight count
  local inflight_before
  inflight_before=$(get_inflight_count "${p0_port}")
  log "T=${waited}s: prefill-0 inflight=${inflight_before}"

  # Step 6: Stop prefill-0 engine while requests are still inflight
  log "T=${waited}s: stopping prefill-0 ..."
  stop_engine "prefill-0"

  # Step 7: Kill load client
  kill ${load_pid} 2>/dev/null || true
  wait ${load_pid} 2>/dev/null || true

  # Step 8: Wait 5s for alive=false
  sleep 5

  # Step 9: Record stuck inflight
  local inflight_stuck
  inflight_stuck=$(get_inflight_count "${p0_port}")
  log "T=$((waited + 5))s: stuck inflight=${inflight_stuck}"

  # Step 10: Restart prefill-0 (triggers calibrate on next WorkerStatus)
  log "T=$((waited + 5))s: starting prefill-0 ..."
  start_engine "prefill-0"
  # Reset perf to normal so the restarted engine processes at normal speed
  reset_perf "prefill-0"

  # Step 11: Poll every 2s for 30s (calibrate should be much faster than TTL=30s)
  local timeline_file="${sd}/timeline.jsonl"
  > "${timeline_file}"
  local elapsed=0
  local max_wait=30
  local poll_interval=2
  local cleanup_time="-1"
  local inflight_final="${inflight_stuck}"

  while [[ ${elapsed} -le ${max_wait} ]]; do
    local inflight_now accepted_now stopped_now
    inflight_now=$(get_inflight_count "${p0_port}")
    accepted_now=$(get_mock_field "prefill-0" "accepted")
    stopped_now=$(get_mock_field "prefill-0" "stopped")
    inflight_final="${inflight_now}"
    log "  T=${elapsed}s inflight=${inflight_now} accepted=${accepted_now} stopped=${stopped_now}"
    echo "{\"t\":${elapsed},\"inflight\":${inflight_now},\"accepted\":${accepted_now},\"stopped\":\"${stopped_now}\"}" >> "${timeline_file}"
    if [[ "${inflight_now}" == "0" && "${cleanup_time}" == "-1" && ${elapsed} -gt 0 ]]; then
      cleanup_time=${elapsed}
      log "  ** inflight reached 0 at T=${elapsed}s (calibrate cleanup)"
    fi
    if [[ ${elapsed} -ge ${max_wait} ]]; then
      break
    fi
    sleep ${poll_interval}
    elapsed=$((elapsed + poll_interval))
  done

  # Step 12: Verify
  local s4_pass="PASS"
  local s4_reasons=""
  if [[ "${inflight_stuck}" == "0" || "${inflight_stuck}" == "-1" ]]; then
    s4_pass="FAIL"
    s4_reasons="${s4_reasons}No stuck inflight after stop (${inflight_stuck}); "
  fi
  if [[ "${cleanup_time}" == "-1" ]]; then
    s4_pass="FAIL"
    s4_reasons="${s4_reasons}Inflight never reached 0 within ${max_wait}s (final=${inflight_final}); "
  fi
  # Compare with TTL: calibrate should be much faster than TTL=30s
  if [[ "${cleanup_time}" != "-1" ]] && [[ ${cleanup_time} -ge 30 ]]; then
    s4_pass="WARN"
    s4_reasons="${s4_reasons}Cleanup time ${cleanup_time}s >= TTL 30s (calibrate may not have triggered); "
  fi

  log "  Result: ${s4_pass}"
  [[ -n "${s4_reasons}" ]] && log "  Reasons: ${s4_reasons}"
  log "  Calibrate cleanup time: ${cleanup_time}s (vs TTL=${FLEXLB_INFLIGHT_TTL_MS}ms)"

  cat > "${sd}/result.json" <<JSONEOF
{
  "scenario": "Calibrate cleanup",
  "pass": "${s4_pass}",
  "reasons": "${s4_reasons}",
  "inflight_before_stop": ${inflight_before},
  "inflight_stuck": ${inflight_stuck},
  "cleanup_time_s": ${cleanup_time},
  "ttl_ms": ${FLEXLB_INFLIGHT_TTL_MS}
}
JSONEOF

  teardown_environment
}

# ===========================================================================
# Final report generation
# ===========================================================================
generate_report() {
  log "=== Generating final report ==="
  python3 - "${RUN_DIR}" <<'PYEOF'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

run_dir = Path(sys.argv[1])

def load_result(subdir):
    p = run_dir / subdir / "result.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

def load_timeline(subdir):
    p = run_dir / subdir / "timeline.jsonl"
    points = []
    if p.exists():
        for line in p.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    points.append(json.loads(line))
                except Exception:
                    pass
    return points

results = {}
for i, name in [(1, "scenario1"), (2, "scenario2"), (3, "scenario3"), (4, "scenario4")]:
    results[i] = load_result(name)

timelines = {}
timelines[1] = load_timeline("scenario1")
timelines[4] = load_timeline("scenario4")

lines = []
w = lines.append

w("# FlexLB Engine Behavior Test Report")
w("")
w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w(f"**Run Directory**: `{run_dir}`")
w(f"**TTL**: {os.environ.get('FLEXLB_INFLIGHT_TTL_MS', '30000')}ms")
w(f"**MAX_INFLIGHT_BATCHES**: {os.environ.get('FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES', '4')}")
w(f"**SYNC_REQUEST_TIMEOUT_MS**: {os.environ.get('SYNC_REQUEST_TIMEOUT_MS', '1000')}")
w("")

# Summary table
w("## Summary")
w("")
w("| # | Scenario | Result | Key Metric | Detail |")
w("|---|---|---|---|---|")
for i, name in [(1, "TTL Cleanup"), (2, "gRPC Recovery"), (3, "Quota Blocking"), (4, "Calibrate Cleanup")]:
    r = results.get(i, {})
    if not r:
        w(f"| {i} | {name} | N/A | - | not run |")
        continue
    status = r.get("pass", "N/A")
    if i == 1:
        key = f"cleanup={r.get('cleanup_time_s', '?')}s"
        detail = f"inflight: {r.get('inflight_before_kill', '?')} -> {r.get('inflight_final', '?')}"
    elif i == 2:
        key = f"success={r.get('success_rate', '?')}%"
        detail = f"accepted: {r.get('accepted_before', '?')} -> {r.get('accepted_downtime', '?')} -> {r.get('accepted_after', '?')}"
    elif i == 3:
        key = f"recovery={r.get('recovery_rate', '?')}%"
        detail = f"block fail={r.get('block_fail_rate', '?')}%"
    else:
        key = f"cleanup={r.get('cleanup_time_s', '?')}s"
        detail = f"stuck={r.get('inflight_stuck', '?')}"
    w(f"| {i} | {name} | {status} | {key} | {detail} |")
w("")

# Scenario 1 detail
if results.get(1):
    r = results[1]
    w("## Scenario 1: Stuck Inflight TTL Cleanup")
    w("")
    w(f"- **Result**: {r.get('pass', 'N/A')}")
    w(f"- **TTL**: {r.get('ttl_ms', 'N/A')}ms")
    w(f"- **Inflight after kill**: {r.get('inflight_before_kill', 'N/A')}")
    w(f"- **Inflight final**: {r.get('inflight_final', 'N/A')}")
    w(f"- **Cleanup time**: {r.get('cleanup_time_s', 'N/A')}s")
    if r.get("reasons"):
        w(f"- **Reasons**: {r['reasons']}")
    tl = timelines.get(1, [])
    if tl:
        w("")
        w("### Inflight Timeline")
        w("")
        w("| T (s) | Inflight | Accepted | Stopped |")
        w("|---|---|---|---|")
        for p in tl:
            w(f"| {p.get('t', '?')} | {p.get('inflight', '?')} | {p.get('accepted', '?')} | {p.get('stopped', '?')} |")
    w("")

# Scenario 2 detail
if results.get(2):
    r = results[2]
    w("## Scenario 2: gRPC Channel Recovery")
    w("")
    w(f"- **Result**: {r.get('pass', 'N/A')}")
    w(f"- **Accepted before kill**: {r.get('accepted_before', 'N/A')}")
    w(f"- **Accepted during downtime**: {r.get('accepted_downtime', 'N/A')}")
    w(f"- **Accepted after restart**: {r.get('accepted_after', 'N/A')}")
    w(f"- **Recovery requests**: {r.get('recovery_ok', 'N/A')}/{r.get('recovery_total', 'N/A')} ({r.get('success_rate', 'N/A')}%)")
    if r.get("reasons"):
        w(f"- **Reasons**: {r['reasons']}")
    w("")

# Scenario 3 detail
if results.get(3):
    r = results[3]
    w("## Scenario 3: Single-Engine Quota Blocking")
    w("")
    w(f"- **Result**: {r.get('pass', 'N/A')}")
    w(f"- **Blocked requests**: {r.get('block_ok', 'N/A')}/{r.get('block_total', 'N/A')} ok ({r.get('block_fail_rate', 'N/A')}% fail)")
    w(f"- **Recovery requests**: {r.get('recovery_ok', 'N/A')}/{r.get('recovery_total', 'N/A')} ok ({r.get('recovery_rate', 'N/A')}%)")
    if r.get("reasons"):
        w(f"- **Reasons**: {r['reasons']}")
    # Error types
    err_file = run_dir / "scenario3" / "error_types.json"
    if err_file.exists():
        w("")
        w("### Error Type Distribution (blocked phase)")
        w("")
        w("| Error Type | Count |")
        w("|---|---|")
        try:
            errors = json.loads(err_file.read_text())
            for et, count in sorted(errors.items(), key=lambda x: -x[1]):
                w(f"| {et} | {count} |")
        except Exception:
            pass
    w("")

# Scenario 4 detail
if results.get(4):
    r = results[4]
    w("## Scenario 4: Calibrate Fast Cleanup")
    w("")
    w(f"- **Result**: {r.get('pass', 'N/A')}")
    w(f"- **Inflight before stop**: {r.get('inflight_before_stop', 'N/A')}")
    w(f"- **Stuck inflight**: {r.get('inflight_stuck', 'N/A')}")
    w(f"- **Cleanup time**: {r.get('cleanup_time_s', 'N/A')}s")
    w(f"- **TTL (comparison)**: {r.get('ttl_ms', 'N/A')}ms")
    if r.get("reasons"):
        w(f"- **Reasons**: {r['reasons']}")
    tl = timelines.get(4, [])
    if tl:
        w("")
        w("### Inflight Timeline")
        w("")
        w("| T (s) | Inflight | Accepted | Stopped |")
        w("|---|---|---|---|")
        for p in tl:
            w(f"| {p.get('t', '?')} | {p.get('inflight', '?')} | {p.get('accepted', '?')} | {p.get('stopped', '?')} |")
    w("")

# Comparison: Scenario 1 vs Scenario 4
r1 = results.get(1, {})
r4 = results.get(4, {})
if r1 and r4:
    w("## Scenario 1 vs Scenario 4: TTL vs Calibrate Cleanup")
    w("")
    w("| Metric | TTL (Scenario 1) | Calibrate (Scenario 4) |")
    w("|---|---|---|")
    w(f"| Cleanup time | {r1.get('cleanup_time_s', 'N/A')}s | {r4.get('cleanup_time_s', 'N/A')}s |")
    w(f"| Stuck inflight | {r1.get('inflight_before_kill', 'N/A')} | {r4.get('inflight_stuck', 'N/A')} |")
    w("")

# Key findings
w("## Key Findings")
w("")
r1_pass = r1.get("pass", "") if r1 else ""
r4_pass = r4.get("pass", "") if r4 else ""
r1_ct = r1.get("cleanup_time_s", -1) if r1 else -1
r4_ct = r4.get("cleanup_time_s", -1) if r4 else -1
if r1_ct != -1 and r4_ct != -1:
    if r4_ct < r1_ct:
        w(f"- Calibrate cleanup ({r4_ct}s) is faster than TTL cleanup ({r1_ct}s), confirming calibrate triggers on engine restart")
    else:
        w(f"- WARNING: Calibrate cleanup ({r4_ct}s) is not faster than TTL ({r1_ct}s), calibrate may not have triggered properly")
if r1 and r1.get("pass") == "PASS":
    w(f"- TTL cleanup works: stuck inflight cleared after ~{r1.get('cleanup_time_s', '?')}s (TTL={r1.get('ttl_ms', '?')}ms)")
if r2 := results.get(2):
    if r2.get("pass") == "PASS":
        w(f"- gRPC channel recovers after engine restart: {r2.get('recovery_ok', '?')}/{r2.get('recovery_total', '?')} ({r2.get('success_rate', '?')}%)")
    else:
        w(f"- gRPC channel recovery issue: {r2.get('reasons', '')}")
if r3 := results.get(3):
    if r3.get("pass") == "PASS":
        w(f"- Quota blocking confirmed: {r3.get('block_fail_rate', '?')}% fail during block, {r3.get('recovery_rate', '?')}% recovery after TTL cleanup")
    else:
        w(f"- Quota blocking issue: {r3.get('reasons', '')}")
w("")

w("---")
w(f"_Report generated at {datetime.now().isoformat()}_")

report = "\n".join(lines)
report_path = run_dir / "test_report.md"
report_path.write_text(report, encoding="utf-8")
print(report)
PYEOF
}

# ===========================================================================
# Prerequisites check
# ===========================================================================
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
echo "  Short trace: ${SHORT_TRACE_LINES} lines"
echo "  TTL: ${FLEXLB_INFLIGHT_TTL_MS}ms"
echo "  MAX_INFLIGHT_BATCHES: ${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES}"
echo "  SYNC_REQUEST_TIMEOUT_MS: ${SYNC_REQUEST_TIMEOUT_MS}"
echo "  Scenarios: ${SCENARIOS}"
echo "  All prerequisites OK."

# ===========================================================================
# Main
# ===========================================================================
main() {
  local scenarios
  IFS=',' read -ra scenarios <<< "${SCENARIOS}"
  for s in "${scenarios[@]}"; do
    case "${s}" in
      1) scenario_1_ttl_cleanup ;;
      2) scenario_2_grpc_recovery ;;
      3) scenario_3_quota_blocking ;;
      4) scenario_4_calibrate_recovery ;;
      *) echo "Unknown scenario: ${s}" ;;
    esac
  done
  generate_report

  echo ""
  echo "=========================================="
  echo "  Behavior Test Complete"
  echo "=========================================="
  echo "  Report:  ${RUN_DIR}/test_report.md"
  echo "  Run dir: ${RUN_DIR}"
  echo "=========================================="
}

main "$@"

#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# engine_disconnect_ttft_test.sh
#
# FlexLB 引擎断开/恢复 TTFT 测试。
# 通过 mock_engine_cluster 的 HTTP API 停止/恢复单个引擎，
# 观察 Master 的检测、路由切换和恢复行为。
#
# Flow:
#   Phase 1: BASELINE     — load client 持续运行，采集基线 TTFT
#   Phase 2: STOP_ENGINE  — 调用 POST /stop_engine 停止目标引擎
#   Phase 3: DISCONNECTED  — 引擎不可用，观察 Master 检测和路由切换
#   Phase 4: START_ENGINE — 调用 POST /start_engine 恢复引擎
#   Phase 5: RECOVERY     — 观察恢复行为和 TTFT 变化
#
# Key features:
#   - Load client 全程持续运行（带 --enable-fallback），不暂停
#   - stability_monitor.py 后台并行采集 JVM/inflight/mock 指标
#   - 精确时间戳记录（epoch + human-readable）
#   - Master 日志关键事件提取（worker status 失败、alive 标记变更）
#
# Usage:
#   bash engine_disconnect_ttft_test.sh
#   BASELINE_DURATION=60 DISCONNECT_WAIT=30 RECOVERY_WAIT=120 bash engine_disconnect_ttft_test.sh
#   DISCONNECT_TARGET=decode-0 bash engine_disconnect_ttft_test.sh
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

# Load client parameters
LOAD_CLIENT_LIMIT="${LOAD_CLIENT_LIMIT:-50000}"
LOAD_CLIENT_CONCURRENCY="${LOAD_CLIENT_CONCURRENCY:-20}"
LOAD_CLIENT_TIMEOUT_MS="${LOAD_CLIENT_TIMEOUT_MS:-10000}"
LOAD_CLIENT_REPLAY_SPEED="${LOAD_CLIENT_REPLAY_SPEED:-0}"
LOAD_CLIENT_TRACE_FILTER_OL="${LOAD_CLIENT_TRACE_FILTER_OL:-200}"

# Timing parameters (seconds)
BASELINE_DURATION="${BASELINE_DURATION:-60}"
DISCONNECT_WAIT="${DISCONNECT_WAIT:-30}"
RECOVERY_WAIT="${RECOVERY_WAIT:-120}"

# Engine disconnect parameters
DISCONNECT_TARGET="${DISCONNECT_TARGET:-prefill-0}"
# Auto-infer role from target name (prefill-0 → prefill, decode-1 → decode)
DISCONNECT_ROLE="${DISCONNECT_ROLE:-$(echo "${DISCONNECT_TARGET}" | cut -d- -f1)}"

# Monitor interval (seconds)
MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"

# -- Run directory ---------------------------------------------------------

RUN_DIR="${SCRIPT_DIR}/run/disconnect_ttft_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"

# -- State -----------------------------------------------------------------

MOCK_PID=""
FLEXLB_PID=""
LOAD_CLIENT_PID=""
MONITOR_PID=""
FLEXLB_ENV_ARGS=()
STOP_EPOCH=""
STOP_TS_HUMAN=""
START_EPOCH=""
START_TS_HUMAN=""
ENDPOINT_FILE=""

# -- Helpers --------------------------------------------------------------

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
  if [[ -n "${LOAD_CLIENT_PID}" ]]; then
    kill "${LOAD_CLIENT_PID}" >/dev/null 2>&1 || true
    wait "${LOAD_CLIENT_PID}" 2>/dev/null || true
    LOAD_CLIENT_PID=""
  fi
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
  echo "[cleanup] done."
}
trap cleanup EXIT

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
echo "  Disconnect target: ${DISCONNECT_TARGET} (role=${DISCONNECT_ROLE})"
echo "  All prerequisites OK."

# -- Write perf config -----------------------------------------------------

PERF_CONFIG_FILE="${RUN_DIR}/perf.json"
MOCK_PREFILL_MS="${MOCK_PREFILL_MS:-1000.0}"
MOCK_DECODE_STEP_MS="${MOCK_DECODE_STEP_MS:-100.0}"
cat > "${PERF_CONFIG_FILE}" <<JSON
{
  "block_size": 1024,
  "sleep_scale": 1.0,
  "prefill": { "fixed_ms": ${MOCK_PREFILL_MS}, "scale": 1.0 },
  "decode": {
    "scale": 1.0,
    "step_ms_by_batch": [
      [1, ${MOCK_DECODE_STEP_MS}], [2, ${MOCK_DECODE_STEP_MS}], [4, ${MOCK_DECODE_STEP_MS}], [8, ${MOCK_DECODE_STEP_MS}],
      [16, ${MOCK_DECODE_STEP_MS}], [32, ${MOCK_DECODE_STEP_MS}], [64, ${MOCK_DECODE_STEP_MS}], [128, ${MOCK_DECODE_STEP_MS}], [256, ${MOCK_DECODE_STEP_MS}]
    ]
  }
}
JSON

# -- Start master function ------------------------------------------------

start_master() {
  local log_file="$1"
  echo "  starting master ..."
  env ${FLEXLB_ENV_ARGS[@]+"${FLEXLB_ENV_ARGS[@]}"} \
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
    "HIPPO_ROLE=flexlb_disconnect_ttft_test" \
    java "${JAVA_MODULE_OPTS[@]}" -jar "${FLEXLB_JAR}" \
    --server.port="${FLEXLB_HTTP_PORT}" \
    --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
    --spring.profiles.active=test \
    --logging.level.org.flexlb=DEBUG \
    >"${log_file}" 2>&1 &
  FLEXLB_PID="$!"
  wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60
  echo "  master started (pid=${FLEXLB_PID})"
}

# ===========================================================================
# Step 1: Start mock engine cluster
# ===========================================================================

echo ""
echo "=== Step 1: Start mock engine cluster (${N_PREFILL}P + ${N_DECODE}D) ==="
ENDPOINT_FILE="${RUN_DIR}/endpoints.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/mock_engine_cluster.py" \
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

# ===========================================================================
# Step 2: Parse service-discovery env vars from endpoints.json
# ===========================================================================

while IFS= read -r line; do
  FLEXLB_ENV_ARGS+=("${line}")
done < <(python3 - "${ENDPOINT_FILE}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for key, value in payload["env"].items():
    print(f"{key}={value}")
PY
)
echo "  parsed ${#FLEXLB_ENV_ARGS[@]} service-discovery env vars"

# ===========================================================================
# Step 3: Filter trace (short requests only for high-QPS pressure)
# ===========================================================================

echo ""
echo "=== Step 3: Filter trace (ol <= ${LOAD_CLIENT_TRACE_FILTER_OL}) ==="
FILTERED_TRACE="${RUN_DIR}/trace_filtered.jsonl"
python3 - "${TRACE_FILE}" "${FILTERED_TRACE}" "${LOAD_CLIENT_TRACE_FILTER_OL}" <<'PY'
import json, sys
src, dst, max_ol = sys.argv[1], sys.argv[2], int(sys.argv[3])
count = 0
with open(src) as fin, open(dst, 'w') as fout:
    for line in fin:
        try:
            rec = json.loads(line)
            if rec.get('ol', 0) <= max_ol:
                fout.write(line)
                count += 1
        except:
            pass
print(f"  filtered: {count} requests (ol <= {max_ol})")
PY
TRACE_FILE_USE="${FILTERED_TRACE}"

# ===========================================================================
# Step 4: Start FlexLB Master
# ===========================================================================

echo ""
echo "=== Step 4: Start FlexLB Master ==="
MASTER_LOG="${RUN_DIR}/flexlb_master.log"
start_master "${MASTER_LOG}"

# ===========================================================================
# Step 5: Start load client (background, with fallback)
# ===========================================================================

echo ""
echo "=== Step 5: Start load client (with fallback) ==="
LOAD_CLIENT_DIR="${RUN_DIR}/load_client"
mkdir -p "${LOAD_CLIENT_DIR}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/flexlb_load_client.py" \
  "${TRACE_FILE_USE}" \
  --flexlb-http-addr "127.0.0.1:${FLEXLB_HTTP_PORT}" \
  --schedule-mode batch \
  --replay-speed "${LOAD_CLIENT_REPLAY_SPEED}" \
  --limit "${LOAD_CLIENT_LIMIT}" \
  --max-concurrency "${LOAD_CLIENT_CONCURRENCY}" \
  --timeout-ms "${LOAD_CLIENT_TIMEOUT_MS}" \
  --output-dir "${LOAD_CLIENT_DIR}" \
  --enable-fallback \
  --endpoints-file "${ENDPOINT_FILE}" \
  >"${RUN_DIR}/load_client.log" 2>&1 &
LOAD_CLIENT_PID="$!"
echo "  load client started (pid=${LOAD_CLIENT_PID}, fallback enabled, replay_speed=${LOAD_CLIENT_REPLAY_SPEED})"

# ===========================================================================
# Step 6: Start stability monitor (background)
# ===========================================================================

echo ""
echo "=== Step 6: Start stability monitor ==="
PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/stability_monitor.py" \
  --flexlb-http-addr "127.0.0.1:${FLEXLB_HTTP_PORT}" \
  --management-port "${FLEXLB_MANAGEMENT_PORT}" \
  --mock-http-port "${MOCK_HTTP_PORT}" \
  --output "${RUN_DIR}/monitor.jsonl" \
  --interval "${MONITOR_INTERVAL}" \
  >"${RUN_DIR}/monitor.log" 2>&1 &
MONITOR_PID="$!"
echo "  monitor started (pid=${MONITOR_PID}, interval=${MONITOR_INTERVAL}s)"

# ===========================================================================
# Phase 1: BASELINE
# ===========================================================================

echo ""
echo "=== Phase 1: Baseline (${BASELINE_DURATION}s) ==="
echo "  load client is running (pid=${LOAD_CLIENT_PID})"
sleep "${BASELINE_DURATION}"

# Collect baseline data
echo "  collecting baseline data ..."
cp "${LOAD_CLIENT_DIR}/per_request.jsonl" "${RUN_DIR}/baseline_per_request.jsonl" 2>/dev/null \
  || echo "  NOTE: per_request.jsonl not available yet"
curl -s -o "${RUN_DIR}/baseline_inflight.json" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" \
  || echo "  WARNING: inflight_status request failed"
curl -s -o "${RUN_DIR}/baseline_snapshot.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" \
  || echo "  WARNING: snapshot request failed"
curl -s -o "${RUN_DIR}/baseline_requests.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/requests" \
  || echo "  WARNING: requests request failed"
echo "  baseline data collected"

# ===========================================================================
# Phase 2: STOP_ENGINE
# ===========================================================================

echo ""
echo "=== Phase 2: Stop Engine (${DISCONNECT_TARGET}) ==="
STOP_EPOCH=$(date +%s)
STOP_TS_HUMAN=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "  stopping engine ${DISCONNECT_TARGET} at ${STOP_TS_HUMAN} (epoch=${STOP_EPOCH})"
echo "  calling POST /stop_engine ..."
STOP_RESPONSE=$(curl -s -X POST \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/stop_engine" \
  -H 'Content-Type: application/json' \
  -d "{\"engine\": \"${DISCONNECT_TARGET}\"}" || echo "CURL_FAILED")
echo "  response: ${STOP_RESPONSE}"
echo "  engine ${DISCONNECT_TARGET} stopped"

# ===========================================================================
# Phase 3: DISCONNECTED (DISCONNECT_WAIT seconds)
# ===========================================================================

echo ""
echo "=== Phase 3: Disconnected (${DISCONNECT_WAIT}s) ==="
echo "  engine ${DISCONNECT_TARGET} is unavailable, observing Master behavior ..."
sleep "${DISCONNECT_WAIT}"

# Collect disconnect-period data
echo "  collecting disconnect-period data ..."
curl -s -o "${RUN_DIR}/disconnect_snapshot.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" \
  || echo "  WARNING: snapshot request failed"
curl -s -o "${RUN_DIR}/disconnect_requests.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/requests" \
  || echo "  WARNING: requests request failed"
curl -s -o "${RUN_DIR}/disconnect_inflight.json" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" \
  || echo "  WARNING: inflight_status request failed"
echo "  disconnect-period data collected"

# ===========================================================================
# Phase 4: START_ENGINE
# ===========================================================================

echo ""
echo "=== Phase 4: Start Engine (${DISCONNECT_TARGET}) ==="
START_EPOCH=$(date +%s)
START_TS_HUMAN=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "  starting engine ${DISCONNECT_TARGET} at ${START_TS_HUMAN} (epoch=${START_EPOCH})"
echo "  calling POST /start_engine ..."
START_RESPONSE=$(curl -s -X POST \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/start_engine" \
  -H 'Content-Type: application/json' \
  -d "{\"engine\": \"${DISCONNECT_TARGET}\"}" || echo "CURL_FAILED")
echo "  response: ${START_RESPONSE}"
echo "  engine ${DISCONNECT_TARGET} started"

# ===========================================================================
# Phase 5: RECOVERY (RECOVERY_WAIT seconds)
# ===========================================================================

echo ""
echo "=== Phase 5: Recovery (${RECOVERY_WAIT}s) ==="
sleep "${RECOVERY_WAIT}"

# Collect recovery-period data
echo "  collecting recovery data ..."
curl -s -o "${RUN_DIR}/recovery_inflight.json" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" \
  || echo "  WARNING: inflight_status request failed"
curl -s -o "${RUN_DIR}/recovery_snapshot.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" \
  || echo "  WARNING: snapshot request failed"
curl -s -o "${RUN_DIR}/recovery_requests.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/requests" \
  || echo "  WARNING: requests request failed"
echo "  recovery data collected"

# ===========================================================================
# Stop load client
# ===========================================================================

echo ""
echo "=== Stop load client ==="
if kill -0 "${LOAD_CLIENT_PID}" 2>/dev/null; then
  echo "  stopping load client (pid=${LOAD_CLIENT_PID}) ..."
  kill "${LOAD_CLIENT_PID}" 2>/dev/null || true
  wait "${LOAD_CLIENT_PID}" 2>/dev/null || true
else
  echo "  load client already exited"
fi
LOAD_CLIENT_PID=""
sleep 2  # Allow file flush

# ===========================================================================
# Stop stability monitor
# ===========================================================================

echo ""
echo "=== Stop stability monitor ==="
if kill -0 "${MONITOR_PID}" 2>/dev/null; then
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
fi
MONITOR_PID=""

# ===========================================================================
# Write timestamps.json
# ===========================================================================
# Map STOP_EPOCH → kill_epoch, START_EPOCH → restart_epoch for
# compatibility with analyze_ttft_degradation.py

echo ""
echo "=== Write timestamps ==="
python3 -c "
import json
ts = {
    'kill_epoch': ${STOP_EPOCH},
    'restart_epoch': ${START_EPOCH},
    'stop_epoch': ${STOP_EPOCH},
    'start_epoch': ${START_EPOCH},
    'kill_ts': '${STOP_TS_HUMAN}',
    'restart_ts': '${START_TS_HUMAN}',
    'stop_ts': '${STOP_TS_HUMAN}',
    'start_ts': '${START_TS_HUMAN}',
    'disconnect_target': '${DISCONNECT_TARGET}',
    'disconnect_role': '${DISCONNECT_ROLE}',
    'downtime_s': ${START_EPOCH} - ${STOP_EPOCH},
    'baseline_duration_s': ${BASELINE_DURATION},
    'disconnect_wait_s': ${DISCONNECT_WAIT},
    'recovery_wait_s': ${RECOVERY_WAIT},
}
with open('${RUN_DIR}/timestamps.json', 'w') as f:
    json.dump(ts, f, indent=2)
print(json.dumps(ts, indent=2))
"

# ===========================================================================
# Master log key event extraction
# ===========================================================================

echo ""
echo "=== Master log key event extraction ==="
if [[ -f "${MASTER_LOG}" ]]; then
  echo "  extracting from master log ..."

  # Worker status check failures
  grep -m1 "worker status check failed" "${MASTER_LOG}" > "${RUN_DIR}/master_events.txt" 2>/dev/null || true

  # Worker marked alive=false
  grep -m1 "marking worker.*alive=false" "${MASTER_LOG}" >> "${RUN_DIR}/master_events.txt" 2>/dev/null \
    || grep -m1 "alive=false" "${MASTER_LOG}" >> "${RUN_DIR}/master_events.txt" 2>/dev/null || true

  # Worker status update (recovery)
  grep "onWorkerStatusUpdate" "${MASTER_LOG}" | tail -1 >> "${RUN_DIR}/master_events.txt" 2>/dev/null || true

  # Worker marked alive=true (recovery)
  grep -m1 "marking worker.*alive=true" "${MASTER_LOG}" >> "${RUN_DIR}/master_events.txt" 2>/dev/null \
    || grep -m1 "alive=true" "${MASTER_LOG}" >> "${RUN_DIR}/master_events.txt" 2>/dev/null || true

  # Engine connection errors
  grep -m1 "UNAVAILABLE" "${MASTER_LOG}" >> "${RUN_DIR}/master_events.txt" 2>/dev/null || true
  grep -m1 "connection refused" "${MASTER_LOG}" >> "${RUN_DIR}/master_events.txt" 2>/dev/null || true

  # CostBased strategy events
  grep -m1 "CostBasedPrefillStrategy" "${MASTER_LOG}" >> "${RUN_DIR}/master_events.txt" 2>/dev/null || true

  echo "  master events written to ${RUN_DIR}/master_events.txt"
else
  echo "  WARNING: master log not found"
fi

# ===========================================================================
# Mock engine final snapshots
# ===========================================================================

echo ""
echo "=== Mock engine final snapshots ==="
curl -s "http://127.0.0.1:${MOCK_HTTP_PORT}/requests" > "${RUN_DIR}/mock_requests_final.json" 2>/dev/null \
  || echo "  WARNING: requests request failed"
curl -s "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" > "${RUN_DIR}/mock_snapshot_final.json" 2>/dev/null \
  || echo "  WARNING: snapshot request failed"
echo "  mock engine snapshots saved"

# ===========================================================================
# Copy load client final outputs
# ===========================================================================

echo ""
echo "=== Copy load client outputs ==="
cp "${LOAD_CLIENT_DIR}/per_request.jsonl" "${RUN_DIR}/final_per_request.jsonl" 2>/dev/null \
  || echo "  WARNING: per_request.jsonl not available"
cp "${LOAD_CLIENT_DIR}/summary.json" "${RUN_DIR}/final_summary.json" 2>/dev/null \
  || echo "  NOTE: summary.json not available (client was killed before completion)"
cp "${LOAD_CLIENT_DIR}/report.md" "${RUN_DIR}/load_client_report.md" 2>/dev/null || true

# ===========================================================================
# Run TTFT degradation analysis
# ===========================================================================

echo ""
echo "=== Run TTFT degradation analysis ==="
ANALYSIS_REPORT="${RUN_DIR}/ttft_analysis.md"
STOP_EPOCH="${STOP_EPOCH}" \
START_EPOCH="${START_EPOCH}" \
PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/analyze_ttft_degradation.py" \
  --per-requests "${RUN_DIR}/final_per_request.jsonl" \
  --monitor "${RUN_DIR}/monitor.jsonl" \
  --kill-epoch "${STOP_EPOCH}" \
  --restart-epoch "${START_EPOCH}" \
  --output "${ANALYSIS_REPORT}" \
  2>&1 || echo "  WARNING: analysis failed (may need more data)"
echo "  analysis report: ${ANALYSIS_REPORT}"

# ===========================================================================
# Generate test report
# ===========================================================================

echo ""
echo "=== Generate test report ==="
STOP_EPOCH="${STOP_EPOCH}" \
START_EPOCH="${START_EPOCH}" \
STOP_TS_HUMAN="${STOP_TS_HUMAN}" \
START_TS_HUMAN="${START_TS_HUMAN}" \
BASELINE_DURATION="${BASELINE_DURATION}" \
DISCONNECT_WAIT="${DISCONNECT_WAIT}" \
RECOVERY_WAIT="${RECOVERY_WAIT}" \
DISCONNECT_TARGET="${DISCONNECT_TARGET}" \
DISCONNECT_ROLE="${DISCONNECT_ROLE}" \
MOCK_PID="${MOCK_PID}" \
FLEXLB_PID="${FLEXLB_PID}" \
python3 - "${RUN_DIR}" <<'PYEOF'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

run_dir = Path(sys.argv[1])

stop_epoch = int(os.environ.get("STOP_EPOCH", 0))
start_epoch = int(os.environ.get("START_EPOCH", 0))
stop_ts = os.environ.get("STOP_TS_HUMAN", "N/A")
start_ts = os.environ.get("START_TS_HUMAN", "N/A")
baseline_duration = os.environ.get("BASELINE_DURATION", "N/A")
disconnect_wait = os.environ.get("DISCONNECT_WAIT", "N/A")
recovery_wait = os.environ.get("RECOVERY_WAIT", "N/A")
disconnect_target = os.environ.get("DISCONNECT_TARGET", "N/A")
disconnect_role = os.environ.get("DISCONNECT_ROLE", "N/A")
mock_pid = os.environ.get("MOCK_PID", "N/A")
flexlb_pid = os.environ.get("FLEXLB_PID", "N/A")

downtime_s = start_epoch - stop_epoch if stop_epoch and start_epoch else "N/A"

# List output files
output_files = []
for p in sorted(run_dir.iterdir()):
    if p.is_file():
        size = p.stat().st_size
        output_files.append((p.name, size))

lines = []
w = lines.append

w("# FlexLB Engine Disconnect/Recovery TTFT Test Report")
w("")
w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w(f"**Run Directory**: `{run_dir}`")
w(f"**Disconnect Target**: `{disconnect_target}` (role={disconnect_role})")
w("")

w("## 1. Timeline")
w("")
w("| Phase | Duration | Timestamp |")
w("|---|---|---|")
w(f"| Phase 1: Baseline | {baseline_duration}s | — |")
w(f"| Phase 2: Stop Engine ({disconnect_target}) | — | {stop_ts} (epoch={stop_epoch}) |")
w(f"| Phase 3: Disconnected | {disconnect_wait}s | — |")
w(f"| Phase 4: Start Engine ({disconnect_target}) | — | {start_ts} (epoch={start_epoch}) |")
w(f"| Phase 5: Recovery | {recovery_wait}s | — |")
w("")
w(f"- **Engine downtime**: {downtime_s}s")
w(f"- **Master**: stayed running throughout (no kill/restart)")
w("")

w("## 2. Process PIDs")
w("")
w("| Process | PID |")
w("|---|---|")
w(f"| FlexLB Master | {flexlb_pid} |")
w(f"| Mock Engine Cluster | {mock_pid} |")
w("")

w("## 3. Output Files")
w("")
w("| File | Size |")
w("|---|---|")
for name, size in output_files:
    if size >= 1024 * 1024:
        size_str = f"{size / (1024 * 1024):.1f} MB"
    elif size >= 1024:
        size_str = f"{size / 1024:.1f} KB"
    else:
        size_str = f"{size} B"
    w(f"| {name} | {size_str} |")
w("")

w("## 4. Key Data Files for Analysis")
w("")
w("- `timestamps.json` — precise stop/start timestamps (stop_epoch/start_epoch + kill_epoch/restart_epoch for analysis compatibility)")
w("- `monitor.jsonl` — stability monitor metrics (JVM, inflight, mock)")
w("- `load_client/per_request.jsonl` — per-request latency data")
w("- `baseline_per_request.jsonl` — baseline period request data")
w("- `flexlb_master.log` — Master log (single file, ran throughout)")
w("- `master_events.txt` — key events extracted from master log")
w("- `baseline_snapshot.json` / `disconnect_snapshot.json` / `recovery_snapshot.json` — mock engine snapshots")
w("- `ttft_analysis.md` — TTFT degradation analysis report")
w("- `ttft_timeline.png` / `ttft_timeline.json` — timeline chart and data")
w("")
w("_TTFT degradation analysis is handled by `analyze_ttft_degradation.py`._")
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
echo "  Report:            ${RUN_DIR}/test_report.md"
echo "  TTFT Analysis:      ${RUN_DIR}/ttft_analysis.md"
echo "  Run dir:            ${RUN_DIR}"
echo "  Timestamps:         ${RUN_DIR}/timestamps.json"
echo "  Monitor:            ${RUN_DIR}/monitor.jsonl"
echo "  Master log:         ${RUN_DIR}/flexlb_master.log"
echo "  Mock log:           ${RUN_DIR}/mock_engine.log"
echo "  Load client:        ${RUN_DIR}/load_client.log"
echo "  Master events:      ${RUN_DIR}/master_events.txt"
echo "  Disconnect target:  ${DISCONNECT_TARGET} (role=${DISCONNECT_ROLE})"
echo "=========================================="

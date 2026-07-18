#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# master_kill_restart_test.sh
#
# FlexLB Master kill-restart destructive test.
#
# Flow:
#   1.  Start mock engine cluster (2 prefill + 2 decode)
#   2.  Start FlexLB Master (batch path)
#   3.  Start load client (background, replaying trace)
#   4.  Wait for steady state
#   5.  Collect baseline data (inflight, snapshot, load client progress)
#   6.  Kill Master (kill -9)
#   7.  Wait (observe failures during downtime)
#   8.  Collect kill-period data (mock engine snapshot/requests)
#   9.  Restart Master
#   10. Wait (observe recovery)
#   11. Stop load client
#   12. Collect post-restart data
#   13. Generate test report
#   14. Cleanup (trap EXIT)
#
# Usage:
#   bash master_kill_restart_test.sh
#   N_PREFILL=2 N_DECODE=2 bash master_kill_restart_test.sh
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
LOAD_CLIENT_LIMIT="${LOAD_CLIENT_LIMIT:-0}"
LOAD_CLIENT_CONCURRENCY="${LOAD_CLIENT_CONCURRENCY:-20}"
LOAD_CLIENT_TIMEOUT_MS="${LOAD_CLIENT_TIMEOUT_MS:-10000}"
LOAD_CLIENT_REPLAY_SPEED="${LOAD_CLIENT_REPLAY_SPEED:-20}"

# Timing parameters (seconds)
STEADY_STATE_WAIT="${STEADY_STATE_WAIT:-8}"
KILL_WAIT="${KILL_WAIT:-8}"
RECOVERY_WAIT="${RECOVERY_WAIT:-15}"

# Kill timing mode: "steady" (sleep STEADY_STATE_WAIT) or "decode" (poll until decode engines have running > 0)
KILL_TIMING="${KILL_TIMING:-steady}"

# -- Run directory ---------------------------------------------------------

RUN_DIR="${SCRIPT_DIR}/run/kill_restart_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"

# -- State -----------------------------------------------------------------

MOCK_PID=""
FLEXLB_PID=""
LOAD_CLIENT_PID=""
FLEXLB_ENV_ARGS=()
MASTER_PID_INITIAL=""
MASTER_PID_RESTART=""
KILL_TS=""
RESTART_TS=""

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
echo "  All prerequisites OK."

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
    "HIPPO_ROLE=flexlb_kill_restart_test" \
    java "${JAVA_MODULE_OPTS[@]}" -jar "${FLEXLB_JAR}" \
    --server.port="${FLEXLB_HTTP_PORT}" \
    --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
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
# Step 3: Start FlexLB Master
# ===========================================================================

echo ""
echo "=== Step 3: Start FlexLB Master (batch path) ==="
start_master "${RUN_DIR}/flexlb_master_initial.log"
MASTER_PID_INITIAL="${FLEXLB_PID}"

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
if [[ "${KILL_TIMING}" == "decode" ]]; then
  echo "=== Step 5: Waiting for requests to reach decode phase ==="
  DECODE_WAIT_TIMEOUT=30
  DECODE_WAIT_ELAPSED=0
  while [[ ${DECODE_WAIT_ELAPSED} -lt ${DECODE_WAIT_TIMEOUT} ]]; do
    DECODE_RUNNING=$(curl -s "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" 2>/dev/null | \
      python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    total = sum(e.get('running', 0) for e in data.get('engines', []) if e.get('role', '') == 'decode')
    print(total)
except:
    print(0)
" 2>/dev/null || echo 0)
    if [[ "${DECODE_RUNNING}" -gt 0 ]]; then
      echo "  decode running: ${DECODE_RUNNING}, proceeding to kill"
      break
    fi
    sleep 1
    DECODE_WAIT_ELAPSED=$((DECODE_WAIT_ELAPSED + 1))
  done
  if [[ ${DECODE_WAIT_ELAPSED} -ge ${DECODE_WAIT_TIMEOUT} ]]; then
    echo "  WARNING: timeout waiting for decode phase, proceeding anyway"
  fi
else
  echo "=== Step 5: Steady state wait (${STEADY_STATE_WAIT}s) ==="
  sleep "${STEADY_STATE_WAIT}"
fi
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
echo "  - inflight_status ..."
curl -s -o "${RUN_DIR}/baseline_inflight.json" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" \
  || echo "  WARNING: inflight_status request failed"
echo "  - mock snapshot ..."
curl -s -o "${RUN_DIR}/baseline_snapshot.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" \
  || echo "  WARNING: snapshot request failed"
echo "  - mock requests ..."
curl -s -o "${RUN_DIR}/baseline_requests.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/requests" \
  || echo "  WARNING: requests request failed"
cp "${RUN_DIR}/baseline_requests.json" "${RUN_DIR}/pre_kill_requests.json" 2>/dev/null \
  || echo "  NOTE: baseline_requests.json not available for pre_kill copy"
echo "  - load client per_request.jsonl ..."
cp "${LOAD_CLIENT_DIR}/per_request.jsonl" "${RUN_DIR}/pre_kill_per_request.jsonl" 2>/dev/null \
  || echo "  NOTE: per_request.jsonl not available yet"
cp "${LOAD_CLIENT_DIR}/summary.json" "${RUN_DIR}/pre_kill_summary.json" 2>/dev/null \
  || echo "  NOTE: summary.json not available (client still running)"

# ===========================================================================
# Step 7: KILL Master (kill -9)
# ===========================================================================

echo ""
echo "=== Step 7: KILL Master (kill -9) ==="
KILL_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "  kill timestamp: ${KILL_TS}"
echo "  killing master (pid=${FLEXLB_PID}) ..."
kill -9 "${FLEXLB_PID}" || true
wait "${FLEXLB_PID}" 2>/dev/null || true
FLEXLB_PID=""

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
echo "  - mock snapshot ..."
curl -s -o "${RUN_DIR}/kill_snapshot.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" \
  || echo "  WARNING: snapshot request failed"
echo "  - mock requests ..."
curl -s -o "${RUN_DIR}/kill_requests.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/requests" \
  || echo "  WARNING: requests request failed"
echo "  - master inflight_status (expected to fail) ..."
curl -s -o "${RUN_DIR}/kill_inflight_attempt.json" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" 2>/dev/null \
  || echo "  master HTTP down (expected)"

# ===========================================================================
# Step 10: Restart Master
# ===========================================================================

echo ""
echo "=== Step 10: Restart Master ==="
RESTART_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "  restart timestamp: ${RESTART_TS}"
start_master "${RUN_DIR}/flexlb_master_restart.log"
MASTER_PID_RESTART="${FLEXLB_PID}"

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
# Step 12.5: Recovery Verification
# ===========================================================================

echo ""
echo "=== Step 12.5: Recovery Verification ==="
# Create recovery trace with only short-output requests (ol <= 200) to avoid
# timeouts from high output_len requests (20000 tokens at 20ms/step = 400s)
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
  >"${RUN_DIR}/recovery_verify.log" 2>&1
echo "  recovery verification completed"
cat "${RECOVERY_VERIFY_DIR}/summary.json" 2>/dev/null || echo "  NOTE: recovery summary not available"
sleep 5  # drain wait for recovery verification in-flight to settle

# ===========================================================================
# Step 13: Collect post-restart data
# ===========================================================================

echo ""
echo "=== Step 13: Collect post-restart data ==="
echo "  - inflight_status ..."
curl -s -o "${RUN_DIR}/post_restart_inflight.json" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" \
  || echo "  WARNING: inflight_status request failed"
echo "  - mock snapshot ..."
curl -s -o "${RUN_DIR}/post_restart_snapshot.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/snapshot" \
  || echo "  WARNING: snapshot request failed"
echo "  - mock requests ..."
curl -s -o "${RUN_DIR}/post_restart_requests.json" \
  "http://127.0.0.1:${MOCK_HTTP_PORT}/requests" \
  || echo "  WARNING: requests request failed"
echo "  - load client outputs ..."
cp "${LOAD_CLIENT_DIR}/summary.json" "${RUN_DIR}/final_summary.json" 2>/dev/null \
  || echo "  NOTE: summary.json not available (client was killed before completion)"
cp "${LOAD_CLIENT_DIR}/per_request.jsonl" "${RUN_DIR}/final_per_request.jsonl" 2>/dev/null \
  || echo "  WARNING: per_request.jsonl not available"
cp "${LOAD_CLIENT_DIR}/report.md" "${RUN_DIR}/load_client_report.md" 2>/dev/null || true

# ===========================================================================
# Step 14: Generate test report
# ===========================================================================

echo ""
echo "=== Step 14: Generate test report ==="
KILL_TS="${KILL_TS}" \
RESTART_TS="${RESTART_TS}" \
MASTER_PID_INITIAL="${MASTER_PID_INITIAL}" \
MASTER_PID_RESTART="${MASTER_PID_RESTART}" \
MOCK_PID="${MOCK_PID}" \
LOAD_CLIENT_PID="${LOAD_CLIENT_PID:-exited}" \
KILL_TIMING="${KILL_TIMING}" \
python3 - "${RUN_DIR}" <<'PYEOF'
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

run_dir = Path(sys.argv[1])

# -- Metadata from environment --
kill_ts = os.environ.get("KILL_TS", "N/A")
restart_ts = os.environ.get("RESTART_TS", "N/A")
master_pid_initial = os.environ.get("MASTER_PID_INITIAL", "N/A")
master_pid_restart = os.environ.get("MASTER_PID_RESTART", "N/A")
mock_pid = os.environ.get("MOCK_PID", "N/A")
load_client_pid = os.environ.get("LOAD_CLIENT_PID", "N/A")
kill_timing = os.environ.get("KILL_TIMING", "steady")

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

# -- Load data --
results = load_jsonl("final_per_request.jsonl")
baseline_inflight = load_json("baseline_inflight.json")
post_restart_inflight = load_json("post_restart_inflight.json")
baseline_snapshot = load_json("baseline_snapshot.json")
kill_snapshot = load_json("kill_snapshot.json")
post_restart_snapshot = load_json("post_restart_snapshot.json")
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

# Use summary.json if available for authoritative numbers
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
    elif "channel shutdown" in el or "channelclosederror" in el or "channel has been shut down" in el:
        error_types["channel_shutdown"] += 1
    elif "deadline" in el or "deadlineexceeded" in el:
        error_types["deadline_exceeded"] += 1
    elif "unavailable" in el:
        error_types["grpc_unavailable"] += 1
    elif "timeout" in el or "timeouterror" in el:
        error_types["timeout"] += 1
    elif "cancelled" in el or "canceled" in el:
        error_types["cancelled"] += 1
    elif "internal" in el and "rpc" in el:
        error_types["grpc_internal"] += 1
    elif "broken pipe" in el:
        error_types["broken_pipe"] += 1
    elif "eof" in el:
        error_types["eof"] += 1
    else:
        short = err[:100].replace("\n", " ")
        error_types[f"other: {short}"] += 1

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
post_restart_in = parse_inflight(post_restart_inflight)

# -- Mock engine residual check --
def check_residual(snapshot):
    if not snapshot:
        return None
    engines = snapshot.get("engines", [])
    total_running = sum(e.get("running", 0) for e in engines)
    return {
        "total_running": total_running,
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

baseline_res = check_residual(baseline_snapshot)
kill_res = check_residual(kill_snapshot)
post_restart_res = check_residual(post_restart_snapshot)

# -- Recovery verification data --
recovery_summary = load_json("recovery_verify/summary.json")
recovery_total = recovery_summary.get("total_requests", 0) if recovery_summary else 0
recovery_ok = recovery_summary.get("completed", 0) if recovery_summary else 0
recovery_success_rate = (recovery_ok / recovery_total * 100) if recovery_total > 0 else 0
recovery_ttft_p50 = "N/A"
if recovery_summary:
    recovery_ttft_p50 = recovery_summary.get("latency", {}).get("ttft_ms", {}).get("p50", "N/A")

# Pre-kill baseline TTFT (from final_summary.json, or fallback to pre_kill_per_request.jsonl)
def _percentile(data, p):
    if not data:
        return "N/A"
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    if idx >= len(data_sorted):
        idx = len(data_sorted) - 1
    return data_sorted[idx]

baseline_ttft_p50 = "N/A"
if final_summary:
    baseline_ttft_p50 = final_summary.get("latency", {}).get("ttft_ms", {}).get("p50", "N/A")
else:
    # Fallback: compute baseline TTFT from pre_kill_per_request.jsonl
    pre_kill_path = run_dir / "pre_kill_per_request.jsonl"
    if pre_kill_path.exists():
        pre_kill_ttfts = []
        with open(pre_kill_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    if req.get("status") == "ok" and req.get("ttft_ms") is not None:
                        pre_kill_ttfts.append(req["ttft_ms"])
                except Exception:
                    pass
        p50 = _percentile(pre_kill_ttfts, 50)
        if p50 != "N/A":
            baseline_ttft_p50 = p50
            print(f"  [report] baseline TTFT p50 computed from pre_kill_per_request.jsonl: {p50}ms ({len(pre_kill_ttfts)} ok requests)")

# TTFT degradation ratio
ttft_degradation = "N/A"
if isinstance(recovery_ttft_p50, (int, float)) and isinstance(baseline_ttft_p50, (int, float)) and baseline_ttft_p50 > 0:
    ttft_degradation = ((recovery_ttft_p50 - baseline_ttft_p50) / baseline_ttft_p50) * 100

# -- Batch ID conflict check --
batch_id_conflict = False
batch_id_conflict_details = ""
pre_kill_requests = load_json("pre_kill_requests.json")
post_restart_requests = load_json("post_restart_requests.json")
if pre_kill_requests and post_restart_requests:
    try:
        pre_kill_ids = set()
        for engine_name, engine_data in pre_kill_requests.items():
            if isinstance(engine_data, dict):
                for req_id, req_info in engine_data.items():
                    if isinstance(req_info, dict) and "batch_id" in req_info:
                        pre_kill_ids.add(req_info["batch_id"])

        post_restart_ids = set()
        for engine_name, engine_data in post_restart_requests.items():
            if isinstance(engine_data, dict):
                for req_id, req_info in engine_data.items():
                    if isinstance(req_info, dict) and "batch_id" in req_info:
                        post_restart_ids.add(req_info["batch_id"])

        new_ids = post_restart_ids - pre_kill_ids
        overlap_ids = post_restart_ids & pre_kill_ids
        batch_id_conflict_details = (
            f"pre-kill IDs: {len(pre_kill_ids)}, "
            f"post-restart IDs: {len(post_restart_ids)}, "
            f"new IDs: {len(new_ids)}, overlap: {len(overlap_ids)}"
        )
    except Exception as e:
        batch_id_conflict_details = f"Failed to parse requests data: {e}"
else:
    batch_id_conflict_details = "Requests data not available"

# Check Master logs for conflict indicators
master_log_conflict = False
for log_file in ["flexlb_master_initial.log", "flexlb_master_restart.log"]:
    log_path = run_dir / log_file
    if log_path.exists():
        try:
            log_content = log_path.read_text(encoding="utf-8", errors="ignore").lower()
            if "stale report" in log_content or "duplicate batch" in log_content or "batch id conflict" in log_content:
                master_log_conflict = True
                batch_id_conflict_details += f" (conflict indicator found in {log_file})"
        except Exception:
            pass

if master_log_conflict:
    batch_id_conflict = True

# -- Pass/Fail determination --
test_passed = True
fail_reasons = []

if total == 0:
    test_passed = False
    fail_reasons.append("no requests were recorded (load client may have failed to start)")

if ok_count == 0 and total > 0:
    test_passed = False
    fail_reasons.append("no successful requests (system may not be working)")

# Hard assertion 1: Post-restart endpoint inflight must be 0
if post_restart_in:
    if not post_restart_in.get("prefill_clean", True):
        test_passed = False
        fail_reasons.append(f"Post-restart prefill endpoint inflight not clean: {post_restart_in.get('prefill_detail', [])}")
    if not post_restart_in.get("decode_clean", True):
        test_passed = False
        fail_reasons.append(f"Post-restart decode endpoint inflight not clean: {post_restart_in.get('decode_detail', [])}")

# Hard assertion 2: Recovery verification success rate must be >= 95%
if recovery_total > 0 and recovery_success_rate < 95.0:
    test_passed = False
    fail_reasons.append(f"Recovery verification success rate {recovery_success_rate:.1f}% < 95% threshold")

# Hard assertion 3: Mock engine no abnormal cancelled
if post_restart_res:
    for engine in post_restart_res.get("engines", []):
        if engine.get("cancelled", 0) > 0:
            test_passed = False
            fail_reasons.append(f"Mock engine {engine.get('name', 'unknown')} has {engine.get('cancelled', 0)} cancelled requests")

# Hard assertion 4: Recovery verification TTFT degradation <= 50%
if isinstance(ttft_degradation, (int, float)) and ttft_degradation > 50.0:
    test_passed = False
    fail_reasons.append(f"Recovery TTFT degradation {ttft_degradation:.1f}% > 50% threshold (baseline={baseline_ttft_p50}ms, recovery={recovery_ttft_p50}ms)")

# Hard assertion 5: Batch ID conflict
if batch_id_conflict:
    test_passed = False
    fail_reasons.append(f"Batch ID conflict detected: {batch_id_conflict_details}")

if not test_passed and not fail_reasons:
    fail_reasons.append("unknown failure")

# -- Generate Markdown report --
lines = []
w = lines.append

w("# FlexLB Master Kill-Restart Destructive Test Report")
w("")
w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w(f"**Run Directory**: `{run_dir}`")
w("")

w("## 1. Environment")
w("")
w("| Parameter | Value |")
w("|---|---|")
w(f"| Master PID (initial) | {master_pid_initial} |")
w(f"| Master PID (restart) | {master_pid_restart} |")
w(f"| Mock Engine PID | {mock_pid} |")
w(f"| Load Client PID | {load_client_pid} |")
w(f"| Kill Timestamp | {kill_ts} |")
w(f"| Restart Timestamp | {restart_ts} |")
w(f"| FLEXLB_HTTP_PORT | 18080 |")
w(f"| FLEXLB_GRPC_PORT | 18082 (HTTP+2) |")
w(f"| FLEXLB_MANAGEMENT_PORT | 18081 |")
w(f"| MOCK_HTTP_PORT | 55150 |")
w(f"| MOCK_BASE_GRPC_PORT | 55151 |")
w(f"| Schedule Mode | batch |")
w(f"| N_PREFILL | {os.environ.get('N_PREFILL', '2')} |")
w(f"| N_DECODE | {os.environ.get('N_DECODE', '2')} |")
w(f"| KILL_TIMING | {kill_timing} |")
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
w(f"| Schedule Errors | {schedule_error_count} |")
w(f"| Exceptions | {exception_count} |")
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

if final_summary:
    w("### Latency Summary (from summary.json)")
    w("")
    for key in ("schedule_latency_ms", "ttft_ms", "total_ms"):
        lat = final_summary.get(key)
        if lat:
            w(f"- **{key}**: p50={lat.get('p50', 'N/A')}, p90={lat.get('p90', 'N/A')}, p99={lat.get('p99', 'N/A')}")
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
write_inflight_section("Post-Restart", post_restart_in)

w("## 4. Mock Engine Residual Check")
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
w(f"| TTFT p50 (recovery) | {recovery_ttft_p50} |")
w(f"| TTFT p50 (baseline) | {baseline_ttft_p50} |")
ttft_deg_str = f"{ttft_degradation:.1f}%" if isinstance(ttft_degradation, (int, float)) else str(ttft_degradation)
w(f"| TTFT Degradation | {ttft_deg_str} |")
w("")

w("## 6. Batch ID Conflict Check")
w("")
w(f"- Conflict detected: **{batch_id_conflict}**")
w(f"- Details: {batch_id_conflict_details}")
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
w(f"- Master was killed at {kill_ts} and restarted at {restart_ts}")
w(f"- Downtime duration: ~{kill_ts} to {restart_ts}")
if total > 0:
    w(f"- {ok_count}/{total} requests succeeded ({(ok_count/total*100):.1f}%)")
    w(f"- {error_count}/{total} requests failed ({failure_rate:.2f}%)")
    if error_count > 0:
        w(f"- Failures during kill period are expected (Master unavailable)")
    if ok_count > 0:
        w(f"- Successful requests confirm system functionality before/after kill")
if post_restart_res:
    w(f"- Post-restart mock engine running: {post_restart_res['total_running']}")
if post_restart_in:
    w(f"- Post-restart scheduler inflight: {post_restart_in['scheduler_inflight']}")
w(f"- Kill timing mode: {kill_timing}")
if recovery_total > 0:
    w(f"- Recovery verification: {recovery_ok}/{recovery_total} succeeded ({recovery_success_rate:.1f}%)")
    if isinstance(ttft_degradation, (int, float)):
        w(f"- Recovery TTFT degradation: {ttft_degradation:.1f}% (baseline={baseline_ttft_p50}ms, recovery={recovery_ttft_p50}ms)")
w(f"- Batch ID conflict: {batch_id_conflict} ({batch_id_conflict_details})")
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
echo "  Master log (initial): ${RUN_DIR}/flexlb_master_initial.log"
echo "  Master log (restart): ${RUN_DIR}/flexlb_master_restart.log"
echo "  Mock log:  ${RUN_DIR}/mock_engine.log"
echo "  Load client log: ${RUN_DIR}/load_client.log"
echo "=========================================="

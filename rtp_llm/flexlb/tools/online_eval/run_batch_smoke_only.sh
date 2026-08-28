#!/usr/bin/env bash
set -euo pipefail

# Batch-only smoke test runner (subset of run_matrix_smoke.sh for CI verification)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_DIR="${SCRIPT_DIR}/run/batch_only_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
echo "run_dir=${RUN_DIR}"

# -- Config --
N_PREFILL=2
N_DECODE=4
MOCK_BASE_GRPC_PORT=55151
MOCK_HTTP_PORT=$((MOCK_BASE_GRPC_PORT - 1))
FLEXLB_HTTP_PORT=19080
FLEXLB_MANAGEMENT_PORT=19081
FLEXLB_JAR="${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar"

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21}"
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

MOCK_PID=""
FLEXLB_PID=""

cleanup() {
  echo ""
  echo "[cleanup] stopping processes ..."
  if [[ -n "${FLEXLB_PID}" ]]; then
    kill "${FLEXLB_PID}" 2>/dev/null || true
    wait "${FLEXLB_PID}" 2>/dev/null || true
  fi
  if [[ -n "${MOCK_PID}" ]]; then
    kill "${MOCK_PID}" 2>/dev/null || true
    wait "${MOCK_PID}" 2>/dev/null || true
  fi
  echo "[cleanup] done."
}
trap cleanup EXIT

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
print(f"TIMEOUT waiting for {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

# -- Perf config --
cat > "${RUN_DIR}/perf.json" <<'JSON'
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

ENDPOINT_FILE="${RUN_DIR}/endpoints.json"

# -- 1. Start mock engine cluster --
echo "[1/4] Starting mock engine cluster (${N_PREFILL}P + ${N_DECODE}D) ..."
PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/mock_engine_cluster.py" \
  --n-prefill "${N_PREFILL}" \
  --n-decode "${N_DECODE}" \
  --base-grpc-port "${MOCK_BASE_GRPC_PORT}" \
  --performance "${RUN_DIR}/perf.json" \
  --prefill-cache-blocks 6000 \
  --decode-cache-blocks 3000 \
  --endpoint-file "${ENDPOINT_FILE}" \
  --env-file "${RUN_DIR}/flexlb_env.txt" \
  >"${RUN_DIR}/mock_engine.log" 2>&1 &
MOCK_PID="$!"
wait_for_port "127.0.0.1" "${MOCK_BASE_GRPC_PORT}" 20
echo "  mock cluster started (pid=${MOCK_PID})"

# -- Parse env from endpoint file --
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

# -- 2. Start FlexLB master (queue + priority ordering, batch dispatch) --
DEFAULT_FLEXLB_CONFIG='{"schemaVersion":2,"scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY"},"decision":{"type":"FIXED_WINDOW","maxRequests":32,"maxCollectionWaitMs":10,"maxPredictedExecutionMs":550},"capacity":{"maxOutstandingRequestsGlobal":5000}},"dispatcher":{"type":"BATCH","maxInflightBatchesPerPrefillWorker":4},"router":{"availabilityHysteresisPercent":0,"roles":{"prefill":{"availability":{"maxPendingRequests":100000},"candidateChoice":{"type":"RANDOM_WITHIN_TOLERANCE","outlierRejection":{"maxPendingVsAverageMultiplier":1.5,"maxProjectedDrainVsAverageMultiplier":3.0}}},"decode":{"availability":{"maxEngineRequests":132}}}}}'
FLEXLB_CONFIG="${FLEXLB_CONFIG:-${DEFAULT_FLEXLB_CONFIG}}"
echo "[2/4] Starting FlexLB master (BATCH / COST_BASED_PREFILL) ..."
env ${FLEXLB_ENV_ARGS[@]+"${FLEXLB_ENV_ARGS[@]}"} \
  "FLEXLB_CONFIG=${FLEXLB_CONFIG}" \
  "OTEL_TRACE_SKIP_PATTERN=.*" \
  "OTEL_EXPORTER_OTLP_ENDPOINT=none" \
  "HIPPO_ROLE=flexlb_batch_smoke" \
  "FLEXLB_EXPECT_FETCH_RESPONSE=true" \
  java "${JAVA_MODULE_OPTS[@]}" -jar "${FLEXLB_JAR}" \
  --server.port="${FLEXLB_HTTP_PORT}" \
  --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
  >"${RUN_DIR}/flexlb.log" 2>&1 &
FLEXLB_PID="$!"
wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60
echo "  master started (pid=${FLEXLB_PID})"

# -- 3. Run batch smoke tests --
echo ""
echo "[3/4] Running batch smoke tests ..."
TOTAL_PASS=0
TOTAL_FAIL=0

run_suite() {
  local name="$1" script="$2" rid_base="$3"
  local cmd_args=(
    --master-ip 127.0.0.1
    --master-http-port "${FLEXLB_HTTP_PORT}"
    --flexlb-http-port "${FLEXLB_HTTP_PORT}"
    --request-id-base "${rid_base}"
  )
  if [[ "${script}" != "cancel_smoke.py" ]]; then
    cmd_args+=(--mock-http-port "${MOCK_HTTP_PORT}")
  fi
  echo ""
  echo "  --- ${name} ---"
  set +e
  PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/${script}" \
    "${cmd_args[@]}" 2>&1 | tee "${RUN_DIR}/${name}.stdout"
  local exit_code=${PIPESTATUS[0]}
  set -e
  if [[ "${exit_code}" -eq 0 ]]; then
    echo "  ${name}: PASS"
    TOTAL_PASS=$((TOTAL_PASS + 1))
  else
    echo "  ${name}: FAIL (exit=${exit_code})"
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
  fi
}

run_suite "cancel_smoke"     "cancel_smoke.py"     10000
run_suite "scheduling_smoke" "scheduling_smoke.py" 20000
run_suite "anomaly_smoke"    "anomaly_smoke.py"    30000

# -- 4. Summary --
echo ""
echo "[4/4] Batch Smoke Summary:"
echo "=========================================="
echo "  Passed: ${TOTAL_PASS}/3"
echo "  Failed: ${TOTAL_FAIL}/3"
echo "  Logs: ${RUN_DIR}/"
echo "=========================================="

if [[ "${TOTAL_FAIL}" -gt 0 ]]; then
  exit 1
fi

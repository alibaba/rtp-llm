#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# run_cancel_smoke.sh — One-shot orchestration for FlexLB cancel smoke tests.
#
# 1. Start the single-JVM Java mock engine cluster (default 1 prefill + 1
#    decode, cancel-tuned performance profile) via lib_load_client.sh
# 2. Generate service-discovery env vars
# 3. Build + start flexlb-api master (Java) with the strict FLEXLB_CONFIG
#    JSON document (schemaVersion 2, priority preemption enabled)
# 4. Run the independent RUNNING-victim priority-preemption scenario
# 5. Optionally run the unchanged six client-cancel scenarios
# 6. Collect results and cleanup (trap EXIT)
#
# The smoke clients (priority_preemption_smoke.py / cancel_smoke.py) are
# gRPC clients: they reach the engines through the master's Schedule
# responses and drive the mock cluster's HTTP control API
# (MOCK_BASE_GRPC_PORT - 1), both of which the Java cluster implements with
# the Python-compatible schema. They are therefore mock-agnostic; only the
# mock engine cluster itself is Java here.
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${FLEXLB_DIR}/../.." && pwd)"

# Shared Java mock jar / JavaLoadClient helpers (start_java_mock_cluster /
# wait_mock_cluster_ready / stop_java_mock_cluster / java_major). The lib
# requires FLEXLB_DIR to be exported before sourcing.
export FLEXLB_DIR
source "${SCRIPT_DIR}/lib_load_client.sh"

# -- Configurable parameters ----------------------------------------------

RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run}"
RUN_ID="${RUN_ID:-cancel_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"

N_PREFILL="${N_PREFILL:-1}"
N_DECODE="${N_DECODE:-1}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-55151}"
PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS:-6000}"
DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS:-3000}"

# Java mock engine cluster JVM sizing. The cancel topology is tiny (1P+1D),
# so default to 1g instead of the lib's 4g; override via env if needed.
MOCK_JVM_XMS="${MOCK_JVM_XMS:-1g}"
MOCK_JVM_XMX="${MOCK_JVM_XMX:-1g}"

FLEXLB_HTTP_PORT="${FLEXLB_HTTP_PORT:-18080}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-18081}"
FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"

START_FLEXLB="${START_FLEXLB:-1}"
START_MOCK="${START_MOCK:-1}"
BUILD_FLEXLB="${BUILD_FLEXLB:-1}"
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"

SCHEDULE_MODE="${SCHEDULE_MODE:-batch}"
# The six legacy scenarios exercise the now-separate frontend/client Cancel
# API.  Keep them intact and runnable, but do not make them part of this
# priority-preemption acceptance by default.
RUN_CLIENT_CANCEL_SMOKE="${RUN_CLIENT_CANCEL_SMOKE:-0}"

# Performance config for cancel tests: enough delay for cancel window.
# prefill=100ms fixed, decode=20ms/step × 10 steps = 200ms total decode.
# Field names follow MockPerformanceModel's JSON schema
# (prefill.fixed_ms / decode.step_ms_by_batch / block_size / sleep_scale).
PERF_CONFIG_DIR="${RUN_DIR}/perf"
PERF_CONFIG_FILE="${PERF_CONFIG_DIR}/cancel_smoke_perf.json"

# The Java mock CLI requires --master-config, but the mock only uses that
# document to extract the prefill execution-time FORMULA
# (MockPerformanceModel.loadPrefillExpression), and a formula takes
# precedence over the perf file's prefill.fixed_ms. The Python cluster
# accepted no master config, so to keep "prefill = fixed 100ms" we generate
# a minimal master config with no FLEXLB_CONFIG env (=> no estimator
# formula => fixed_ms stays authoritative). Point MOCK_MASTER_CONFIG at an
# existing file (e.g. data/config/master_fixed_window.json) to switch the
# mock to formula-based prefill timing instead.
MOCK_MASTER_CONFIG="${MOCK_MASTER_CONFIG:-${RUN_DIR}/mock_master_config.json}"

# FlexLB has one strict JSON configuration surface. Callers may replace this
# document wholesale through FLEXLB_CONFIG. allowedVictimStages must keep
# DECODE_ENGINE_OWNED: the default priority_preemption_smoke below is the
# only end-to-end coverage of RUNNING-victim priority preemption.
DEFAULT_FLEXLB_CONFIG='{"schemaVersion":2,"scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY","defaultPriority":50,"preemption":{"allowedVictimStages":["DECODE_RESERVED","DECODE_ENGINE_OWNED"],"engineCancellation":{"ackTimeoutMs":50,"completionTimeoutMs":1000}}},"decision":{"type":"FIXED_WINDOW","maxRequests":32,"maxCollectionWaitMs":10,"maxPredictedExecutionMs":550},"capacity":{"maxOutstandingRequestsGlobal":5000}},"dispatcher":{"type":"BATCH","maxInflightBatchesPerPrefillWorker":4,"enqueueRpcTimeoutMs":5000},"router":{"availabilityHysteresisPercent":30,"roles":{"prefill":{"availability":{"maxPendingRequests":100000},"executionTimeEstimator":{"type":"FORMULA"},"selector":{"type":"ESTIMATED_TTFT","candidateChoice":{"type":"RANDOM_WITHIN_TOLERANCE"}}},"decode":{"availability":{"maxKvUsagePercent":90,"maxEngineRequests":1},"kvReservation":{"maxOutputTokensForEstimate":1000},"selector":{"type":"KV_USAGE_WEIGHTED_RANDOM"}},"vit":{"selector":{"type":"RANDOM"}}}}}'
FLEXLB_CONFIG="${FLEXLB_CONFIG:-${DEFAULT_FLEXLB_CONFIG}}"

OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN:-.*}"
OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-none}"
HIPPO_ROLE="${HIPPO_ROLE:-flexlb_cancel_smoke_master}"

# Engine-Cancel transport for priority preemption. The Java mock cluster
# now implements the gRPC RpcService/Cancel method too (added with the
# Java-only intake): the gRPC Cancel handler, the HTTP /cancel_request
# endpoint and the in-process channel share one cancel contract, so either
# transport can carry the cancel intent. This suite still defaults to the
# HTTP control plane — the explicit cross-process wiring the
# accepted-eviction (8429) path was wired to test:
# flexlb.test.mock-cancel-control-url selects HttpMockEngineCancelChannel,
# which POSTs {port, request_id} to <url>/cancel_request. Defaults to the
# control port (MOCK_BASE_GRPC_PORT - 1); set to an explicit URL when
# reusing an externally started cluster, or empty ("") to fall back to the
# production gRPC channel (now served by the Java mock's Cancel handler).
MOCK_CANCEL_CONTROL_URL="${MOCK_CANCEL_CONTROL_URL:-http://127.0.0.1:$((MOCK_BASE_GRPC_PORT - 1))}"

PREFILL_REQUEST_CAP="${PREFILL_REQUEST_CAP:-0}"

# -- Internal state --------------------------------------------------------

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
  # Idempotent: no-op when no mock_engine.pid exists (e.g. START_MOCK=0).
  stop_java_mock_cluster "${RUN_DIR}"
  echo "[cleanup] done."
}
trap cleanup EXIT

# -- Setup -----------------------------------------------------------------

require_java21

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

# Generate the minimal mock master config (see the MOCK_MASTER_CONFIG comment).
if [[ ! -f "${MOCK_MASTER_CONFIG}" ]]; then
  cat > "${MOCK_MASTER_CONFIG}" <<'JSON'
{
  "zone_name": "master",
  "zone_process_setting": {
    "global": {},
    "resource_plan": {
      "resources": [],
      "meta_tag_list": []
    },
    "process_info": {
      "args": [],
      "envs": []
    }
  }
}
JSON
fi
echo "mock_master_config=${MOCK_MASTER_CONFIG}"

# -- Start mock engine cluster --------------------------------------------

if [[ "${START_MOCK}" == "1" ]]; then
  echo ""
  echo "[1/5] Starting Java mock engine cluster (${N_PREFILL} prefill, ${N_DECODE} decode) ..."
  export MOCK_N_PREFILL="${N_PREFILL}"
  export MOCK_N_DECODE="${N_DECODE}"
  export MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT}"
  export MOCK_PERFORMANCE_FILE="${PERF_CONFIG_FILE}"
  export MOCK_MASTER_CONFIG="${MOCK_MASTER_CONFIG}"
  export MOCK_PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS}"
  export MOCK_DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS}"
  export MOCK_ENDPOINT_FILE="${ENDPOINT_FILE}"
  export MOCK_ENV_FILE="${FLEXLB_ENV_FILE}"
  start_java_mock_cluster "${RUN_DIR}"
  if ! wait_mock_cluster_ready "${MOCK_BASE_GRPC_PORT}" "$((N_PREFILL + N_DECODE))" 60; then
    echo "Java mock engine cluster failed to become ready" >&2
    tail -50 "${RUN_DIR}/mock_engine.log" >&2 || true
    exit 1
  fi
  echo "  mock cluster started (pid=$(cat "${RUN_DIR}/mock_engine.pid"))"
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
  echo "[2/5] Starting flexlb-api master ..."

  if [[ "$(java_major java)" -lt 21 ]]; then
    echo "Java 21 is required to build/start flexlb-api. Set JAVA21_HOME or JAVA_HOME." >&2
    exit 1
  fi

  if [[ "${BUILD_FLEXLB}" == "1" ]]; then
    echo "  Building flexlb-api (mvnw) ..."
    (cd "${FLEXLB_DIR}" && ./mvnw -P"${MAVEN_PROFILES}" -pl flexlb-api -am package -DskipTests)
  elif [[ ! -f "${FLEXLB_JAR}" ]]; then
    echo "BUILD_FLEXLB=0 but FlexLB jar does not exist: ${FLEXLB_JAR}" >&2
    exit 1
  fi

  cancel_control_args=()
  if [[ -n "${MOCK_CANCEL_CONTROL_URL}" ]]; then
    cancel_control_args=("--flexlb.test.mock-cancel-control-url=${MOCK_CANCEL_CONTROL_URL}")
  fi

  env ${FLEXLB_ENV_ARGS[@]+"${FLEXLB_ENV_ARGS[@]}"} \
    "FLEXLB_CONFIG=${FLEXLB_CONFIG}" \
    "OTEL_TRACE_SKIP_PATTERN=${OTEL_TRACE_SKIP_PATTERN}" \
    "OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT}" \
    "HIPPO_ROLE=${HIPPO_ROLE}" \
    "FLEXLB_EXPECT_FETCH_RESPONSE=true" \
    java "${JAVA_MODULE_OPTS[@]}" -jar "${FLEXLB_JAR}" \
    --server.port="${FLEXLB_HTTP_PORT}" \
    --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
    --spring.profiles.active="${SPRING_PROFILE:-default}" \
    "${cancel_control_args[@]+${cancel_control_args[@]}}" \
    >"${RUN_DIR}/flexlb.log" 2>&1 &
  FLEXLB_PID="$!"
  echo "  master starting (pid=${FLEXLB_PID}), waiting for port ${FLEXLB_HTTP_PORT} ..."
  wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60
  echo "  master started"
else
  echo "  [skipped] flexlb-api master already running"
fi

# -- Run priority-preemption acceptance -----------------------------------

echo ""
echo "[3/5] Running RUNNING-victim priority-preemption smoke test ..."
echo ""

set +e
PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/priority_preemption_smoke.py" \
  --master-ip 127.0.0.1 \
  --master-http-port "${FLEXLB_HTTP_PORT}" \
  --mock-http-port "$((MOCK_BASE_GRPC_PORT - 1))" \
  --flexlb-http-port "${FLEXLB_HTTP_PORT}" \
  --schedule-mode "${SCHEDULE_MODE}" \
  --prefill-request-cap "${PREFILL_REQUEST_CAP}" \
  2>&1 | tee "${RUN_DIR}/priority_preemption_smoke.stdout"

PRIORITY_SMOKE_EXIT="${PIPESTATUS[0]}"
set -e

echo ""
echo "[4/5] Existing client-cancel smoke tests ..."
echo ""

SMOKE_EXIT=0
if [[ "${RUN_CLIENT_CANCEL_SMOKE}" == "1" ]]; then
  set +e
  PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/cancel_smoke.py" \
    --master-ip 127.0.0.1 \
    --master-http-port "${FLEXLB_HTTP_PORT}" \
    --mock-http-port "$((MOCK_BASE_GRPC_PORT - 1))" \
    --flexlb-http-port "${FLEXLB_HTTP_PORT}" \
    --schedule-mode "${SCHEDULE_MODE}" \
    2>&1 | tee "${RUN_DIR}/cancel_smoke.stdout"
  SMOKE_EXIT="${PIPESTATUS[0]}"
  set -e
else
  echo "  skipped (set RUN_CLIENT_CANCEL_SMOKE=1 to run the unchanged six scenarios)"
fi

# -- Collect results -------------------------------------------------------

echo ""
echo "[5/5] Results:"
echo "  client_cancel_exit_code=${SMOKE_EXIT}"
echo "  priority_preemption_exit_code=${PRIORITY_SMOKE_EXIT}"
echo "  stdout=${RUN_DIR}/cancel_smoke.stdout"
echo "  priority_stdout=${RUN_DIR}/priority_preemption_smoke.stdout"
echo "  mock_log=${RUN_DIR}/mock_engine.log"
echo "  flexlb_log=${RUN_DIR}/flexlb.log"
echo ""

if [[ "${SMOKE_EXIT}" != "0" || "${PRIORITY_SMOKE_EXIT}" != "0" ]]; then
  exit 1
fi
exit 0

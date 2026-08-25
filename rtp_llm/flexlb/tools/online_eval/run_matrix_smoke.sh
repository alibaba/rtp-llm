#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# run_matrix_smoke.sh — Matrix orchestration for FlexLB smoke tests.
#
# Runs three test suites (cancel, scheduling, anomaly) across three
# path/algorithm configurations (batch+fixed_window, direct, queue)
# against a single Java mock engine cluster (2P + 4D).
#
# Flow:
#   1. Start the single-JVM Java mock engine cluster once (reused across
#      all groups) via lib_load_client.sh
#   2. For each group: set FLEXLB_CONFIG (strict JSON, schemaVersion 2) →
#      start master → run 3 suites → stop master
#   3. Summarise pass/fail per group
#   4. cleanup (stop mock cluster)
#
# The smoke clients (cancel/scheduling/anomaly) are gRPC clients: they
# reach the engines through the master's Schedule responses and drive the
# mock cluster's HTTP control API (MOCK_BASE_GRPC_PORT - 1), both of which
# the Java cluster implements with the Python-compatible schema. They are
# therefore mock-agnostic; only the mock engine cluster itself is Java
# here.
#
# Usage:
#   bash run_matrix_smoke.sh
#   START_MOCK=0 ENDPOINT_FILE=... bash run_matrix_smoke.sh  # reuse cluster
#
# Known-diff scenario accounting:
#   The Java mock engine and the retired Python mock have documented
#   semantic differences, so some smoke scenarios fail deterministically
#   against the Java cluster. Scenario IDs with such KNOWN differences are
#   listed in KNOWN_FAIL_SCENARIOS below. A suite whose failing scenarios
#   are all in that list prints "FAIL [known-diff]" and does NOT trip the
#   exit code; any failure outside the list is "failed (new)" and exits 1.
#   Racy scenarios (cancel T2/T6) are deliberately NOT exempted — an
#   intermittent failure must stay visible instead of being silenced.
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${FLEXLB_DIR}/../.." && pwd)"

# Shared Java mock jar / JavaLoadClient helpers (start_java_mock_cluster /
# wait_mock_cluster_ready / stop_java_mock_cluster / java_major). The lib
# requires FLEXLB_DIR to be exported before sourcing.
export FLEXLB_DIR
source "${SCRIPT_DIR}/lib_load_client.sh"

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

# Java mock engine cluster JVM sizing. The matrix topology (2P+4D) is a
# smoke-scale cluster, so default to 2g instead of the lib's 4g; override
# via env if needed.
MOCK_JVM_XMS="${MOCK_JVM_XMS:-2g}"
MOCK_JVM_XMX="${MOCK_JVM_XMX:-2g}"

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

# Scenario IDs whose failures are a KNOWN Java-vs-Python mock semantic
# difference (stable, deterministic). A suite failing only in these IDs is
# reported "FAIL [known-diff]" and does not count toward the exit code;
# failures outside the list are "new" and exit 1. Override via env to
# extend or empty the list (e.g. KNOWN_FAIL_SCENARIOS="" to disable).
#   T3  cancel: multi_request_isolation — single-completion-item model
#       difference vs the Python mock.
#   S1/S4/S6  scheduling: routing-algorithm interaction differences
#       (load_balance_distribution / hotspot_filter / cost_based_determinism).
# Deliberately NOT listed: cancel T2 (cancel_idempotency) and T6
# (cancel_at_prefill_vs_decode) are race-window scenarios — not stable
# enough to silence; anomaly E1-E3 pass against the Java cluster.
KNOWN_FAIL_SCENARIOS="${KNOWN_FAIL_SCENARIOS:-T3 S1 S4 S6}"

FLEXLB_HTTP_PORT="${FLEXLB_HTTP_PORT:-18080}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-18081}"
FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"
START_MOCK="${START_MOCK:-1}"

# -- Common process config -------------------------------------------------

OTEL_TRACE_SKIP_PATTERN="${OTEL_TRACE_SKIP_PATTERN:-.*}"
OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-none}"
HIPPO_ROLE="${HIPPO_ROLE:-flexlb_matrix_smoke_master}"

# -- Internal state --------------------------------------------------------

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

# -- Start mock engine cluster (once, reused across all groups) ------------

if [[ "${START_MOCK}" == "1" ]]; then
  echo ""
  echo "[1/3] Starting Java mock engine cluster (${N_PREFILL}P + ${N_DECODE}D) ..."
  export MOCK_N_PREFILL="${N_PREFILL}"
  export MOCK_N_DECODE="${N_DECODE}"
  export MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT}"
  export MOCK_PERFORMANCE_FILE="${PERF_CONFIG_FILE}"
  export MOCK_MASTER_CONFIG="${MOCK_MASTER_CONFIG}"
  export MOCK_PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS}"
  export MOCK_DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS}"
  export MOCK_ENDPOINT_FILE="${ENDPOINT_FILE}"
  export MOCK_ENV_FILE="${RUN_DIR}/flexlb_env.txt"
  start_java_mock_cluster "${RUN_DIR}"
  if ! wait_mock_cluster_ready "${MOCK_BASE_GRPC_PORT}" "$((N_PREFILL + N_DECODE))" 60; then
    echo "Java mock engine cluster failed to become ready" >&2
    tail -50 "${RUN_DIR}/mock_engine.log" >&2 || true
    exit 1
  fi
  echo "  mock cluster started (pid=$(cat "${RUN_DIR}/mock_engine.pid"), http=${MOCK_HTTP_PORT})"
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

# Sets the one strict FlexLB JSON document (schemaVersion 2) for each matrix
# axis combination. The mock cluster is reused across groups; only the
# master restarts per group with a different FLEXLB_CONFIG.
set_group_config() {
  case "$1" in
    batch)
      SCHEDULING_PROFILE="queue-priority-batch"
      FLEXLB_CONFIG='{"schemaVersion":2,"scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY"},"decision":{"type":"FIXED_WINDOW","maxRequests":32,"maxCollectionWaitMs":10,"maxPredictedExecutionMs":550},"capacity":{"maxOutstandingRequestsGlobal":5000,"maxWaitingRequestsPerPrefillWorker":1024}},"dispatcher":{"type":"BATCH","maxInflightBatchesPerPrefillWorker":4,"enqueueRpcTimeoutMs":5000},"router":{"availabilityHysteresisPercent":0,"roles":{"prefill":{"availability":{"maxPendingRequests":100000},"executionTimeEstimator":{"type":"FORMULA"},"selector":{"type":"ESTIMATED_TTFT","candidateChoice":{"type":"RANDOM_WITHIN_TOLERANCE","outlierRejection":{"maxPendingVsAverageMultiplier":1.5,"maxWaitVsAverageMultiplier":3.0}}}},"decode":{"availability":{"maxKvUsagePercent":90,"maxEngineRequests":132},"kvReservation":{"maxOutputTokensForEstimate":1000},"selector":{"type":"KV_USAGE_WEIGHTED_RANDOM"}},"vit":{"selector":{"type":"RANDOM"}}}}}'
      TEST_RID_BASES=(10000 20000 30000)
      ;;
    direct)
      SCHEDULING_PROFILE="direct-non-batch"
      FLEXLB_CONFIG='{"schemaVersion":2,"scheduler":{"type":"DIRECT"},"dispatcher":{"type":"NON_BATCH"},"router":{"availabilityHysteresisPercent":0,"roles":{"prefill":{"availability":{"maxPendingRequests":100000},"executionTimeEstimator":{"type":"FORMULA"},"selector":{"type":"ESTIMATED_TTFT","candidateChoice":{"type":"LEAST_RECENTLY_USED_IN_POOL","pool":{"type":"RATIO","ratio":0.3,"minimumWorkers":1}}}},"decode":{"availability":{"maxKvUsagePercent":90,"maxEngineRequests":132},"kvReservation":{"maxOutputTokensForEstimate":1000},"selector":{"type":"KV_USAGE_WEIGHTED_RANDOM"}},"vit":{"selector":{"type":"RANDOM"}}}}}'
      TEST_RID_BASES=(40000 50000 60000)
      ;;
    queue)
      SCHEDULING_PROFILE="queue-fifo-non-batch"
      FLEXLB_CONFIG='{"schemaVersion":2,"scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},"decision":{"type":"SINGLE"},"capacity":{"maxOutstandingRequestsGlobal":5000}},"dispatcher":{"type":"NON_BATCH"},"router":{"availabilityHysteresisPercent":0,"roles":{"prefill":{"availability":{"maxPendingRequests":100000},"executionTimeEstimator":{"type":"FORMULA"},"selector":{"type":"ESTIMATED_TTFT","candidateChoice":{"type":"LEAST_RECENTLY_USED_IN_POOL","pool":{"type":"RATIO","ratio":0.3,"minimumWorkers":1}}}},"decode":{"availability":{"maxKvUsagePercent":90,"maxEngineRequests":132},"kvReservation":{"maxOutputTokensForEstimate":1000},"selector":{"type":"KV_USAGE_WEIGHTED_RANDOM"}},"vit":{"selector":{"type":"RANDOM"}}}}}'
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
  echo "  starting master (group=${group}, profile=${SCHEDULING_PROFILE}) ..."
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
#
# On suite failure, parses the suite stdout for per-scenario result lines
# (flexlb_smoke_base's "<<< <ID>: <name>: FAIL" contract) and classifies
# every failing scenario ID against KNOWN_FAIL_SCENARIOS. Sets the global
# SUITE_OUTCOME to one of:
#   pass  — suite exited 0
#   known — suite failed, but every failing scenario ID is in the known list
#   new   — suite failed with at least one scenario outside the known list
#           (or with no parseable scenario lines at all, e.g. a crash)
# Only "new" propagates a non-zero return value.
SUITE_OUTCOME="pass"
run_test_suite() {
  local name="$1" script="$2" rid_base="$3" group="$4"
  local group_dir="${RUN_DIR}/${group}"
  echo ""
  echo "  --- ${name} (rid_base=${rid_base}) ---"
  local cmd_args=(
    --master-ip 127.0.0.1
    --master-http-port "${FLEXLB_HTTP_PORT}"
    --flexlb-http-port "${FLEXLB_HTTP_PORT}"
    --request-id-base "${rid_base}"
  )
  # All three smoke clients accept --mock-http-port (cancel_smoke.py needs
  # it to verify engine-side cancellation state via /snapshot; without it the
  # Python default 55150 silently decouples from MOCK_BASE_GRPC_PORT - 1 and
  # every engine-side assertion fails on a non-default port).
  cmd_args+=(--mock-http-port "${MOCK_HTTP_PORT}")
  set +e
  PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/${script}" \
    "${cmd_args[@]}" 2>&1 | tee "${group_dir}/${name}.stdout"
  exit_code=${PIPESTATUS[0]}
  set -e
  if [[ "${exit_code}" -eq 0 ]]; then
    SUITE_OUTCOME="pass"
    echo "  ${name}: PASS"
    return 0
  fi
  # Classify the failing scenario IDs from the per-scenario result lines.
  local fail_ids=() known_fails=() new_fails=() id
  while IFS= read -r id; do
    if [[ -n "${id}" ]]; then
      fail_ids+=("${id}")
    fi
  done < <(sed -n 's/^<<< \([A-Z][A-Z]*[0-9][0-9]*\):.*: FAIL  .*/\1/p' \
    "${group_dir}/${name}.stdout" | sort -u)
  for id in ${fail_ids[@]+"${fail_ids[@]}"}; do
    if [[ " ${KNOWN_FAIL_SCENARIOS} " == *" ${id} "* ]]; then
      known_fails+=("${id}")
    else
      new_fails+=("${id}")
    fi
  done
  for id in ${known_fails[@]+"${known_fails[@]}"}; do
    echo "  ${name} scenario ${id}: FAIL [known-diff]"
  done
  for id in ${new_fails[@]+"${new_fails[@]}"}; do
    echo "  ${name} scenario ${id}: FAIL (new)"
  done
  if [[ ${#known_fails[@]} -gt 0 && ${#new_fails[@]} -eq 0 ]]; then
    SUITE_OUTCOME="known"
    echo "  ${name}: FAIL [known-diff] (all failing scenarios known; exit=${exit_code})"
    return 0
  fi
  SUITE_OUTCOME="new"
  if [[ ${#fail_ids[@]} -eq 0 ]]; then
    echo "  ${name}: FAIL (new) — no per-scenario result lines parsed (exit=${exit_code})"
  else
    echo "  ${name}: FAIL (new) (new: ${new_fails[*]:-none}, known-diff: ${known_fails[*]:-none}; exit=${exit_code})"
  fi
  return 1
}

# -- Main loop: 3 groups x 3 test suites ------------------------------------

GROUP_NAMES=("batch" "direct" "queue")
TEST_NAMES=("cancel_smoke" "scheduling_smoke" "anomaly_smoke")
TEST_SCRIPTS=("cancel_smoke.py" "scheduling_smoke.py" "anomaly_smoke.py")
TOTAL_PASS=0
TOTAL_FAIL_KNOWN=0
TOTAL_FAIL_NEW=0
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
  group_fail_known=0
  group_fail_new=0
  for i in "${!TEST_NAMES[@]}"; do
    run_test_suite "${TEST_NAMES[$i]}" "${TEST_SCRIPTS[$i]}" "${TEST_RID_BASES[$i]}" "${group}" || true
    case "${SUITE_OUTCOME}" in
      pass)  group_pass=$((group_pass + 1)) ;;
      known) group_fail_known=$((group_fail_known + 1)) ;;
      *)     group_fail_new=$((group_fail_new + 1)) ;;
    esac
  done

  TOTAL_PASS=$((TOTAL_PASS + group_pass))
  TOTAL_FAIL_KNOWN=$((TOTAL_FAIL_KNOWN + group_fail_known))
  TOTAL_FAIL_NEW=$((TOTAL_FAIL_NEW + group_fail_new))
  GROUP_RESULTS+=("${group}: ${group_pass}/3 passed, ${group_fail_known} failed (known-diff), ${group_fail_new} failed (new)")

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
echo "  Total: ${TOTAL_PASS} passed, ${TOTAL_FAIL_KNOWN} failed (known-diff), ${TOTAL_FAIL_NEW} failed (new) (out of $((TOTAL_PASS + TOTAL_FAIL_KNOWN + TOTAL_FAIL_NEW)) suites)"
echo "  known-diff scenarios: ${KNOWN_FAIL_SCENARIOS:-none}"
echo "  logs: ${RUN_DIR}/<group>/<test>.stdout"
echo "=========================================="

# Known-diff failures are expected Java-vs-Python mock semantic differences
# and do not fail the run; only new (unclassified) failures do.
if [[ "${TOTAL_FAIL_NEW}" -gt 0 ]]; then
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# engine_kill_restart_test.sh
#
# FlexLB Engine kill-restart destructive test — pure Java stack.
#
# Unlike master_kill_restart_test.sh (which kills the Master), this script
# kills mock ENGINE processes while the Master stays running.
#
# Architecture note (Java stack):
#   JavaMockEngineCluster runs all engines of one deployment inside a single
#   JVM, so a per-engine kill needs TWO cluster instances:
#     * the "main" cluster JVM hosting the surviving engines, and
#     * the "victim" cluster JVM — a second start_java_mock_cluster instance
#       hosting exactly ONE engine, of the killed role only (KILL_TARGET=
#       prefill -> 1P+0D, decode -> 0P+1D), started on a disjoint port range.
#       Only the tested engine dies on kill -9: hosting the surviving role in
#       the victim JVM would silently kill a healthy engine as well and shrink
#       that role's capacity, polluting the recovery assertions.
#   The victim JVM is killed with kill -9 (not a graceful stop) and later
#   restarted with the same parameters (a fresh start_java_mock_cluster
#   instance in a separate run dir).  The Master discovers all engines via
#   static DOMAIN_ADDRESS env vars (no health check, no dynamic removal):
#   PREFILL_DOMAIN_ADDR / DECODE_DOMAIN_ADDR are the comma-joined engine
#   HTTP addresses (grpc port - 1) of cluster + victim, and the Master
#   computes each engine's gRPC port via toGrpcPort(httpPort) = httpPort + 1.
#   Load generation uses the Java JavaLoadClient via run_java_load_client.
#
#   lib_load_client.sh is the single source of truth for the cluster and
#   load-client lifecycles.  Multi-instance pitfall: start_java_mock_cluster
#   exports MOCK_ENDPOINT_FILE / MOCK_ENV_FILE / MOCK_CONTROL_PORT globally,
#   so a second call overwrites the first call's values.  This script
#   therefore re-assigns every MOCK_* input (including explicit
#   MOCK_ENDPOINT_FILE / MOCK_ENV_FILE paths) before each instance start and
#   snapshots the artifacts it cares about into CLUSTER_* / VICTIM_* variables
#   right after each call.  Note: the MOCK_* assignments must be plain
#   assignments, NOT `VAR=... func` env-prefix calls — bash revokes the lib's
#   exports when the function returns, leaving the variables unset (set -u).
#
# Flow:
#   1.  Start Java mock engine cluster (surviving engines)
#   2.  Start victim cluster JVM (standalone process)
#   3.  Start FlexLB Master (batch path, FLEXLB_CONFIG strict JSON)
#   4.  Start Java load client (background)
#   5.  Wait for steady state
#   6.  Collect baseline data
#   7.  KILL victim engine (kill -9)
#   8.  Wait (observe failures during downtime)
#   9.  Collect kill-period data (Master still alive? routing?)
#   10. Restart victim engine (fresh cluster JVM, same parameters)
#   11. Wait (observe recovery)
#   12. Stop load client
#   13. Recovery verification (100 short requests)
#   14. Collect post-restart data
#   15. Generate test report with 5 hard assertions
#
#   The 5 hard assertions (see Step 15):
#     1. Master did not crash during the kill period
#     2. Surviving engines kept accepting requests (multi)
#        OR Master gracefully degraded (single)
#     3. Post-restart inflight of the killed role = 0
#     4. Recovery success rate >= 95%
#     5. No abnormal cancelled requests on any mock engine
#   test_passed is written to ${RUN_DIR}/test_passed and propagated to the
#   script's exit code (0 = PASS, 1 = FAIL).
#
# Usage:
#   bash engine_kill_restart_test.sh                          # multi, kill prefill
#   KILL_TARGET=decode bash engine_kill_restart_test.sh       # multi, kill decode
#   ENGINE_MODE=single bash engine_kill_restart_test.sh       # single, kill prefill
#   ENGINE_MODE=single KILL_TARGET=decode bash engine_kill_restart_test.sh
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export FLEXLB_DIR

# Shared Java mock cluster / JavaLoadClient helpers (start_java_mock_cluster,
# wait_mock_cluster_ready, stop_java_mock_cluster, mock_http,
# run_java_load_client, require_java21, ensure_java_mock_engine_jar).
source "${SCRIPT_DIR}/lib_load_client.sh"

FLEXLB_JAR="${FLEXLB_JAR:-${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar}"
TRACE_FILE="${TRACE_FILE:-${SCRIPT_DIR}/data/online_logs/trace_30min.jsonl}"

# -- Java setup ------------------------------------------------------------

require_java21

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

if [[ "${ENGINE_MODE}" != "multi" && "${ENGINE_MODE}" != "single" ]]; then
  echo "ERROR: ENGINE_MODE must be 'multi' or 'single' (got: '${ENGINE_MODE}')" >&2
  exit 1
fi
if [[ "${KILL_TARGET}" != "prefill" && "${KILL_TARGET}" != "decode" ]]; then
  echo "ERROR: KILL_TARGET must be 'prefill' or 'decode' (got: '${KILL_TARGET}')" >&2
  exit 1
fi

if [[ "${ENGINE_MODE}" == "single" ]]; then
  N_PREFILL_TOTAL="${N_PREFILL_TOTAL:-1}"
  N_DECODE_TOTAL="${N_DECODE_TOTAL:-1}"
else
  N_PREFILL_TOTAL="${N_PREFILL_TOTAL:-2}"
  N_DECODE_TOTAL="${N_DECODE_TOTAL:-2}"
fi

# Victim JVM engine counts.  The victim mirrors the v2 standalone-engine
# victim: exactly ONE engine, of the killed role only (v2 sets
# victim_n_prefill=1 / victim_n_decode=0 for KILL_TARGET=prefill, and the
# mirrored pair for decode).  kill -9 must destroy exactly one engine of the
# tested role.  Override with VICTIM_N_PREFILL / VICTIM_N_DECODE for custom
# topologies (multi-engine victims are allowed; every victim engine dies
# together on kill -9).
if [[ -n "${VICTIM_N_PREFILL:-}" || -n "${VICTIM_N_DECODE:-}" ]]; then
  VICTIM_N_PREFILL="${VICTIM_N_PREFILL:-0}"
  VICTIM_N_DECODE="${VICTIM_N_DECODE:-0}"
elif [[ "${KILL_TARGET}" == "prefill" ]]; then
  VICTIM_N_PREFILL=1
  VICTIM_N_DECODE=0
else
  VICTIM_N_PREFILL=0
  VICTIM_N_DECODE=1
fi

# The main cluster hosts the remaining engines.
CLUSTER_N_PREFILL=$((N_PREFILL_TOTAL - VICTIM_N_PREFILL))
CLUSTER_N_DECODE=$((N_DECODE_TOTAL - VICTIM_N_DECODE))

# Topology sanity checks (fail fast with a clear message).
if (( VICTIM_N_PREFILL < 0 || VICTIM_N_DECODE < 0 )); then
  echo "ERROR: VICTIM_N_PREFILL/VICTIM_N_DECODE must be >= 0" >&2
  exit 1
fi
if (( VICTIM_N_PREFILL + VICTIM_N_DECODE < 1 )); then
  echo "ERROR: victim JVM must host at least one engine" >&2
  exit 1
fi
if (( CLUSTER_N_PREFILL < 0 || CLUSTER_N_DECODE < 0 )); then
  echo "ERROR: N_PREFILL_TOTAL/N_DECODE_TOTAL too small for the victim topology" >&2
  exit 1
fi
if (( CLUSTER_N_PREFILL + CLUSTER_N_DECODE < 1 )); then
  echo "ERROR: main cluster JVM must host at least one engine" >&2
  exit 1
fi
if [[ "${KILL_TARGET}" == "prefill" && "${VICTIM_N_PREFILL}" -lt 1 ]]; then
  echo "ERROR: KILL_TARGET=prefill requires VICTIM_N_PREFILL >= 1" >&2
  exit 1
fi
if [[ "${KILL_TARGET}" == "decode" && "${VICTIM_N_DECODE}" -lt 1 ]]; then
  echo "ERROR: KILL_TARGET=decode requires VICTIM_N_DECODE >= 1" >&2
  exit 1
fi
# Assertion 2 (surviving engines) needs a same-role survivor in the cluster;
# when there is none the report falls back to the graceful-degradation check,
# which is the documented semantic of ENGINE_MODE=single.
if [[ "${KILL_TARGET}" == "prefill" ]]; then
  CLUSTER_KILL_ROLE_ENGINES=${CLUSTER_N_PREFILL}
else
  CLUSTER_KILL_ROLE_ENGINES=${CLUSTER_N_DECODE}
fi

MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-55151}"
CLUSTER_CONTROL_PORT=$((MOCK_BASE_GRPC_PORT - 1))
VICTIM_BASE_GRPC_PORT="${VICTIM_BASE_GRPC_PORT:-$((MOCK_BASE_GRPC_PORT + 150))}"
VICTIM_CONTROL_PORT=$((VICTIM_BASE_GRPC_PORT - 1))
VICTIM_GRPC_PORT_MIN=${VICTIM_BASE_GRPC_PORT}
VICTIM_GRPC_PORT_MAX=$((VICTIM_BASE_GRPC_PORT + VICTIM_N_PREFILL + VICTIM_N_DECODE - 1))
FLEXLB_HTTP_PORT="${FLEXLB_HTTP_PORT:-18080}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-18081}"
PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS:-6000}"
DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS:-3000}"

# Java mock cluster JVM sizing / threading.  The victim is a tiny standalone
# deployment, so it gets a smaller heap than the main cluster.
CLUSTER_JVM_XMS="${CLUSTER_JVM_XMS:-4g}"
CLUSTER_JVM_XMX="${CLUSTER_JVM_XMX:-4g}"
VICTIM_JVM_XMS="${VICTIM_JVM_XMS:-1g}"
VICTIM_JVM_XMX="${VICTIM_JVM_XMX:-1g}"
MOCK_EVENT_LOOP_THREADS="${MOCK_EVENT_LOOP_THREADS:-8}"
MOCK_COMPLETION_THREADS="${MOCK_COMPLETION_THREADS:-4}"
VICTIM_EVENT_LOOP_THREADS="${VICTIM_EVENT_LOOP_THREADS:-8}"
VICTIM_COMPLETION_THREADS="${VICTIM_COMPLETION_THREADS:-4}"

# --master-config for the Java mock clusters (discovery-file template only;
# the Master itself is configured through the env vars in start_master).
MOCK_MASTER_CONFIG="${MOCK_MASTER_CONFIG:-${SCRIPT_DIR}/data/config/master_fixed_window.json}"

# Victim engine name (detected from the victim's control /snapshot after
# startup; set VICTIM_NAME explicitly to override detection).  The victim JVM
# names its engines from index 0, so "prefill-0" exists in both the cluster
# and the victim — never identify the victim by name, use its gRPC port
# range instead (see the report generator).
VICTIM_NAME="${VICTIM_NAME:-}"

# -- Compute engine addresses (per JVM, then merged) -----------------------
# Each JVM allocates gRPC ports sequentially starting at its base port:
# prefill-0, prefill-1, ..., decode-0, decode-1, ...  DOMAIN_ADDRESS must
# contain HTTP ports (grpc port - 1); the FlexLB Master computes each gRPC
# port via toGrpcPort(httpPort) = httpPort + 1.
CLUSTER_PREFILL_ADDRS=""
CLUSTER_DECODE_ADDRS=""
_port=${MOCK_BASE_GRPC_PORT}
for ((i = 0; i < CLUSTER_N_PREFILL; i++)); do
  if [[ -z "${CLUSTER_PREFILL_ADDRS}" ]]; then
    CLUSTER_PREFILL_ADDRS="127.0.0.1:$((_port - 1))"
  else
    CLUSTER_PREFILL_ADDRS="${CLUSTER_PREFILL_ADDRS},127.0.0.1:$((_port - 1))"
  fi
  _port=$((_port + 1))
done
for ((i = 0; i < CLUSTER_N_DECODE; i++)); do
  if [[ -z "${CLUSTER_DECODE_ADDRS}" ]]; then
    CLUSTER_DECODE_ADDRS="127.0.0.1:$((_port - 1))"
  else
    CLUSTER_DECODE_ADDRS="${CLUSTER_DECODE_ADDRS},127.0.0.1:$((_port - 1))"
  fi
  _port=$((_port + 1))
done
VICTIM_PREFILL_ADDRS=""
VICTIM_DECODE_ADDRS=""
_vport=${VICTIM_BASE_GRPC_PORT}
for ((i = 0; i < VICTIM_N_PREFILL; i++)); do
  if [[ -z "${VICTIM_PREFILL_ADDRS}" ]]; then
    VICTIM_PREFILL_ADDRS="127.0.0.1:$((_vport - 1))"
  else
    VICTIM_PREFILL_ADDRS="${VICTIM_PREFILL_ADDRS},127.0.0.1:$((_vport - 1))"
  fi
  _vport=$((_vport + 1))
done
for ((i = 0; i < VICTIM_N_DECODE; i++)); do
  if [[ -z "${VICTIM_DECODE_ADDRS}" ]]; then
    VICTIM_DECODE_ADDRS="127.0.0.1:$((_vport - 1))"
  else
    VICTIM_DECODE_ADDRS="${VICTIM_DECODE_ADDRS},127.0.0.1:$((_vport - 1))"
  fi
  _vport=$((_vport + 1))
done

# comma_join <a> <b>: join two possibly-empty comma-separated lists.
comma_join() {
  if [[ -z "$1" ]]; then
    printf '%s' "$2"
  elif [[ -z "$2" ]]; then
    printf '%s' "$1"
  else
    printf '%s,%s' "$1" "$2"
  fi
}

PREFILL_DOMAIN_ADDR="$(comma_join "${CLUSTER_PREFILL_ADDRS}" "${VICTIM_PREFILL_ADDRS}")"
DECODE_DOMAIN_ADDR="$(comma_join "${CLUSTER_DECODE_ADDRS}" "${VICTIM_DECODE_ADDRS}")"

# -- Model service config (constant JSON) ----------------------------------
readonly MODEL_SERVICE_CONFIG_JSON='{"service_id":"aigc.text-generation.generation.engine_service","load_balance":true,"role_endpoints":[{"group":"mock","prefill_endpoint":{"address":"mock.prefill.hosts.address","protocol":"http","path":"/"},"decode_endpoint":{"address":"mock.decode.hosts.address","protocol":"http","path":"/"}}]}'

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
CLUSTER_RUN_DIR="${RUN_DIR}/cluster"
VICTIM_RUN_DIR="${RUN_DIR}/victim_initial"
VICTIM_RESTART_RUN_DIR="${RUN_DIR}/victim_restart"
MASTER_LOG_DIR="${RUN_DIR}/master_logs"
mkdir -p "${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"

# -- State -----------------------------------------------------------------
CLUSTER_PID=""
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

# Hard-kill every victim JVM pid file (initial + restart).  The destructive
# scenario uses kill -9 semantics for the victim; graceful drain is not part
# of what this test verifies.
kill_victim_hard() {
  local pid_file pid
  for pid_file in "${VICTIM_RUN_DIR}/mock_engine.pid" "${VICTIM_RESTART_RUN_DIR}/mock_engine.pid"; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      echo "[cleanup] kill -9 victim JVM pid ${pid} (${pid_file})"
    fi
    rm -f "${pid_file}"
  done
  VICTIM_PID=""
}

cleanup() {
  echo ""
  echo "[cleanup] stopping processes ..."
  if [[ -n "${LOAD_CLIENT_PID}" ]]; then
    kill "${LOAD_CLIENT_PID}" >/dev/null 2>&1 || true
    wait "${LOAD_CLIENT_PID}" 2>/dev/null || true
    LOAD_CLIENT_PID=""
  fi
  if [[ -n "${FLEXLB_PID}" ]]; then
    kill "${FLEXLB_PID}" >/dev/null 2>&1 || true
    wait "${FLEXLB_PID}" 2>/dev/null || true
    FLEXLB_PID=""
  fi
  kill_victim_hard
  if [[ -n "${CLUSTER_RUN_DIR}" ]]; then
    stop_java_mock_cluster "${CLUSTER_RUN_DIR}"
  fi
  echo "[cleanup] done."
}
trap cleanup EXIT

# start_victim_cluster <run_dir>
#
# Starts the standalone victim JVM via the shared lib helper with the same
# parameters on every call (initial start and restart after kill -9).
#
# start_java_mock_cluster exports MOCK_ENDPOINT_FILE / MOCK_ENV_FILE /
# MOCK_CONTROL_PORT globally: the values from a previous call (e.g. the main
# cluster) would leak into this one, so every instance-defining env var is
# re-assigned explicitly here — including MOCK_ENDPOINT_FILE/MOCK_ENV_FILE,
# otherwise the lib's ${VAR:-default} fallback would reuse the previous
# instance's paths and overwrite its discovery files.  (Plain assignments
# instead of `VAR=... func` env-prefix calls: an env-prefix call would have
# bash revoke the lib's export when the function returns, leaving the
# variables unset under set -u.)
start_victim_cluster() {
  local victim_run_dir="$1"
  echo "  starting victim cluster JVM (${VICTIM_N_PREFILL}P + ${VICTIM_N_DECODE}D, base grpc ${VICTIM_BASE_GRPC_PORT}, run_dir=${victim_run_dir}) ..."
  # Pre-flight: ensure every victim port (control + engines) is free.
  local port
  for port in $(seq "${VICTIM_CONTROL_PORT}" "${VICTIM_GRPC_PORT_MAX}"); do
    check_port_free "${port}" || {
      echo "  attempting to kill process on port ${port} ..." >&2
      local stale_pid
      stale_pid=$(lsof -ti :"${port}" -sTCP:LISTEN 2>/dev/null || true)
      if [[ -n "${stale_pid}" ]]; then
        kill -9 "${stale_pid}" 2>/dev/null || true
        sleep 2
      fi
      check_port_free "${port}" || return 1
    }
  done
  MOCK_N_PREFILL="${VICTIM_N_PREFILL}"
  MOCK_N_DECODE="${VICTIM_N_DECODE}"
  MOCK_BASE_GRPC_PORT="${VICTIM_BASE_GRPC_PORT}"
  MOCK_PERFORMANCE_FILE="${PERF_CONFIG_FILE}"
  MOCK_MASTER_CONFIG="${MOCK_MASTER_CONFIG}"
  MOCK_PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS}"
  MOCK_DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS}"
  MOCK_JVM_XMS="${VICTIM_JVM_XMS}"
  MOCK_JVM_XMX="${VICTIM_JVM_XMX}"
  MOCK_EVENT_LOOP_THREADS="${VICTIM_EVENT_LOOP_THREADS}"
  MOCK_COMPLETION_THREADS="${VICTIM_COMPLETION_THREADS}"
  MOCK_ENDPOINT_FILE="${victim_run_dir}/endpoints.json"
  MOCK_ENV_FILE="${victim_run_dir}/flexlb_env.txt"
  start_java_mock_cluster "${victim_run_dir}"
  VICTIM_PID="$(cat "${victim_run_dir}/mock_engine.pid")"
  wait_mock_cluster_ready "${VICTIM_BASE_GRPC_PORT}" \
    "$((VICTIM_N_PREFILL + VICTIM_N_DECODE))" 60
  # Verify the victim JVM is still alive after the ports came up
  # (guards against the case where gRPC binds first but HTTP fails, causing
  # the JVM to exit).
  if ! kill -0 "${VICTIM_PID}" 2>/dev/null; then
    echo "ERROR: victim cluster JVM died during startup" >&2
    echo "--- victim engine log ---" >&2
    cat "${victim_run_dir}/mock_engine.log" >&2
    return 1
  fi
  # Detect the killed-role engine name from the victim's control /snapshot
  # unless the caller pinned VICTIM_NAME (the victim JVM names its engines
  # from index 0, colliding with cluster engine names).
  if [[ -z "${VICTIM_NAME}" ]]; then
    local _detected
    _detected=$(mock_http GET "${VICTIM_CONTROL_PORT}" /snapshot | \
      python3 -c "
import json, sys
role = sys.argv[1]
try:
    data = json.load(sys.stdin)
    for e in data.get('engines', []):
        if e.get('role') == role:
            print(e.get('name', ''))
            break
except Exception:
    pass
" "${KILL_TARGET}" 2>/dev/null || true)
    if [[ -z "${_detected}" ]]; then
      echo "ERROR: could not detect victim engine name from snapshot" >&2
      return 1
    fi
    VICTIM_NAME="${_detected}"
    echo "  detected victim engine name: ${VICTIM_NAME}"
  fi
  echo "  victim cluster JVM started (pid=${VICTIM_PID}, name=${VICTIM_NAME})"
}

start_master() {
  local log_file="$1"
  # Strict FLEXLB_CONFIG (schemaVersion 2) — the authoritative shape on this
  # branch (see run_online_eval.sh).  scheduler.lifecycle.staleInflightTimeoutMs
  # keeps the inflight TTL short (same intent as the legacy v2
  # FLEXLB_INFLIGHT_TTL_MS=20000): requests stranded on the killed engine are
  # swept before the recovery verification instead of saturating decode
  # capacity with zombie inflight entries.
  local default_flexlb_config='{"schemaVersion":2,"scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY"},"decision":{"type":"FIXED_WINDOW","maxRequests":32,"maxCollectionWaitMs":10,"maxPredictedExecutionMs":550},"capacity":{"maxOutstandingRequestsGlobal":5000},"lifecycle":{"staleInflightTimeoutMs":20000}},"dispatcher":{"type":"BATCH","maxInflightBatchesPerPrefillWorker":4},"router":{"availabilityHysteresisPercent":0,"roles":{"prefill":{"availability":{"maxPendingRequests":100000},"selector":{"type":"ESTIMATED_TTFT","candidateChoice":{"type":"RANDOM_WITHIN_TOLERANCE","outlierRejection":{"maxPendingVsAverageMultiplier":1.5,"maxWaitVsAverageMultiplier":3.0}}}},"decode":{"availability":{"maxEngineRequests":132}}}}}'
  local flexlb_config="${FLEXLB_CONFIG:-${default_flexlb_config}}"
  echo "  starting master ..."
  # Log routing: the jar's logback-spring.xml routes the "flexlbLogger"
  # tree (all scheduler/endpoint events: event=scheduler_inflight_ttl_eviction,
  # event=engine_fence_quarantine_summary, event=endpoint_inflight_ttl_eviction,
  # ...) to ${flexlb.log.path}/flexlb.log with additivity=false, so neither
  # --logging.file.name nor --logging.level.org.flexlb has any effect on it.
  # Point flexlb.log.path at the run dir instead; application.log follows the
  # same property.
  mkdir -p "${MASTER_LOG_DIR}"
  env \
    "MODEL_SERVICE_CONFIG=${MODEL_SERVICE_CONFIG_JSON}" \
    "DOMAIN_ADDRESS:mock.prefill.hosts.address=${PREFILL_DOMAIN_ADDR}" \
    "DOMAIN_ADDRESS:mock.decode.hosts.address=${DECODE_DOMAIN_ADDR}" \
    "FLEXLB_CONFIG=${flexlb_config}" \
    "FLEXLB_EXPECT_FETCH_RESPONSE=true" \
    "OTEL_TRACE_SKIP_PATTERN=.*" \
    "OTEL_EXPORTER_OTLP_ENDPOINT=none" \
    "HIPPO_ROLE=flexlb_engine_kill_test" \
    java "${JAVA_MODULE_OPTS[@]}" -jar "${FLEXLB_JAR}" \
    --server.port="${FLEXLB_HTTP_PORT}" \
    --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
    --flexlb.log.path="${MASTER_LOG_DIR}" \
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
  echo "Build it with: (cd \"${FLEXLB_DIR}\" && ./mvnw -Popensource,!internal -pl flexlb-api -am package -DskipTests)" >&2
  exit 1
fi
if [[ ! -f "${TRACE_FILE}" ]]; then
  echo "ERROR: Trace file not found: ${TRACE_FILE}" >&2
  exit 1
fi
if [[ ! -f "${MOCK_MASTER_CONFIG}" ]]; then
  echo "ERROR: mock cluster master config not found: ${MOCK_MASTER_CONFIG}" >&2
  exit 1
fi
ensure_java_mock_engine_jar || exit 1
java -version 2>&1 | head -1
echo "  JAR: ${FLEXLB_JAR}"
echo "  Trace: ${TRACE_FILE} ($(wc -l < "${TRACE_FILE}") lines)"
echo "  Engine Mode: ${ENGINE_MODE}"
echo "  Kill Target: ${KILL_TARGET}"
echo "  Cluster: ${CLUSTER_N_PREFILL}P + ${CLUSTER_N_DECODE}D (base grpc ${MOCK_BASE_GRPC_PORT}, control http ${CLUSTER_CONTROL_PORT})"
echo "  Victim: ${VICTIM_N_PREFILL}P + ${VICTIM_N_DECODE}D (base grpc ${VICTIM_BASE_GRPC_PORT}, control http ${VICTIM_CONTROL_PORT})"
echo "  Victim gRPC port range: ${VICTIM_GRPC_PORT_MIN}-${VICTIM_GRPC_PORT_MAX}"
echo "  Prefill domain: ${PREFILL_DOMAIN_ADDR}"
echo "  Decode domain: ${DECODE_DOMAIN_ADDR}"
echo "  All prerequisites OK."

# -- Pre-flight port check -------------------------------------------------
echo ""
echo "=== Pre-flight Port Check ==="
_preflight_ports=("${FLEXLB_HTTP_PORT}" "${FLEXLB_MANAGEMENT_PORT}")
for ((i = 0; i < CLUSTER_N_PREFILL + CLUSTER_N_DECODE; i++)); do
  _preflight_ports+=("$((MOCK_BASE_GRPC_PORT + i))")
done
[[ "${CLUSTER_N_PREFILL}" -gt 0 || "${CLUSTER_N_DECODE}" -gt 0 ]] && _preflight_ports+=("${CLUSTER_CONTROL_PORT}")
for ((i = 0; i < VICTIM_N_PREFILL + VICTIM_N_DECODE; i++)); do
  _preflight_ports+=("$((VICTIM_BASE_GRPC_PORT + i))")
done
_preflight_ports+=("${VICTIM_CONTROL_PORT}")
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
# Step 1: Start Java mock engine cluster (surviving engines)
# ===========================================================================

echo ""
echo "=== Step 1: Start Java mock engine cluster (${CLUSTER_N_PREFILL}P + ${CLUSTER_N_DECODE}D) ==="
mkdir -p "${CLUSTER_RUN_DIR}"
# Plain assignments (NOT `VAR=... func` env-prefix calls): start_java_mock_cluster
# exports MOCK_ENDPOINT_FILE / MOCK_ENV_FILE / MOCK_CONTROL_PORT, and with an
# env-prefix call bash revokes that export when the function returns (leaving
# the variables unset under set -u).  Plain assignments keep the exported
# state alive; every later instance start (victim / victim restart) re-assigns
# the full set, so the globals always describe the most recently started JVM.
MOCK_N_PREFILL="${CLUSTER_N_PREFILL}"
MOCK_N_DECODE="${CLUSTER_N_DECODE}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT}"
MOCK_PERFORMANCE_FILE="${PERF_CONFIG_FILE}"
MOCK_MASTER_CONFIG="${MOCK_MASTER_CONFIG}"
MOCK_PREFILL_CACHE_BLOCKS="${PREFILL_CACHE_BLOCKS}"
MOCK_DECODE_CACHE_BLOCKS="${DECODE_CACHE_BLOCKS}"
MOCK_JVM_XMS="${CLUSTER_JVM_XMS}"
MOCK_JVM_XMX="${CLUSTER_JVM_XMX}"
MOCK_EVENT_LOOP_THREADS="${MOCK_EVENT_LOOP_THREADS}"
MOCK_COMPLETION_THREADS="${MOCK_COMPLETION_THREADS}"
MOCK_ENDPOINT_FILE="${CLUSTER_RUN_DIR}/endpoints.json"
MOCK_ENV_FILE="${CLUSTER_RUN_DIR}/flexlb_env.txt"
start_java_mock_cluster "${CLUSTER_RUN_DIR}"
CLUSTER_PID="$(cat "${CLUSTER_RUN_DIR}/mock_engine.pid")"
wait_mock_cluster_ready "${MOCK_BASE_GRPC_PORT}" \
  "$((CLUSTER_N_PREFILL + CLUSTER_N_DECODE))" 60
# Verify the cluster JVM is still alive (guards against a crash during
# startup after the ports came up).
if ! kill -0 "${CLUSTER_PID}" 2>/dev/null; then
  echo "ERROR: mock cluster JVM died during startup" >&2
  echo "--- mock engine log ---" >&2
  cat "${CLUSTER_RUN_DIR}/mock_engine.log" >&2
  exit 1
fi
# The master connects to this cluster through static DOMAIN_ADDRESS env
# vars (direct-address mode — no health check, no dynamic removal), so the
# lib's exported MOCK_ENDPOINT_FILE / MOCK_ENV_FILE have no consumer here;
# the run dir's endpoints.json is only a run-record artifact.
echo "  mock cluster started (pid=${CLUSTER_PID}, control http=${CLUSTER_CONTROL_PORT})"

# ===========================================================================
# Step 2: Start victim cluster JVM (standalone process)
# ===========================================================================

echo ""
echo "=== Step 2: Start victim cluster JVM (standalone) ==="
mkdir -p "${VICTIM_RUN_DIR}"
start_victim_cluster "${VICTIM_RUN_DIR}"

# ===========================================================================
# Step 3: Start FlexLB Master
# ===========================================================================

echo ""
echo "=== Step 3: Start FlexLB Master (batch path) ==="
start_master "${RUN_DIR}/flexlb_master.log"

# ===========================================================================
# Step 4: Start Java load client (background)
# ===========================================================================

echo ""
echo "=== Step 4: Start Java load client ==="
LOAD_CLIENT_DIR="${RUN_DIR}/load_client"
mkdir -p "${LOAD_CLIENT_DIR}"
# The Java client only writes summary.json/per_request.jsonl on a normal
# exit (no signal handling), so bound the replay volume (LIMIT) so it
# finishes by itself around T+40s, still covering steady-state + kill +
# restart windows.  At 20x replay speed 2500 requests take ~37s.
_java_limit="${LOAD_CLIENT_LIMIT}"
if [[ "${_java_limit}" -le 0 ]]; then
  _java_limit=2500
fi
run_java_load_client \
  "TRACE_FILE=${TRACE_FILE}" \
  "TARGET_ADDR=127.0.0.1:${FLEXLB_HTTP_PORT}" \
  "REPLAY_SPEED=${LOAD_CLIENT_REPLAY_SPEED}" \
  "LIMIT=${_java_limit}" \
  "MAX_CONCURRENCY=${LOAD_CLIENT_CONCURRENCY}" \
  "TIMEOUT_MS=${LOAD_CLIENT_TIMEOUT_MS}" \
  "OUTPUT_DIR=${LOAD_CLIENT_DIR}" \
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
mock_http GET "${CLUSTER_CONTROL_PORT}" /snapshot > "${RUN_DIR}/baseline_cluster_snapshot.json" \
  || echo '{"engines":[]}' > "${RUN_DIR}/baseline_cluster_snapshot.json"
echo "  - victim snapshot ..."
mock_http GET "${VICTIM_CONTROL_PORT}" /snapshot > "${RUN_DIR}/baseline_victim_snapshot.json" \
  || echo '{"engines":[]}' > "${RUN_DIR}/baseline_victim_snapshot.json"
echo "  - load client per_request.jsonl ..."
cp "${LOAD_CLIENT_DIR}/per_request.jsonl" "${RUN_DIR}/pre_kill_per_request.jsonl" 2>/dev/null \
  || echo "  NOTE: per_request.jsonl not available yet"

# ===========================================================================
# Step 7: KILL victim engine (kill -9)
# ===========================================================================

echo ""
echo "=== Step 7: KILL victim cluster JVM (kill -9) ==="
KILL_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "  kill timestamp: ${KILL_TS}"
echo "  killing victim JVM (pid=${VICTIM_PID}, victim ${KILL_TARGET} engine=${VICTIM_NAME}) ..."
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
mock_http GET "${CLUSTER_CONTROL_PORT}" /snapshot > "${RUN_DIR}/kill_cluster_snapshot.json" \
  || echo '{"engines":[]}' > "${RUN_DIR}/kill_cluster_snapshot.json"
echo "  - victim snapshot (expected to fail) ..."
mock_http GET "${VICTIM_CONTROL_PORT}" /snapshot > "${RUN_DIR}/kill_victim_snapshot.json" 2>/dev/null \
  || echo '{"engines":[]}' > "${RUN_DIR}/kill_victim_snapshot.json"
echo "  - load client per_request.jsonl ..."
cp "${LOAD_CLIENT_DIR}/per_request.jsonl" "${RUN_DIR}/kill_per_request.jsonl" 2>/dev/null \
  || echo "  NOTE: per_request.jsonl not available"

# ===========================================================================
# Step 10: Restart victim engine
# ===========================================================================

echo ""
echo "=== Step 10: Restart victim cluster JVM ==="
RESTART_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "  restart timestamp: ${RESTART_TS}"
sleep 1  # brief pause to ensure ports are released
mkdir -p "${VICTIM_RESTART_RUN_DIR}"
start_victim_cluster "${VICTIM_RESTART_RUN_DIR}"

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
  echo "  master flexlb log (last 30 lines):"
  tail -30 "${MASTER_LOG_DIR}/flexlb.log" 2>/dev/null || echo "  (no flexlb log available)"
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
# Wait for the scheduler inflight left over from the kill period to drain.
# Stranded requests are only cleaned up by the master's periodic inflight-TTL
# sweep (scheduler.lifecycle.staleInflightTimeoutMs); without waiting, the
# leftover entries keep decode capacity saturated and recovery requests fail
# with NO_DECODE_WORKER.
echo "  waiting for kill-period inflight to drain ..."
for _ in $(seq 1 60); do
  _inflight=$(curl -s "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" 2>/dev/null \
    | python3 -c "import json,sys
try:
    print(json.load(sys.stdin).get('scheduler_inflight', -1))
except Exception:
    print(-1)" 2>/dev/null || echo "-1")
  if [[ "${_inflight}" -ge 0 && "${_inflight}" -le 5 ]]; then
    echo "  inflight drained to ${_inflight}"
    break
  fi
  sleep 2
done
echo "  inflight before recovery verification: ${_inflight}"
RECOVERY_TRACE="${RUN_DIR}/recovery_trace.jsonl"
# Tag each record with a run-unique marker: both load clients derive the
# request_id from the raw record content, so the main client already left
# matching ids in the master's terminal-state map (duplicate request_id),
# which would make assertion 4 fail 0/100.
python3 -c "
import json, time
tag = '${KILL_TS}-' + str(int(time.time() * 1000))
with open('${TRACE_FILE}') as f:
    for line in f:
        req = json.loads(line)
        if req.get('ol', 0) <= 200:
            req['_rt'] = tag
            print(json.dumps(req))
" > "${RECOVERY_TRACE}" 2>/dev/null
RECOVERY_TRACE_LINES=$(wc -l < "${RECOVERY_TRACE}" 2>/dev/null || echo 0)
echo "  recovery trace: ${RECOVERY_TRACE_LINES} short-output requests (ol <= 200)"

RECOVERY_VERIFY_DIR="${RUN_DIR}/recovery_verify"
mkdir -p "${RECOVERY_VERIFY_DIR}"
# Subshell so run_java_load_client's exec only replaces the subshell.
( run_java_load_client \
    "TRACE_FILE=${RECOVERY_TRACE}" \
    "TARGET_ADDR=127.0.0.1:${FLEXLB_HTTP_PORT}" \
    "REPLAY_SPEED=0" \
    "LIMIT=100" \
    "MAX_CONCURRENCY=10" \
    "TIMEOUT_MS=10000" \
    "OUTPUT_DIR=${RECOVERY_VERIFY_DIR}" \
  >"${RUN_DIR}/recovery_verify.log" 2>&1 ) || true
echo "  recovery verification completed"
cat "${RECOVERY_VERIFY_DIR}/summary.json" 2>/dev/null || echo "  NOTE: recovery summary not available"
# Master health check after recovery verification
_post_recovery_health=$(curl -s -o /dev/null -w "%{http_code}" \
  "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/inflight_status" 2>/dev/null || echo "000")
echo "  master health after recovery verification: HTTP ${_post_recovery_health}"
if [[ "${_post_recovery_health}" != "200" ]]; then
  echo "  WARNING: master is not responding after recovery verification!"
  echo "  master flexlb log (last 50 lines):"
  tail -50 "${MASTER_LOG_DIR}/flexlb.log" 2>/dev/null || echo "  (no flexlb log available)"
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
mock_http GET "${CLUSTER_CONTROL_PORT}" /snapshot > "${RUN_DIR}/post_restart_cluster_snapshot.json" \
  || echo '{"engines":[]}' > "${RUN_DIR}/post_restart_cluster_snapshot.json"
echo "  - victim snapshot (restarted JVM) ..."
mock_http GET "${VICTIM_CONTROL_PORT}" /snapshot > "${RUN_DIR}/post_restart_victim_snapshot.json" \
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
CLUSTER_PID="${CLUSTER_PID}" \
CLUSTER_N_PREFILL="${CLUSTER_N_PREFILL}" \
CLUSTER_N_DECODE="${CLUSTER_N_DECODE}" \
CLUSTER_KILL_ROLE_ENGINES="${CLUSTER_KILL_ROLE_ENGINES}" \
VICTIM_NAME="${VICTIM_NAME}" \
VICTIM_N_PREFILL="${VICTIM_N_PREFILL}" \
VICTIM_N_DECODE="${VICTIM_N_DECODE}" \
VICTIM_BASE_GRPC_PORT="${VICTIM_BASE_GRPC_PORT}" \
VICTIM_GRPC_PORT_MIN="${VICTIM_GRPC_PORT_MIN}" \
VICTIM_GRPC_PORT_MAX="${VICTIM_GRPC_PORT_MAX}" \
LOAD_CLIENT_PID="${LOAD_CLIENT_PID:-exited}" \
MASTER_HEALTH_CODE="${MASTER_HEALTH_CODE}" \
N_PREFILL_TOTAL="${N_PREFILL_TOTAL}" \
N_DECODE_TOTAL="${N_DECODE_TOTAL}" \
FLEXLB_HTTP_PORT_VAL="${FLEXLB_HTTP_PORT}" \
CLUSTER_CONTROL_PORT_VAL="${CLUSTER_CONTROL_PORT}" \
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
cluster_pid = os.environ.get("CLUSTER_PID", "N/A")
cluster_n_prefill = int(os.environ.get("CLUSTER_N_PREFILL", "0"))
cluster_n_decode = int(os.environ.get("CLUSTER_N_DECODE", "0"))
cluster_kill_role_engines = int(os.environ.get("CLUSTER_KILL_ROLE_ENGINES", "0"))
victim_name = os.environ.get("VICTIM_NAME", "N/A")
victim_n_prefill = int(os.environ.get("VICTIM_N_PREFILL", "0"))
victim_n_decode = int(os.environ.get("VICTIM_N_DECODE", "0"))
victim_base_grpc_port = os.environ.get("VICTIM_BASE_GRPC_PORT", "N/A")
victim_port_min = int(os.environ.get("VICTIM_GRPC_PORT_MIN", "0"))
victim_port_max = int(os.environ.get("VICTIM_GRPC_PORT_MAX", "0"))
load_client_pid = os.environ.get("LOAD_CLIENT_PID", "N/A")
master_health_code = os.environ.get("MASTER_HEALTH_CODE", "000")
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
                "grpc_addr": e.get("grpc_addr", "?"),
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

# The victim JVM names its engines from index 0 (e.g. "prefill-0"), which
# collides with cluster engine names.  Identify victim engines by their gRPC
# port range instead of by name.
def victim_port(engine):
    try:
        return int(str(engine.get("grpc_addr", "")).rsplit(":", 1)[1])
    except (ValueError, IndexError):
        return -1

def is_victim(engine):
    return victim_port_min <= victim_port(engine) <= victim_port_max

# -- Recovery verification data --
recovery_summary = load_json("recovery_verify/summary.json")
recovery_total = recovery_summary.get("total_requests", 0) if recovery_summary else 0
recovery_ok = recovery_summary.get("completed", 0) if recovery_summary else 0
recovery_success_rate = (recovery_ok / recovery_total * 100) if recovery_total > 0 else 0

# -- Assertion 1: Master did not crash (HTTP port available during kill) --
master_alive = master_health_code == "200"

# -- Assertion 2: Surviving engines continued accepting requests (when the
#    cluster still hosts engines of the killed role) OR Master gracefully
#    degraded (single-engine topology: no same-role survivor) --
has_same_role_survivor = cluster_kill_role_engines > 0
if has_same_role_survivor:
    # Compare surviving engine accepted counts (same role as killed)
    baseline_accepted = sum(
        e.get("accepted", 0) for e in (baseline_res or {}).get("engines", [])
        if e.get("role") == kill_target and not is_victim(e)
    )
    kill_accepted = sum(
        e.get("accepted", 0) for e in (kill_res or {}).get("engines", [])
        if e.get("role") == kill_target and not is_victim(e)
    )
    surviving_engines_ok = kill_accepted > baseline_accepted
    assertion2_detail = (
        f"surviving {kill_target} engines accepted: "
        f"baseline={baseline_accepted} -> kill={kill_accepted}"
    )
else:
    # Single-engine topology: check Master returned errors (not hang)
    # Master alive + load client still running = graceful degradation
    surviving_engines_ok = master_alive
    assertion2_detail = (
        f"no same-role survivor (single-engine topology): master_alive={master_alive}, "
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
    if has_same_role_survivor:
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
                f"Mock engine {engine.get('name', '?')} ({engine.get('grpc_addr', '?')}) has "
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
w(f"| Victim JVM | {victim_n_prefill}P + {victim_n_decode}D (base grpc {victim_base_grpc_port}, range {victim_port_min}-{victim_port_max}) |")
w(f"| Victim Engine (killed role) | {victim_name} ({kill_target}) |")
w(f"| Mock Cluster PID | {cluster_pid} |")
w(f"| Load Client PID | {load_client_pid} |")
w(f"| Kill Timestamp | {kill_ts} |")
w(f"| Restart Timestamp | {restart_ts} |")
w(f"| Master HTTP (kill period) | {master_health_code} |")
w(f"| FLEXLB_HTTP_PORT | {os.environ.get('FLEXLB_HTTP_PORT_VAL', '18080')} |")
w(f"| Cluster Control HTTP | {os.environ.get('CLUSTER_CONTROL_PORT_VAL', 'N/A')} |")
w(f"| MOCK_BASE_GRPC_PORT | {os.environ.get('MOCK_BASE_GRPC_PORT_VAL', '55151')} |")
w(f"| Schedule Mode | batch (FLEXLB_CONFIG strict JSON, schemaVersion 2) |")
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
    w("  | Engine | Role | gRPC Addr | Running | Accepted | Completed | Cancelled |")
    w("  |---|---|---|---|---|---|---|")
    for e in res["engines"]:
        victim_mark = " (victim)" if is_victim(e) else ""
        w(f"  | {e['name']}{victim_mark} | {e['role']} | {e['grpc_addr']} | {e['running']} | {e['accepted']} | {e['completed']} | {e['cancelled']} |")
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
w(f"| 2 | {'Surviving engines accepted requests' if has_same_role_survivor else 'Master graceful degradation'} | {'PASS' if surviving_engines_ok else 'FAIL'} | {assertion2_detail} |")
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
            cancelled_detail = f"{e['name']} ({e.get('grpc_addr', '?')}): {e['cancelled']}"
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
w(f"- Victim JVM ({victim_n_prefill}P + {victim_n_decode}D, victim {kill_target} engine {victim_name}) was killed at {kill_ts} and restarted at {restart_ts}")
w(f"- Engine mode: {engine_mode}, kill target: {kill_target}, same-role survivors in cluster: {cluster_kill_role_engines}")
w(f"- Master remained alive during kill period: {master_alive} (HTTP {master_health_code})")
if total > 0:
    w(f"- Load client: {ok_count}/{total} succeeded ({(ok_count/total*100) if total > 0 else 0:.1f}%)")
    if error_count > 0:
        w(f"- {error_count} requests failed during kill period (expected — engine unavailable)")
if kill_res:
    surviving = [e for e in kill_res.get("engines", []) if not is_victim(e)]
    w(f"- Surviving engines during kill: {len(surviving)} engines")
if post_restart_res:
    w(f"- Post-restart total running: {post_restart_res['total_running']}")
if post_restart_in:
    w(f"- Post-restart scheduler inflight: {post_restart_in['scheduler_inflight']}")
if recovery_total > 0:
    w(f"- Recovery verification: {recovery_ok}/{recovery_total} succeeded ({recovery_success_rate:.1f}%)")
w("")

# Master flexlb.log events (TTL sweep / fence quarantine evidence).  These
# lines explain WHY post-restart inflight may be non-zero on this branch:
# the scheduler fences stranded requests and waits for an authoritative
# engine terminal, while the mock engine has no Cancel RPC — quarantined
# fences are retained indefinitely (retained=... stays flat across sweeps).
w("### Master Log Events (TTL sweep / fence quarantine)")
w("")
master_log = run_dir / "master_logs" / "flexlb.log"
event_lines = []
if master_log.exists():
    try:
        for line in master_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "event=" in line:
                event_lines.append(line.strip())
    except Exception:
        pass
if event_lines:
    shown = event_lines[-40:]
    w("```")
    for line in shown:
        w(line)
    if len(event_lines) > len(shown):
        w(f"... ({len(event_lines) - len(shown)} earlier event lines omitted)")
    w("```")
else:
    w("- _No event= lines found in master_logs/flexlb.log_")
w("")

w("---")
w(f"_Report generated at {datetime.now().isoformat()}_")

report = "\n".join(lines)
report_path = run_dir / "test_report.md"
report_path.write_text(report, encoding="utf-8")
# Exit-code contract: the orchestrating shell reads this file and propagates
# PASS/FAIL to its own exit status.
(run_dir / "test_passed").write_text("true" if test_passed else "false", encoding="utf-8")
print(report)
PYEOF

# ===========================================================================
# Done — propagate test_passed to the exit code
# ===========================================================================

echo ""
echo "=========================================="
echo "  Test Complete"
echo "=========================================="
echo "  Report:    ${RUN_DIR}/test_report.md"
echo "  Run dir:   ${RUN_DIR}"
echo "  Master stdout: ${RUN_DIR}/flexlb_master.log"
echo "  Master flexlb log: ${MASTER_LOG_DIR}/flexlb.log"
echo "  Mock log:  ${CLUSTER_RUN_DIR}/mock_engine.log"
echo "  Victim log (initial): ${VICTIM_RUN_DIR}/mock_engine.log"
echo "  Victim log (restart): ${VICTIM_RESTART_RUN_DIR}/mock_engine.log"
echo "  Load client log: ${RUN_DIR}/load_client.log"
echo "=========================================="

TEST_RESULT="missing"
if [[ -f "${RUN_DIR}/test_passed" ]]; then
  TEST_RESULT="$(cat "${RUN_DIR}/test_passed")"
fi
if [[ "${TEST_RESULT}" == "true" ]]; then
  echo ""
  echo "RESULT: PASS (all 5 hard assertions held)"
  exit 0
else
  echo ""
  echo "RESULT: FAIL (test_passed=${TEST_RESULT}; see ${RUN_DIR}/test_report.md)"
  exit 1
fi

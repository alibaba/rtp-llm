#!/usr/bin/env bash
# ===========================================================================
# common.sh — Shared functions for FlexLB online_eval test scripts.
#
# This file is sourced by run_smoke.sh, run_perf.sh, and run_resilience.sh.
# It provides:
#   - Path derivation (SCRIPT_DIR, FLEXLB_DIR, REPO_ROOT)
#   - Java 21 detection
#   - Port helpers (wait_for_port, assert_ports_free)
#   - Process cleanup (cleanup_flexlb, cleanup_mock, cleanup_clients, cleanup_all)
#   - Master lifecycle (start_master, stop_master)
#   - Mock cluster lifecycle (start_mock_cluster, stop_mock_cluster)
#   - Build helper (build_flexlb_jar)
#   - Endpoint parsing, health checks, perf config generation
#   - Environment variable summary printer
#
# Design principle: functions read env vars directly from the environment.
# Callers are responsible for setting all configuration before calling.
# No multi-layer overrides (no default->env->cmdline chains).
# ===========================================================================

# -- Constants --------------------------------------------------------------

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

# -- Path derivation --------------------------------------------------------

flexlb_init_paths() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  REPO_ROOT="$(cd "${FLEXLB_DIR}/../.." && pwd)"
}

# -- Preflight: user identity check --

assert_not_root() {
    if [ "$(whoami)" = "root" ]; then
        echo "FATAL: Running as root. Use 'docker exec -u \${SSH_USER:-wuran.wzy}' to enter container." >&2
        exit 1
    fi
}

# -- Java detection ---------------------------------------------------------

java_major() {
  local java_bin="${1:-java}"
  "${java_bin}" -version 2>&1 | awk -F'[\".]' '/version/ {print ($2 == "1" ? $3 : $2); exit}'
}

detect_java21_home() {
  if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    if [[ "$(java_major "${JAVA_HOME}/bin/java")" -ge 21 ]]; then
      echo "${JAVA_HOME}"
      return 0
    fi
  fi
  if [[ -n "${JAVA21_HOME:-}" && -x "${JAVA21_HOME}/bin/java" ]]; then
    echo "${JAVA21_HOME}"
    return 0
  fi
  if [[ -x "${HOME}/java21/bin/java" \
        && "$(java_major "${HOME}/java21/bin/java")" -ge 21 ]]; then
    echo "${HOME}/java21"
    return 0
  fi
  local java_bin
  while IFS= read -r java_bin; do
    if [[ -x "${java_bin}" && "$(java_major "${java_bin}")" -ge 21 ]]; then
      dirname "$(dirname "${java_bin}")"
      return 0
    fi
  done < <(
    {
      alternatives --display java 2>/dev/null || true
      update-alternatives --display java 2>/dev/null || true
    } | awk '/bin\/java/ {print $1}' | sort -u
  )
  return 1
}

setup_java21() {
  local detected
  detected="$(detect_java21_home || true)"
  if [[ -n "${detected}" ]]; then
    export JAVA_HOME="${detected}"
    export PATH="${JAVA_HOME}/bin:${PATH}"
  fi
}

# -- Port helpers -----------------------------------------------------------

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

assert_ports_free() {
  python3 - "$@" <<'PY'
import socket, sys
sockets = []
try:
    for raw_port in sys.argv[1:]:
        port = int(raw_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            print(f"required port {port} is not available: {exc}", file=sys.stderr)
            sys.exit(1)
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
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

assert_no_concurrent_flexlb_test() {
  local matches
  matches="$(pgrep -af 'flexlb_load_client\.py|mock_engine_shard_launcher\.py|flexlb-api-[^ ]*\.jar|flexlb-mock-engine-[^ ]*\.jar' || true)"
  if [[ -n "${matches}" ]]; then
    echo "Concurrent FlexLB performance processes detected on the host:" >&2
    echo "${matches}" >&2
    echo "Wait for them to finish, or set FLEXLB_FAIL_ON_CONCURRENT_TEST=0 to override." >&2
    return 1
  fi
}

# -- Endpoint helpers -------------------------------------------------------

parse_endpoint_env() {
  local endpoint_file="$1"
  FLEXLB_ENV_ARGS=()
  while IFS= read -r line; do
    FLEXLB_ENV_ARGS+=("${line}")
  done < <(python3 - "${endpoint_file}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for key, value in payload["env"].items():
    print(f"{key}={value}")
PY
)
  echo "  parsed ${#FLEXLB_ENV_ARGS[@]} service-discovery env vars"
}

wait_for_endpoints_ready() {
  local master_port=$1
  local expected_prefill=$2
  local expected_decode=$3
  local max_wait="${ENDPOINT_READY_TIMEOUT_S:-120}"
  local elapsed=0
  echo "[wait_for_endpoints_ready] Waiting for ${expected_prefill} prefill + ${expected_decode} decode endpoints..."
  while [ "${elapsed}" -lt "${max_wait}" ]; do
    local response
    response=$(curl -s -X POST "http://127.0.0.1:${master_port}/rtp_llm/master/info" \
        -H "Content-Type: application/json" -H "Accept: application/json" \
        -d '{}' 2>/dev/null) || true
    if [ -n "${response}" ]; then
      local result
      result=$(echo "${response}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    ready = data.get('ready', False)
    ws = data.get('worker_summary', {})
    p = ws.get('PREFILL', {}); d = ws.get('DECODE', {})
    print(f'{ready}|{p.get(\"discovered\",0)}|{p.get(\"alive\",0)}|{d.get(\"discovered\",0)}|{d.get(\"alive\",0)}')
except Exception:
    print('False|0|0|0|0')
" 2>/dev/null) || result="False|0|0|0|0"
      local ready p_disc p_alive d_disc d_alive
      IFS='|' read -r ready p_disc p_alive d_disc d_alive <<< "${result}"
      if [ "${ready}" = "True" ] && [ "${p_alive}" -ge "${expected_prefill}" ] && [ "${d_alive}" -ge "${expected_decode}" ]; then
        echo "[wait_for_endpoints_ready] All endpoints ready (${elapsed}s)"
        return 0
      fi
      echo "[wait_for_endpoints_ready] Not ready: prefill=${p_alive}/${expected_prefill}, decode=${d_alive}/${expected_decode} (${elapsed}s)"
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "[wait_for_endpoints_ready] ERROR: Timeout after ${max_wait}s" >&2
  return 1
}

# -- Health checks ----------------------------------------------------------

save_master_info() {
  local output=$1
  curl -fsS -X POST "http://127.0.0.1:${FLEXLB_HTTP_PORT}/rtp_llm/master/info" \
    -H "Content-Type: application/json" -H "Accept: application/json" \
    -d '{}' >"${output}"
}

save_master_prometheus() {
  local output=$1
  local path
  for path in prometheus actuator/prometheus; do
    if curl -fsS "http://127.0.0.1:${FLEXLB_MANAGEMENT_PORT}/${path}" >"${output}"; then
      return 0
    fi
  done
  rm -f "${output}"
  echo "WARNING: unable to save Master Prometheus snapshot" >&2
  return 1
}

assert_mock_engine_healthy() {
  if [[ "${START_MOCK:-1}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${MOCK_PID:-}" ]] || ! kill -0 "${MOCK_PID}" >/dev/null 2>&1; then
    echo "Mock engine is not running" >&2
    tail -80 "${RUN_DIR}/mock_engine.log" >&2 || true
    return 1
  fi
  if grep -q "OutOfMemoryError" "${RUN_DIR}/mock_engine.log" 2>/dev/null; then
    echo "Mock engine encountered OutOfMemoryError" >&2
    tail -80 "${RUN_DIR}/mock_engine.log" >&2 || true
    return 1
  fi
}

# -- Cleanup functions ------------------------------------------------------
# Split so different scripts can clean up different process sets.

cleanup_flexlb() {
  if [[ -n "${FLEXLB_PID:-}" ]]; then
    kill "${FLEXLB_PID}" >/dev/null 2>&1 || true
    wait "${FLEXLB_PID}" 2>/dev/null || true
    FLEXLB_PID=""
  fi
}

cleanup_mock() {
  if [[ -n "${MOCK_PID:-}" ]]; then
    kill "${MOCK_PID}" >/dev/null 2>&1 || true
    wait "${MOCK_PID}" 2>/dev/null || true
    MOCK_PID=""
  fi
}

cleanup_clients() {
  local pid
  for pid in "${CLIENT_PIDS[@]:-}" "${LOAD_CLIENT_PID:-}" "${MONITOR_PID:-}" "${VICTIM_PID:-}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  for pid in "${CLIENT_PIDS[@]:-}" "${LOAD_CLIENT_PID:-}" "${MONITOR_PID:-}" "${VICTIM_PID:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  done
  CLIENT_PIDS=()
  LOAD_CLIENT_PID=""
  MONITOR_PID=""
  VICTIM_PID=""
}

cleanup_all() {
  echo ""
  echo "[cleanup] stopping all processes ..."
  cleanup_clients
  cleanup_flexlb
  cleanup_mock
  echo "[cleanup] done."
}

# -- Master lifecycle -------------------------------------------------------
# start_master <log_file> <env_var_assignments...>
#
# Env var assignments are passed via `env` to the java process.
# The caller is responsible for building the full env var list.
# Optional arrays:
#   FLEXLB_JVM_OPTS  — extra JVM options (heap, JFR, system props)
#   FLEXLB_SPRING_ARGS — extra Spring Boot args (--logging.level, etc.)

start_master() {
  local log_file="$1"; shift
  echo "  starting master ..."
  env "$@" \
    java "${JAVA_MODULE_OPTS[@]}" \
    ${FLEXLB_JVM_OPTS[@]+"${FLEXLB_JVM_OPTS[@]}"} \
    -jar "${FLEXLB_JAR}" \
    --server.port="${FLEXLB_HTTP_PORT}" \
    --management.server.port="${FLEXLB_MANAGEMENT_PORT}" \
    --spring.profiles.active="${SPRING_PROFILE:-default}" \
    ${FLEXLB_SPRING_ARGS[@]+"${FLEXLB_SPRING_ARGS[@]}"} \
    >"${log_file}" 2>&1 &
  FLEXLB_PID="$!"
  if ! wait_for_port "127.0.0.1" "${FLEXLB_HTTP_PORT}" 60; then
    if ! kill -0 "${FLEXLB_PID}" >/dev/null 2>&1; then
      echo "ERROR: master exited during startup" >&2
      tail -50 "${log_file}" >&2 || true
    fi
    return 1
  fi
  if ! kill -0 "${FLEXLB_PID}" 2>/dev/null; then
    echo "ERROR: master process died during startup" >&2
    tail -50 "${log_file}" >&2 || true
    return 1
  fi
  echo "  master started (pid=${FLEXLB_PID})"
}

stop_master() {
  if [[ -n "${FLEXLB_PID:-}" ]]; then
    echo "  stopping master (pid=${FLEXLB_PID}) ..."
    kill "${FLEXLB_PID}" >/dev/null 2>&1 || true
    wait "${FLEXLB_PID}" 2>/dev/null || true
    FLEXLB_PID=""
    sleep "${STOP_MASTER_DELAY:-2}"
  fi
}

# -- Mock cluster lifecycle -------------------------------------------------

start_mock_cluster() {
  local endpoint_file="$1"
  local env_file="$2"
  local log_file="$3"

  if [[ "${START_MOCK:-1}" != "1" ]]; then
    if [[ ! -f "${endpoint_file}" ]]; then
      echo "START_MOCK=0 requires endpoint file at ${endpoint_file}" >&2
      return 1
    fi
    echo "  [skipped] mock cluster already running (using ${endpoint_file})"
    return 0
  fi

  if [[ "${MOCK_ENGINE_IMPL:-python}" == "java" ]]; then
    mapfile -t JAVA_MOCK_PORTS < <(seq "${MOCK_BASE_GRPC_PORT}" \
      "$((MOCK_BASE_GRPC_PORT + N_PREFILL + N_DECODE - 1))")
    assert_ports_free "${JAVA_MOCK_PORTS[@]}"
    if [[ ! -f "${JAVA_MOCK_ENGINE_JAR}" ]]; then
      echo "Java mock engine jar not found: ${JAVA_MOCK_ENGINE_JAR}" >&2
      return 1
    fi
    java -Xms"${JAVA_MOCK_JVM_XMS:-32g}" -Xmx"${JAVA_MOCK_JVM_XMX:-32g}" \
      -XX:+ExitOnOutOfMemoryError \
      -Xlog:gc*,safepoint:"${RUN_DIR}/mock_engine_gc.log":time,uptime,level,tags:filecount=3,filesize=20m \
      -jar "${JAVA_MOCK_ENGINE_JAR}" \
      --n-prefill "${N_PREFILL}" --n-decode "${N_DECODE}" \
      --base-grpc-port "${MOCK_BASE_GRPC_PORT}" \
      --event-loop-threads "${JAVA_MOCK_EVENT_LOOP_THREADS:-32}" \
      --performance "${PERFORMANCE_FILE}" \
      --master-config "${PROCESS_CONFIG_FILE:-/dev/null}" \
      --prefill-cache-blocks "${PREFILL_CACHE_BLOCKS}" \
      --decode-cache-blocks "${DECODE_CACHE_BLOCKS}" \
      --endpoint-file "${endpoint_file}" --env-file "${env_file}" \
      >"${log_file}" 2>&1 &
    MOCK_PID="$!"
    wait_for_port "127.0.0.1" "$((MOCK_BASE_GRPC_PORT + N_PREFILL + N_DECODE - 1))" 60
  else
    local mock_script="${SCRIPT_DIR}/mock_engine_cluster.py"
    local extra_args=()
    if [[ "${N_SHARDS:-1}" -gt 1 ]]; then
      mock_script="${SCRIPT_DIR}/mock_engine_shard_launcher.py"
      extra_args=(--n-shards "${N_SHARDS}")
    fi
    PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN:-python3}" "${mock_script}" \
      --n-prefill "${N_PREFILL}" --n-decode "${N_DECODE}" \
      --base-grpc-port "${MOCK_BASE_GRPC_PORT}" \
      --performance "${PERFORMANCE_FILE}" \
      --prefill-cache-blocks "${PREFILL_CACHE_BLOCKS}" \
      --decode-cache-blocks "${DECODE_CACHE_BLOCKS}" \
      --endpoint-file "${endpoint_file}" --env-file "${env_file}" \
      "${extra_args[@]}" \
      >"${log_file}" 2>&1 &
    MOCK_PID="$!"
    if [[ "${N_SHARDS:-1}" -gt 1 ]]; then
      wait_for_port "127.0.0.1" "${MOCK_PROXY_PORT}" 180
    else
      wait_for_port "127.0.0.1" "${MOCK_BASE_GRPC_PORT}" 20
    fi
  fi

  if ! kill -0 "${MOCK_PID}" 2>/dev/null; then
    echo "ERROR: mock cluster died during startup" >&2
    tail -50 "${log_file}" >&2 || true
    return 1
  fi
  echo "  mock cluster started (pid=${MOCK_PID})"
}

stop_mock_cluster() {
  cleanup_mock
}

# -- Build helper -----------------------------------------------------------

build_flexlb_jar() {
  if [[ ! -f "${FLEXLB_JAR}" ]]; then
    echo "  Building flexlb-api (mvnw) ..."
    (cd "${FLEXLB_DIR}" && ./mvnw -P"${MAVEN_PROFILES:-opensource,!internal}" \
      -pl flexlb-api -am package -DskipTests)
    if [[ ! -f "${FLEXLB_JAR}" ]]; then
      echo "ERROR: JAR build failed: ${FLEXLB_JAR}" >&2
      return 1
    fi
  fi
  echo "  JAR: ${FLEXLB_JAR}"
}

# -- Perf config generation -------------------------------------------------

generate_perf_config() {
  local output="$1"
  local prefill_ms="${2:-100.0}"
  local decode_step_ms="${3:-20.0}"
  cat > "${output}" <<JSON
{
  "block_size": 1024,
  "sleep_scale": 1.0,
  "prefill": { "fixed_ms": ${prefill_ms}, "scale": 1.0 },
  "decode": {
    "scale": 1.0,
    "step_ms_by_batch": [
      [1, ${decode_step_ms}], [2, ${decode_step_ms}], [4, ${decode_step_ms}], [8, ${decode_step_ms}],
      [16, ${decode_step_ms}], [32, ${decode_step_ms}], [64, ${decode_step_ms}], [128, ${decode_step_ms}], [256, ${decode_step_ms}]
    ]
  }
}
JSON
  echo "  perf_config=${output}"
}

# -- Python detection -------------------------------------------------------

detect_python() {
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${HOME}/.venvs/flexlb-eval/bin/python3" ]]; then
      PYTHON_BIN="${HOME}/.venvs/flexlb-eval/bin/python3"
    else
      PYTHON_BIN="$(command -v python3 || true)"
    fi
  fi
  if [[ -z "${PYTHON_BIN}" ]] \
      || ! "${PYTHON_BIN}" -c 'import aiohttp, grpc' >/dev/null 2>&1; then
    echo "Python with aiohttp and grpc is required; set PYTHON_BIN to the eval venv" >&2
    return 1
  fi
  export PYTHON_BIN
}

# -- Environment variable summary -------------------------------------------
# Prints effective values of specified env vars for debugging.
# Usage: print_env_summary VAR1 VAR2 VAR3 ...

print_env_summary() {
  echo "============================================"
  echo "  Effective Configuration"
  echo "============================================"
  local var
  for var in "$@"; do
    local value="${!var:-}"
    if [[ -z "${value}" ]]; then
      printf "  %-40s = (unset)\n" "${var}"
    else
      printf "  %-40s = %s\n" "${var}" "${value}"
    fi
  done
  echo "============================================"
  echo ""
}

# -- Trace filtering --------------------------------------------------------

filter_trace_by_ol() {
  local input="$1"
  local output="$2"
  local max_ol="$3"
  python3 - "${input}" "${output}" "${max_ol}" <<'PY'
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
        except Exception:
            pass
print(f"  filtered: {count} requests (ol <= {max_ol})")
PY
}

# -- Network isolation (for perf tests) -------------------------------------

flexlb_maybe_enter_namespace() {
  if [[ "${FLEXLB_NETWORK_ISOLATED:-0}" == "1" \
        && "${FLEXLB_NETWORK_NAMESPACE_ACTIVE:-0}" != "1" ]]; then
    exec unshare -Urn bash -c \
      'ip link set lo up; export FLEXLB_NETWORK_NAMESPACE_ACTIVE=1; exec "$@"' \
      bash bash "$0" "$@"
  fi
}

# -- Remote execution helper (for scripts running on local Mac) --
# Automatically injects -u to prevent root container access.
# Usage: remote_exec "command inside container"
# Required env vars: SSH_USER, REMOTE_IP, CONTAINER
remote_exec() {
    if [[ -z "${SSH_USER:-}" || -z "${REMOTE_IP:-}" || -z "${CONTAINER:-}" ]]; then
        echo "FATAL: remote_exec requires SSH_USER, REMOTE_IP, CONTAINER env vars" >&2
        return 1
    fi
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        "${SSH_USER}@${REMOTE_IP}" \
        "docker exec --user ${SSH_USER} ${CONTAINER} bash -lc '$*'"
}

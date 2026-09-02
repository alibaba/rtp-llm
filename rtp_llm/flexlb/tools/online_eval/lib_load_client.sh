#!/usr/bin/env bash
# ===========================================================================
# lib_load_client.sh — shared helpers for the Java mock engine cluster jar
# and the JavaLoadClient.
#
# Sourced by orchestration scripts (run_online_eval.sh) so that the
# JavaLoadClient env-var mapping lives in exactly one place. It is also the
# common contract for the smoke / fault suites: start_java_mock_cluster,
# wait_mock_cluster_ready, mock_http and stop_java_mock_cluster (bottom of
# this file) drive the single-JVM JavaMockEngineCluster lifecycle.
#
# ---- Divergence from the feat/flexlb_mock_engine_v2 baseline (intentional) ----
#
# This lib keeps this branch's HEAD semantics instead of the v2 defaults:
#   * Load client JVM sizing: no -Xms by default, -Xmx defaults to 16g
#     (v2: -Xms4g -Xmx4g), plus -XX:+ExitOnOutOfMemoryError.
#   * PRIORITY is part of JAVA_LOAD_CLIENT_ENV_VARS and therefore blanked
#     unless passed explicitly (v2: PRIORITY not in the blank list, so an
#     ambient PRIORITY can leak into the JVM).
# If these scripts are ever merged back onto the v2 baseline, the divergence
# above must be re-reviewed in the MR — it is deliberate, not drift.
#
# Requires the sourcing script to define FLEXLB_DIR (the flexlb Maven root,
# i.e. rtp_llm/flexlb).
# ===========================================================================

MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"
JAVA_MOCK_ENGINE_JAR="${JAVA_MOCK_ENGINE_JAR:-${FLEXLB_DIR}/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar}"
JAVA_LOAD_CLIENT_MAIN_CLASS="org.flexlb.mockengine.JavaLoadClient"
# The load client ships inside the same fat jar as the mock engine cluster
# (JAVA_LOAD_CLIENT_JAR defaults to JAVA_MOCK_ENGINE_JAR, so callers that
# only set one of the two keep working).
JAVA_LOAD_CLIENT_JAR="${JAVA_LOAD_CLIENT_JAR:-${JAVA_MOCK_ENGINE_JAR}}"
# Load client JVM sizing, mirroring run_online_eval.sh's historical knobs:
# no -Xms by default, -Xmx defaults to 16g (JAVA_LOAD_CLIENT_HEAP_SIZE's
# old default). Override via JAVA_LOAD_CLIENT_JVM_XMS /
# JAVA_LOAD_CLIENT_JVM_XMX before sourcing this lib.
JAVA_LOAD_CLIENT_JVM_XMS="${JAVA_LOAD_CLIENT_JVM_XMS:-}"
JAVA_LOAD_CLIENT_JVM_XMX="${JAVA_LOAD_CLIENT_JVM_XMX:-16g}"

# ---- JDK 21 detection (extracted from run_online_eval.sh) ----
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

# Ensure a JDK >= 21 is active for subsequent java invocations (mock engine
# jar, load client). Fails hard instead of letting a default JDK 17 blow up
# at runtime.
require_java21() {
  local home
  home="$(detect_java21_home || true)"
  if [[ -z "${home}" ]]; then
    echo "ERROR: Java 21+ is required to run the FlexLB mock engine / load client." >&2
    echo "Set JAVA21_HOME or JAVA_HOME to a JDK 21 installation." >&2
    exit 1
  fi
  export JAVA_HOME="${home}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
}

# Build flexlb-mock-engine (which bundles JavaMockEngineCluster,
# MockControlServer and JavaLoadClient) when the fat jar is missing.
ensure_java_mock_engine_jar() {
  require_java21
  if [[ ! -f "${JAVA_MOCK_ENGINE_JAR}" ]]; then
    echo "Java mock engine jar not found, building: ${JAVA_MOCK_ENGINE_JAR}" >&2
    (cd "${FLEXLB_DIR}" && ./mvnw -P"${MAVEN_PROFILES}" -pl flexlb-mock-engine -am package -DskipTests) >&2
    if [[ ! -f "${JAVA_MOCK_ENGINE_JAR}" ]]; then
      echo "Failed to build Java mock engine jar: ${JAVA_MOCK_ENGINE_JAR}" >&2
      return 1
    fi
  fi
  return 0
}

# Every env var read by JavaLoadClient.Config.fromEnv(). Listed here so the
# mapping cannot drift between scripts. PRIORITY is part of the surface
# (env-level default priority; JavaLoadClient's built-in default is 50, the
# neutral QoS level — priority 0 is rejected by master admission — and an
# explicit PRIORITY=0 leaves the field unset on the wire; per-record trace
# priority overrides both), so it is blanked here too — callers that want
# an env-level default pass "PRIORITY=<n>" explicitly. FORCE_PRIORITY
# (single-QoS pin that overrides both the trace field and the PRIORITY
# default) is blanked by the same rule: callers pass "FORCE_PRIORITY=<n>"
# explicitly.
JAVA_LOAD_CLIENT_ENV_VARS=(
  TRACE_FILE
  TARGET_ADDR
  GRPC_TARGET
  DURATION_S
  MAX_CONCURRENCY
  REPLAY_SPEED
  LOAD_CLIENT_WORKERS
  OUTPUT_DIR
  NUM_SHARDS
  SHARD_INDEX
  LIMIT
  TIMEOUT_MS
  SLA_TTFT_MS
  FETCH_OUTPUT_STREAM
  LOOP
  REPLAY_UNIQUE_PREFIX
  N_CHANNELS
  EVENT_LOOP_THREADS
  START_AT_EPOCH_MS
  RESPONSE_TIMEOUT
  SKIP_SERVER_LATENCY
  MODEL
  API_KEY
  GRADIENT
  GRADIENT_START_SPEED
  GRADIENT_MAX_SPEED
  MAX_INPUT_LEN
  MAX_OUTPUT_LEN
  PUSHGATEWAY_URL
  ENABLE_FALLBACK
  ENDPOINTS_FILE
  DRY_RUN
  PRIORITY
  FORCE_PRIORITY
  SEND_MODE
  SEND_MODE_QPS
  RAMP_UP_SECONDS
)

# run_java_load_client VAR=value [VAR=value ...]
#
# Starts one JavaLoadClient instance via exec (so the caller's background
# pid is the JVM itself and kill/STOP/CONT behave as with the Python
# client). Every JavaLoadClient env var is exported explicitly: variables
# given as arguments take the given value, all others are exported empty —
# JavaLoadClient treats empty env as "unset" and falls back to its built-in
# default, so no ambient environment can leak in.
#
# Jar handling: this function only checks JAVA_LOAD_CLIENT_JAR (the jar on
# the -cp classpath) — it does NOT auto-build. Auto-building the mock
# engine jar here would fire one Maven build per shard when callers launch
# N load clients with a custom JAVA_LOAD_CLIENT_JAR. Building is the
# sourcing script's job (run_online_eval.sh auto-builds once before any
# shard is launched; ensure_java_mock_engine_jar and
# start_java_mock_cluster below remain available for scripts that want
# build-on-demand behavior).
#
# Usage:
#   run_java_load_client \
#     "TRACE_FILE=${TRACE_FILE}" \
#     "TARGET_ADDR=127.0.0.1:${FLEXLB_HTTP_PORT}" \
#     ... \
#     >"${RUN_DIR}/load_client.log" 2>&1 &
run_java_load_client() {
  require_java21
  if [[ ! -f "${JAVA_LOAD_CLIENT_JAR}" ]]; then
    echo "ERROR: Java load client jar not found: ${JAVA_LOAD_CLIENT_JAR}" >&2
    echo "Build it with: (cd \"${FLEXLB_DIR}\" && ./mvnw -P\"${MAVEN_PROFILES}\" -pl flexlb-mock-engine -am package -DskipTests)" >&2
    echo "or set JAVA_LOAD_CLIENT_JAR to an existing jar path." >&2
    return 1
  fi
  local var kv
  for var in "${JAVA_LOAD_CLIENT_ENV_VARS[@]}"; do
    export "${var}="
  done
  for kv in "$@"; do
    export "${kv?run_java_load_client: arguments must be VAR=value pairs}"
  done
  local java_opts=(-XX:+ExitOnOutOfMemoryError)
  if [[ -n "${JAVA_LOAD_CLIENT_JVM_XMS}" ]]; then
    java_opts+=(-Xms"${JAVA_LOAD_CLIENT_JVM_XMS}")
  fi
  if [[ -n "${JAVA_LOAD_CLIENT_JVM_XMX}" ]]; then
    java_opts+=(-Xmx"${JAVA_LOAD_CLIENT_JVM_XMX}")
  fi
  exec java "${java_opts[@]}" \
    -cp "${JAVA_LOAD_CLIENT_JAR}" "${JAVA_LOAD_CLIENT_MAIN_CLASS}"
}

# ===========================================================================
# JavaMockEngineCluster lifecycle helpers (single-JVM mock cluster)
# ===========================================================================
# Common contract for the smoke / fault suites: bring a Java mock cluster up
# with start_java_mock_cluster, wait for it with wait_mock_cluster_ready,
# drive fault injection / control through mock_http, and tear it down with
# stop_java_mock_cluster. Port layout (see JavaMockEngineCluster.main): the
# engines listen on MOCK_BASE_GRPC_PORT .. base+n_prefill+n_decode-1 and the
# HTTP control server on base-1. Startup order inside the JVM is: all gRPC
# ports bound -> discovery files written -> control server started, so a
# reachable control port implies the discovery files exist.

# start_java_mock_cluster <run_dir>
#
# Launches one JavaMockEngineCluster JVM in the background and records its
# pid in <run_dir>/mock_engine.pid. Cluster tuning comes from the
# environment (all optional unless marked REQUIRED; defaults mirror the
# Java CLI built-ins):
#
#   MOCK_N_PREFILL / MOCK_N_DECODE       engine counts (default 2 / 4)
#   MOCK_BASE_GRPC_PORT                  first gRPC port (default 61000;
#                                        control HTTP port = base - 1)
#   MOCK_PERFORMANCE_FILE                --performance (REQUIRED)
#   MOCK_MASTER_CONFIG                   --master-config (REQUIRED)
#   MOCK_PREFILL_CACHE_BLOCKS            default 6000
#   MOCK_DECODE_CACHE_BLOCKS             default 3000
#   MOCK_EVENT_LOOP_THREADS              default 32
#   MOCK_COMPLETION_THREADS              default 8
#   MOCK_JVM_XMS / MOCK_JVM_XMX          JVM heap (default 4g / 4g)
#   MOCK_STATS_INTERVAL_MS / MOCK_DECODE_MAX_CONCURRENCY /
#   MOCK_TOTAL_KV_TOKENS / MOCK_BLOCK_SIZE / MOCK_HOST /
#   MOCK_PREFILL_DOMAIN / MOCK_DECODE_DOMAIN
#                                        optional: passed through only
#                                        when set
#
# Side effects: the JVM writes <run_dir>/endpoints.json and
# <run_dir>/flexlb_env.txt (service discovery); stdout/stderr and GC logs go
# to <run_dir>/mock_engine.log and mock_engine_gc.log. On success exports
# MOCK_ENDPOINT_FILE / MOCK_ENV_FILE / MOCK_CONTROL_PORT for the sibling
# helpers. The jar is auto-built once when missing (a cluster starts once
# per test, never per shard, so a Maven build here cannot fan out). Callers
# that need port exclusivity should run their own assert_ports_free first.
start_java_mock_cluster() {
  local run_dir="$1"
  local n_prefill="${MOCK_N_PREFILL:-2}"
  local n_decode="${MOCK_N_DECODE:-4}"
  local base_port="${MOCK_BASE_GRPC_PORT:-61000}"
  local perf_file="${MOCK_PERFORMANCE_FILE:-}"
  local master_config="${MOCK_MASTER_CONFIG:-}"
  if [[ -z "${perf_file}" || ! -f "${perf_file}" ]]; then
    echo "start_java_mock_cluster: MOCK_PERFORMANCE_FILE is required and must exist (got: '${perf_file}')" >&2
    return 1
  fi
  if [[ -z "${master_config}" || ! -f "${master_config}" ]]; then
    echo "start_java_mock_cluster: MOCK_MASTER_CONFIG is required and must exist (got: '${master_config}')" >&2
    return 1
  fi
  ensure_java_mock_engine_jar || return 1

  mkdir -p "${run_dir}"
  MOCK_ENDPOINT_FILE="${MOCK_ENDPOINT_FILE:-${run_dir}/endpoints.json}"
  MOCK_ENV_FILE="${MOCK_ENV_FILE:-${run_dir}/flexlb_env.txt}"
  export MOCK_ENDPOINT_FILE MOCK_ENV_FILE
  export MOCK_CONTROL_PORT="$((base_port - 1))"

  local cluster_args=(
    --n-prefill "${n_prefill}"
    --n-decode "${n_decode}"
    --base-grpc-port "${base_port}"
    --event-loop-threads "${MOCK_EVENT_LOOP_THREADS:-32}"
    --completion-threads "${MOCK_COMPLETION_THREADS:-8}"
    --prefill-cache-blocks "${MOCK_PREFILL_CACHE_BLOCKS:-6000}"
    --decode-cache-blocks "${MOCK_DECODE_CACHE_BLOCKS:-3000}"
    --performance "${perf_file}"
    --master-config "${master_config}"
    --endpoint-file "${MOCK_ENDPOINT_FILE}"
    --env-file "${MOCK_ENV_FILE}"
  )
  if [[ -n "${MOCK_STATS_INTERVAL_MS:-}" ]]; then
    cluster_args+=(--stats-interval-ms "${MOCK_STATS_INTERVAL_MS}")
  fi
  if [[ -n "${MOCK_DECODE_MAX_CONCURRENCY:-}" ]]; then
    cluster_args+=(--decode-max-concurrency "${MOCK_DECODE_MAX_CONCURRENCY}")
  fi
  if [[ -n "${MOCK_TOTAL_KV_TOKENS:-}" ]]; then
    cluster_args+=(--total-kv-tokens "${MOCK_TOTAL_KV_TOKENS}")
  fi
  if [[ -n "${MOCK_BLOCK_SIZE:-}" ]]; then
    cluster_args+=(--block-size "${MOCK_BLOCK_SIZE}")
  fi
  if [[ -n "${MOCK_HOST:-}" ]]; then
    cluster_args+=(--host "${MOCK_HOST}")
  fi
  if [[ -n "${MOCK_PREFILL_DOMAIN:-}" ]]; then
    cluster_args+=(--prefill-domain "${MOCK_PREFILL_DOMAIN}")
  fi
  if [[ -n "${MOCK_DECODE_DOMAIN:-}" ]]; then
    cluster_args+=(--decode-domain "${MOCK_DECODE_DOMAIN}")
  fi

  java -Xms"${MOCK_JVM_XMS:-4g}" -Xmx"${MOCK_JVM_XMX:-4g}" \
    -XX:+ExitOnOutOfMemoryError \
    -Xlog:gc*,safepoint:"${run_dir}/mock_engine_gc.log":time,uptime,level,tags:filecount=3,filesize=20m \
    -jar "${JAVA_MOCK_ENGINE_JAR}" "${cluster_args[@]}" \
    >"${run_dir}/mock_engine.log" 2>&1 &
  local mock_pid="$!"
  echo "${mock_pid}" >"${run_dir}/mock_engine.pid"
  echo "[start_java_mock_cluster] pid=${mock_pid} prefill=${n_prefill} decode=${n_decode} grpc=${base_port}-$((base_port + n_prefill + n_decode - 1)) control=${MOCK_CONTROL_PORT} run_dir=${run_dir}" >&2
}

# wait_mock_cluster_ready <base_port> <n_engines> [timeout_s]
#
# Blocks until a cluster started by start_java_mock_cluster is usable:
# the last gRPC engine port (base_port + n_engines - 1) accepts connections,
# the HTTP control server (base_port - 1) accepts connections, and — when
# MOCK_ENDPOINT_FILE is exported (start_java_mock_cluster does that) — the
# discovery file exists and is non-empty. n_engines = n_prefill + n_decode.
# timeout_s defaults to 60. Prints progress on stderr; returns non-zero on
# timeout.
wait_mock_cluster_ready() {
  local base_port="$1"
  local n_engines="$2"
  local timeout_s="${3:-60}"
  python3 - "${base_port}" "$((base_port + n_engines - 1))" \
    "$((base_port - 1))" "${timeout_s}" "${MOCK_ENDPOINT_FILE:-}" <<'PY'
import os
import socket
import sys
import time

base_port, last_grpc, control, timeout_s, endpoint_file = (
    int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
    float(sys.argv[4]), sys.argv[5])

def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False

def endpoint_ready():
    if not endpoint_file:
        return True
    try:
        return os.path.getsize(endpoint_file) > 0
    except OSError:
        return False

deadline = time.time() + timeout_s
while True:
    if port_open(last_grpc) and port_open(control) and endpoint_ready():
        print(f"[wait_mock_cluster_ready] cluster ready: "
              f"grpc={base_port}-{last_grpc} control={control}"
              + (f" endpoints={endpoint_file}" if endpoint_file else ""),
              file=sys.stderr)
        sys.exit(0)
    if time.time() >= deadline:
        print(f"[wait_mock_cluster_ready] ERROR: timeout after {timeout_s:.0f}s "
              f"(grpc {last_grpc} open={port_open(last_grpc)}, control {control} "
              f"open={port_open(control)})", file=sys.stderr)
        sys.exit(1)
    time.sleep(0.5)
PY
}

# stop_java_mock_cluster <run_dir>
#
# Gracefully stops the cluster started by start_java_mock_cluster: SIGTERM
# first (the JVM runs a shutdown hook that drains in-flight requests), then
# SIGKILL after MOCK_STOP_GRACE_SECONDS (default 10) if it is still alive.
# Idempotent: a missing or stale pid file is a no-op. The pid file is
# always removed on exit.
stop_java_mock_cluster() {
  local run_dir="$1"
  local pid_file="${run_dir}/mock_engine.pid"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    return 0
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    local grace="${MOCK_STOP_GRACE_SECONDS:-10}"
    local deadline=$((SECONDS + grace))
    while kill -0 "${pid}" 2>/dev/null; do
      if (( SECONDS >= deadline )); then
        echo "[stop_java_mock_cluster] pid ${pid} still alive after ${grace}s, sending SIGKILL" >&2
        kill -9 "${pid}" 2>/dev/null || true
        break
      fi
      sleep 0.5
    done
    # Reap the child when this shell owns it; silent no-op for foreign pids.
    wait "${pid}" 2>/dev/null || true
    echo "[stop_java_mock_cluster] stopped pid ${pid}" >&2
  else
    echo "[stop_java_mock_cluster] pid ${pid} already gone" >&2
  fi
  rm -f "${pid_file}"
}

# mock_http <method> <port> <path> [json_body]
#
# Fires one request at the mock engine's HTTP control server and echoes the
# response body on stdout. <port> is the control port (base gRPC port - 1,
# exported as MOCK_CONTROL_PORT by start_java_mock_cluster). GET endpoints:
# /health /snapshot /metrics /requests; POST endpoints: /inject
# /clear_inject /set_perf /set_kv_pressure /set_queue_depth /stop_engine
# /start_engine /cancel_request. Examples:
#   mock_http GET  62099 /health
#   mock_http GET  62099 /snapshot
#   mock_http POST 62099 /inject \
#     '{"engine":"decode-1","type":"generate_delay","enabled":true,"delay_ms":500}'
# Exit status is curl's: connection failures return non-zero; HTTP 4xx/5xx
# still print the error body and return 0 — inspect the body (or pipe
# through jq) when the status matters.
mock_http() {
  local method="$1"
  local port="$2"
  local path="$3"
  local body="${4:-}"
  if [[ -n "${body}" ]]; then
    curl -s -X "${method}" -H "Content-Type: application/json" \
      -d "${body}" "http://127.0.0.1:${port}${path}"
  else
    curl -s -X "${method}" "http://127.0.0.1:${port}${path}"
  fi
}

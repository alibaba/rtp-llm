#!/usr/bin/env bash
# ===========================================================================
# lib_load_client.sh — shared helpers for the Java mock engine cluster jar
# and the JavaLoadClient.
#
# Sourced by orchestration scripts (run_online_eval.sh) so that the
# JavaLoadClient env-var mapping lives in exactly one place. On the
# feat/flexlb_mock_engine_v2 baseline this lib is also sourced by
# run_cancel_smoke.sh, run_matrix_smoke.sh and the *_test.sh fault suites;
# on this branch those scripts run the Python mock engine / Python load
# client and therefore do not source it.
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
# (env-level default priority, 0 = unset; per-record trace priority
# overrides it), so it is blanked here too — callers that want an env-level
# default pass "PRIORITY=<n>" explicitly.
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
  ZERO_OUTPUT_POLICY
  SCHEDULE_ONLY
  LOOP
  N_CHANNELS
  EVENT_LOOP_THREADS
  START_AT_EPOCH_MS
  RESPONSE_TIMEOUT
  SKIP_SERVER_LATENCY
  MODEL
  API_KEY
  FLEXLB_EXPECT_FETCH_RESPONSE
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
  SEND_MODE
  SEND_MODE_QPS
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
# sourcing script's job (run_online_eval.sh's LOAD_CLIENT_IMPL=java branch
# auto-builds once before any shard is launched; ensure_java_mock_engine_jar
# remains available for scripts that want the old behavior).
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

#!/usr/bin/env bash
# ===========================================================================
# lib_load_client.sh — shared helpers for the Java mock engine cluster jar
# and the JavaLoadClient.
#
# Sourced by orchestration scripts (run_cancel_smoke.sh, run_matrix_smoke.sh,
# master_recovery_ttft_test.sh, master_kill_restart_test.sh, ...) so that the
# JavaLoadClient env-var mapping lives in exactly one place, mirroring the
# mapping table in run_online_eval.sh (Phase 3).
#
# Requires the sourcing script to define FLEXLB_DIR (the flexlb Maven root,
# i.e. rtp_llm/flexlb).
# ===========================================================================

MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"
JAVA_MOCK_ENGINE_JAR="${JAVA_MOCK_ENGINE_JAR:-${FLEXLB_DIR}/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar}"
JAVA_LOAD_CLIENT_MAIN_CLASS="org.flexlb.mockengine.JavaLoadClient"
# Load client JVM sizing (same as run_online_eval.sh Phase 3).
JAVA_LOAD_CLIENT_JVM_XMS="${JAVA_LOAD_CLIENT_JVM_XMS:-4g}"
JAVA_LOAD_CLIENT_JVM_XMX="${JAVA_LOAD_CLIENT_JVM_XMX:-4g}"

# Build flexlb-mock-engine (which bundles JavaMockEngineCluster,
# MockControlServer and JavaLoadClient) when the fat jar is missing.
ensure_java_mock_engine_jar() {
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
# mapping cannot drift between scripts.
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
# Usage:
#   run_java_load_client \
#     "TRACE_FILE=${TRACE_FILE}" \
#     "TARGET_ADDR=127.0.0.1:${FLEXLB_HTTP_PORT}" \
#     ... \
#     >"${RUN_DIR}/load_client.log" 2>&1 &
run_java_load_client() {
  ensure_java_mock_engine_jar || return 1
  local var kv
  for var in "${JAVA_LOAD_CLIENT_ENV_VARS[@]}"; do
    export "${var}="
  done
  for kv in "$@"; do
    export "${kv?run_java_load_client: arguments must be VAR=value pairs}"
  done
  exec java -Xms"${JAVA_LOAD_CLIENT_JVM_XMS}" -Xmx"${JAVA_LOAD_CLIENT_JVM_XMX}" \
    -cp "${JAVA_MOCK_ENGINE_JAR}" "${JAVA_LOAD_CLIENT_MAIN_CLASS}"
}

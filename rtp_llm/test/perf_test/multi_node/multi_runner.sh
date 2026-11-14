#!/bin/bash
set -x;
SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

remote_environment() {
  while IFS= read -r -d '' assignment; do
    printf '%q ' "$assignment"
  done < <(env -0)
}

multi_build_script() {
  # Split ip list by comma
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  # Check essential environment variables
  if [ -z "$IP_ARRAY" ] || [ -z "$RUN_USER" ] || [ -z "$SSH_PORT" ]; then
    echo "SSH parameters are not set"
    exit 1
  fi
  if [ -z "$OPEN_SOURCE_REF" ]; then
    echo "OPEN_SOURCE_REF should be set"
    exit 1
  fi
  # Export essential environment variables
  export BAZEL_BUILD_ARGS=${BAZEL_BUILD_ARGS:-" --jobs 64 --verbose_failures --config=cuda13 "}
  export BUILD_FROM_SCRATCH=${BUILD_FROM_SCRATCH:-2}
  # Run build script on each ip with environment variables
  (
    trap 'kill 0' SIGINT;
    for IP in "${IP_ARRAY[@]}"
    do
      (
        scp -P ${SSH_PORT} "${SCRIPT_DIRECTORY}/multi_local_executor.sh" ${RUN_USER}@${IP}:/tmp/multi_local_executor.sh;
        # Concat all environment variables
        ENV_STR=$(remote_environment);
        echo "Environment variables: $ENV_STR";
        ssh "${RUN_USER}@${IP}" -p "${SSH_PORT}" "env $ENV_STR bash /tmp/multi_local_executor.sh";
      ) &
    done;
    wait;
  )
}

multi_kill_script() {
  # Split ip list by comma
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  # Check essential environment variables
  if [ -z "$IP_ARRAY" ] || [ -z "$RUN_USER" ] || [ -z "$SSH_PORT" ]; then
    echo "SSH parameters are not set"
    exit 1
  fi
  # Run kill script on each ip with environment variables
  (
    trap 'kill 0' SIGINT;
    for IP in "${IP_ARRAY[@]}"
    do
      (
        scp -P ${SSH_PORT} "${SCRIPT_DIRECTORY}/multi_local_executor.sh" ${RUN_USER}@${IP}:/tmp/multi_local_executor.sh;
        # Concat all environment variables
        ENV_STR=$(remote_environment);
        echo "Environment variables: $ENV_STR";
        ssh "${RUN_USER}@${IP}" -p "${SSH_PORT}" "env $ENV_STR bash /tmp/multi_local_executor.sh";
      ) &
    done;
    wait;
  )
}

multi_copy_script() {
  # Split ip list by comma
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  # Check essential environment variables
  if [ -z "$IP_ARRAY" ] || [ -z "$RUN_USER" ] || [ -z "$SSH_PORT" ]; then
    echo "SSH parameters are not set"
    exit 1
  fi
  if [ -z "$TASK_OUTPUT_DIR" ]; then
    echo "TASK_OUTPUT_DIR should be set"
    exit 1
  fi
  # Copy test output result to local
  (
    trap 'kill 0' SIGINT;
    export WORLD_RANK=0;
    pids=()
    for IP in "${IP_ARRAY[@]}"
    do
      (
        # Concat all environment variables
        ENV_STR=$(remote_environment)
        TEST_OUTPUT_PATH=$(ssh "${RUN_USER}@${IP}" -p "${SSH_PORT}" "env $ENV_STR bash /tmp/multi_local_executor.sh") || exit 1
        echo "TEST_OUTPUT_PATH=${TEST_OUTPUT_PATH}";
        TEST_OUTPUT_NAME=$(basename "$TEST_OUTPUT_PATH")
        if [ "${DIAGNOSTICS_ONLY:-0}" -eq 1 ]; then
          mkdir -p "${TASK_OUTPUT_DIR}"
          scp -r -P "${SSH_PORT}" "${RUN_USER}@${IP}:${TEST_OUTPUT_PATH}" "${TASK_OUTPUT_DIR}/${TEST_OUTPUT_NAME}" || exit 1
          exit 0
        fi
        if [ "${COLLECT_DIAGNOSTICS:-1}" -eq 1 ]; then
          scp -P "${SSH_PORT}" "${RUN_USER}@${IP}:${TEST_OUTPUT_PATH}/main_logs/process.log" "${TASK_OUTPUT_DIR}/process_logs/process_${TEST_OUTPUT_NAME}.log" || exit 1
          printf -v quoted_output '%q' "$TEST_OUTPUT_PATH"
          traces_present=$(ssh "${RUN_USER}@${IP}" -p "${SSH_PORT}" "find $quoted_output -maxdepth 1 -name 'normal_*' -printf x -quit") || exit 1
          if [ -n "$traces_present" ]; then
            scp -r -P "${SSH_PORT}" "${RUN_USER}@${IP}:${TEST_OUTPUT_PATH}/normal_*" "${TASK_OUTPUT_DIR}/trace_files/" || exit 1
          fi
        fi
        if [ $WORLD_RANK -eq 0 ]; then
          mkdir -p "${TASK_OUTPUT_DIR}/run"
          for artifact in results reports profiles manifest.json execution_plan.json resolved_config.json source_config.json; do
            scp -r -P "${SSH_PORT}" "${RUN_USER}@${IP}:${TEST_OUTPUT_PATH}/run/${artifact}" "${TASK_OUTPUT_DIR}/run/" || exit 1
          done
          printf -v quoted_run '%q' "${TEST_OUTPUT_PATH}/run"
          retained_traces=$(ssh "${RUN_USER}@${IP}" -p "${SSH_PORT}" "find $quoted_run -maxdepth 1 -type d -name traces -printf x -quit") || exit 1
          if [ -n "$retained_traces" ]; then
            scp -r -P "${SSH_PORT}" "${RUN_USER}@${IP}:${TEST_OUTPUT_PATH}/run/traces" "${TASK_OUTPUT_DIR}/run/" || exit 1
          fi
        else
          printf -v quoted_output '%q' "$TEST_OUTPUT_PATH"
          decode_traces=$(ssh "${RUN_USER}@${IP}" -p "${SSH_PORT}" "find $quoted_output -maxdepth 1 -type f -name 'decode_*.json' -printf x -quit") || exit 1
          if [ -n "$decode_traces" ]; then
            shard_index=$((WORLD_RANK / ${LOCAL_WORLD_SIZE:-8} + 1))
            mkdir -p "${TASK_OUTPUT_DIR}/run/traces/shard_${shard_index}"
            scp -P "${SSH_PORT}" "${RUN_USER}@${IP}:${TEST_OUTPUT_PATH}/decode_*.json" "${TASK_OUTPUT_DIR}/run/traces/shard_${shard_index}/" || exit 1
          fi
        fi
      ) &
      pids+=("$!")
      export WORLD_RANK=$((WORLD_RANK + ${LOCAL_WORLD_SIZE:-8}));
    done;
    status=0
    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
    exit "$status"
  )
}

multi_clean_script() {
  # Split ip list by comma
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  # Check essential environment variables
  if [ -z "$IP_ARRAY" ] || [ -z "$RUN_USER" ] || [ -z "$SSH_PORT" ]; then
    echo "SSH parameters are not set"
    exit 1
  fi
  # Run clean script on each ip with environment variables
  (
    trap 'kill 0' SIGINT;
    for IP in "${IP_ARRAY[@]}"
    do
      (
        scp -P ${SSH_PORT} "${SCRIPT_DIRECTORY}/multi_local_executor.sh" ${RUN_USER}@${IP}:/tmp/multi_local_executor.sh;
        # Concat all environment variables
        ENV_STR=$(remote_environment);
        echo "Environment variables: $ENV_STR";
        ssh "${RUN_USER}@${IP}" -p "${SSH_PORT}" "env $ENV_STR bash /tmp/multi_local_executor.sh";
      ) &
    done;
    wait;
  )
}

multi_test_script() {
  if [ -z "$PERF_TEST_CONFIG_PATH" ] || [ ! -r "$PERF_TEST_CONFIG_PATH" ]; then
    echo "PERF_TEST_CONFIG_PATH must name a readable perf config"
    exit 1
  fi
  local config_path="$PERF_TEST_CONFIG_PATH"
  local run_id="${PERF_TEST_RUN_ID:-${START_PORT:-12333}}"
  if [[ ! "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Invalid PERF_TEST_RUN_ID"
    exit 1
  fi
  local remote_config_dir="/tmp/rtp_llm_perf_${run_id}"
  # Split ip list by comma
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  # Calculate parallel parameters
  NUM_WORKERS=${#IP_ARRAY[@]}
  export EP_SIZE=${EP_SIZE:-$((TP_SIZE * DP_SIZE))}
  export WORLD_SIZE=$((TP_SIZE * DP_SIZE * ${PP_SIZE:-1}))
  export LOCAL_WORLD_SIZE=$((WORLD_SIZE / NUM_WORKERS))
  if ((WORLD_SIZE % NUM_WORKERS != 0)); then
    echo "WORLD_SIZE should be multiple of NUM_WORKERS"
    exit 1
  fi
  # Generate GANG_CONFIG_STRING: join ip list with server ports
  GANG_CONFIG_STRING=""
  WORKER_RANK=0
  for IP in "${IP_ARRAY[@]}"
  do
    GANG_CONFIG_STRING="${GANG_CONFIG_STRING}name:test_part${WORKER_RANK},ip:${IP},port:${START_PORT};"
    WORKER_RANK=$((WORKER_RANK + 1))
  done
  # Remove last ; from GANG_CONFIG_STRING
  export GANG_CONFIG_STRING="${GANG_CONFIG_STRING::-1}";
  # Check essential environment variables
  if [ -z "$IP_ARRAY" ] || [ -z "$RUN_USER" ] || [ -z "$SSH_PORT" ]; then
    echo "SSH parameters are not set"
    exit 1
  fi
  if [ -z "$OPEN_SOURCE_REF" ]; then
    echo "OPEN_SOURCE_REF should be set"
    exit 1
  fi
  if [ -z "$TP_SIZE" ] || [ -z "$DP_SIZE" ] || [ -z "$EP_SIZE" ] || [ -z "$WORLD_SIZE" ] || [ -z "$LOCAL_WORLD_SIZE" ] || [ -z "$GANG_CONFIG_STRING" ]; then
    echo "Parallel parameters are not set"
    exit 1
  fi
  if [ -z "$MODEL_TYPE" ] || [ -z "$TOKENIZER_PATH" ] || [ -z "$CHECKPOINT_PATH" ]; then
    echo "Model parameters are not set"
    exit 1
  fi
  # Export essential environment variables
  export BAZEL_BUILD_ARGS=${BAZEL_BUILD_ARGS:-" --jobs 64 --verbose_failures --config=cuda13 "}
  export FRONTEND_SERVER_COUNT=${FRONTEND_SERVER_COUNT:-16}
  export START_PORT=${START_PORT:-12333}
  export WARM_UP=${WARM_UP:-1}
  export RESERVER_RUNTIME_MEM_MB=${RESERVER_RUNTIME_MEM_MB:-0}
  export LOG_LEVEL=${LOG_LEVEL:-INFO}
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
  export NVSHMEM_DEBUG=${NVSHMEM_DEBUG:-INFO}
  export GEN_TIMELINE_SYNC=${GEN_TIMELINE_SYNC:-1}
  # Run test script on each ip with environment variables
  (
    trap 'kill 0' SIGINT;
    export WORLD_RANK=0;
    pids=()
    for IP in "${IP_ARRAY[@]}"
    do
      (
        ssh "${RUN_USER}@${IP}" -p "${SSH_PORT}" "mkdir -p '${remote_config_dir}'" || exit 1
        scp -P "${SSH_PORT}" "$config_path" "${RUN_USER}@${IP}:${remote_config_dir}/config.json" || exit 1
        if [ -n "$PERF_TEST_DISTRIBUTION_PATH" ]; then
          scp -P "${SSH_PORT}" "$PERF_TEST_DISTRIBUTION_PATH" "${RUN_USER}@${IP}:${remote_config_dir}/distribution.json" || exit 1
        fi
        if [ -n "$PERF_TEST_RESOLVED_WORKLOAD_PATH" ]; then
          scp -P "${SSH_PORT}" "$PERF_TEST_RESOLVED_WORKLOAD_PATH" "${RUN_USER}@${IP}:${remote_config_dir}/resolved_workload.json" || exit 1
          export PERF_TEST_RESOLVED_WORKLOAD_PATH="${remote_config_dir}/resolved_workload.json"
        fi
        export PERF_TEST_CONFIG_PATH="${remote_config_dir}/config.json"
        scp -P "${SSH_PORT}" "${SCRIPT_DIRECTORY}/multi_local_executor.sh" "${RUN_USER}@${IP}:/tmp/multi_local_executor.sh" || exit 1
        # Concat all environment variables
        ENV_STR=$(remote_environment);
        echo "Environment variables: $ENV_STR";
        ssh "${RUN_USER}@${IP}" -p "${SSH_PORT}" "env $ENV_STR bash /tmp/multi_local_executor.sh" || exit 1
      ) &
      pids+=("$!")
      export WORLD_RANK=$((WORLD_RANK + LOCAL_WORLD_SIZE));
    done;
    status=0
    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
    exit "$status"
  )
}

main() {
  SUB_CMD=${SUB_CMD}

  case ${SUB_CMD} in
      build)
          echo "execute multi_build_script"
          multi_build_script
          ;;
      kill)
          echo "execute multi_kill_script"
          multi_kill_script
          ;;
      copy)
          echo "execute multi_copy_script"
          multi_copy_script
          ;;
      clean)
          echo "execute multi_clean_script"
          multi_clean_script
          ;;
      test)
          echo "execute multi_test_script"
          multi_test_script
          ;;
  esac
}

main;

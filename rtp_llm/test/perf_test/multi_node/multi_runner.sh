#!/bin/bash
set -x;

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
  export BAZEL_BUILD_ARGS=${BAZEL_BUILD_ARGS:-" --jobs 64 --verbose_failures --config=cuda12_6 "}
  export BUILD_FROM_SCRATCH=${BUILD_FROM_SCRATCH:-2}
  # Run build script on each ip with environment variables
  (
    trap 'kill 0' SIGINT;
    for IP in "${IP_ARRAY[@]}"
    do
      (
        scp -P ${SSH_PORT} multi_local_executor.sh ${RUN_USER}@${IP}:/tmp/multi_local_executor.sh;
        # Concat all environment variables
        ENV_STR=$(printenv | paste -sd " ");
        echo "Environment variables: $ENV_STR";
        ssh ${RUN_USER}@${IP} -p ${SSH_PORT} "$ENV_STR bash /tmp/multi_local_executor.sh";
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
        scp -P ${SSH_PORT} multi_local_executor.sh ${RUN_USER}@${IP}:/tmp/multi_local_executor.sh;
        # Concat all environment variables
        ENV_STR=$(printenv | paste -sd " ");
        echo "Environment variables: $ENV_STR";
        ssh ${RUN_USER}@${IP} -p ${SSH_PORT} "$ENV_STR bash /tmp/multi_local_executor.sh";
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
    for IP in "${IP_ARRAY[@]}"
    do
      (
        # Concat all environment variables
        ENV_STR=$(printenv | paste -sd " ");
        echo "Environment variables: $ENV_STR";
        TEST_OUTPUT_PATH=$(ssh ${RUN_USER}@${IP} -p ${SSH_PORT} "$ENV_STR bash /tmp/multi_local_executor.sh");
        echo "TEST_OUTPUT_PATH=${TEST_OUTPUT_PATH}";
        TEST_OUTPUT_NAME=$(basename "$TEST_OUTPUT_PATH")
        scp -P ${SSH_PORT} ${RUN_USER}@${IP}:${TEST_OUTPUT_PATH}/main_logs/process.log ${TASK_OUTPUT_DIR}/process_logs/process_${TEST_OUTPUT_NAME}.log;
        scp -P ${SSH_PORT} ${RUN_USER}@${IP}:${TEST_OUTPUT_PATH}/normal_* ${TASK_OUTPUT_DIR}/trace_files/;
        if [ $WORLD_RANK -eq 0 ]; then
          scp -P ${SSH_PORT} ${RUN_USER}@${IP}:${TEST_OUTPUT_PATH}/*Result.json ${TASK_OUTPUT_DIR}/;
        fi
      ) &
      export WORLD_RANK=$((WORLD_RANK + 8));
    done;
    wait;
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
        scp -P ${SSH_PORT} multi_local_executor.sh ${RUN_USER}@${IP}:/tmp/multi_local_executor.sh;
        # Concat all environment variables
        ENV_STR=$(printenv | paste -sd " ");
        echo "Environment variables: $ENV_STR";
        ssh ${RUN_USER}@${IP} -p ${SSH_PORT} "$ENV_STR bash /tmp/multi_local_executor.sh";
      ) &
    done;
    wait;
  )
}

multi_test_script() {
  # Split ip list by comma
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  # Calculate parallel parameters
  NUM_WORKERS=${#IP_ARRAY[@]}
  export EP_SIZE=${EP_SIZE:-$((TP_SIZE * DP_SIZE))}
  export WORLD_SIZE=$EP_SIZE
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
  export GANG_CONFIG_STRING="\"${GANG_CONFIG_STRING::-1}\"";
  # Check essential environment variables
  if [ -z "$IP_ARRAY" ] || [ -z "$RUN_USER" ] || [ -z "$SSH_PORT" ]; then
    echo "SSH parameters are not set"
    exit 1
  fi
  if [ -z "$OPEN_SOURCE_REF" ]; then
    echo "OPEN_SOURCE_REF should be set"
    exit 1
  fi
  if [ -z "$TP_SIZE"] || [ -z "$DP_SIZE" ] || [ -z "$EP_SIZE" ] || [ -z "$WORLD_SIZE" ] || [ -z "$LOCAL_WORLD_SIZE" ] || [ -z "$GANG_CONFIG_STRING" ]; then
    echo "Parallel parameters are not set"
    exit 1
  fi
  if [ -z "$MODEL_TYPE"] || [ -z "$TOKENIZER_PATH" ] || [ -z "$CHECKPOINT_PATH" ]; then
    echo "Model parameters are not set"
    exit 1
  fi
  # Export essential environment variables
  export BAZEL_BUILD_ARGS=${BAZEL_BUILD_ARGS:-" --jobs 64 --verbose_failures --config=cuda12_6 "}
  export FRONTEND_SERVER_COUNT=${FRONTEND_SERVER_COUNT:-16}
  export LOAD_CKPT_NUM_PROCESS=${LOAD_CKPT_NUM_PROCESS:-64}
  export START_PORT=${START_PORT:-12333}
  export WARM_UP=${WARM_UP:-1}
  export RESERVER_RUNTIME_MEM_MB=${RESERVER_RUNTIME_MEM_MB:-0}
  export DEVICE_RESERVE_MEMORY_BYTES=${DEVICE_RESERVE_MEMORY_BYTES:-0}
  export LOG_LEVEL=${LOG_LEVEL:-INFO}
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
  export NVSHMEM_DEBUG=${NVSHMEM_DEBUG:-INFO}
  export GEN_TIMELINE_SYNC=${GEN_TIMELINE_SYNC:-1}
  # Run test script on each ip with environment variables
  (
    trap 'kill 0' SIGINT;
    export WORLD_RANK=0;
    for IP in "${IP_ARRAY[@]}"
    do
      (
        scp -P ${SSH_PORT} multi_local_executor.sh ${RUN_USER}@${IP}:/tmp/multi_local_executor.sh;
        # Concat all environment variables
        ENV_STR=$(printenv | paste -sd " ");
        echo "Environment variables: $ENV_STR";
        ssh ${RUN_USER}@${IP} -p ${SSH_PORT} "$ENV_STR bash /tmp/multi_local_executor.sh";
      ) &
      export WORLD_RANK=$((WORLD_RANK + LOCAL_WORLD_SIZE));
    done;
    wait;
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

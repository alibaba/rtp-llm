#!/bin/bash
set -x;

multi_build_script() {
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  (
    trap 'kill 0' SIGINT;
    for IP in "${IP_ARRAY[@]}"
    do
      (
        scp -P ${SSH_PORT} multi_local_executor.sh $RUN_USER@$IP:/tmp/multi_local_executor.sh && \
        ssh $RUN_USER@$IP -p ${SSH_PORT} \
        "FT_SUB_DIR=${FT_SUB_DIR} \
        GIT_CHECKOUT_REF=${GIT_CHECKOUT_REF} \
        BAZEL_BUILD_ARGS=${BAZEL_BUILD_ARGS} \
        BUILD_FROM_SCRATCH=${BUILD_FROM_SCRATCH} \
        SUB_CMD=${SUB_CMD} \
        bash /tmp/multi_local_executor.sh"
      ) &
    done;
    wait;
  )
}

multi_kill_script() {
  pgrep -f 'multi_benchmark' | xargs kill;
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  (
    trap 'kill 0' SIGINT;
    for IP in "${IP_ARRAY[@]}"
    do
      (
        scp -P ${SSH_PORT} multi_local_executor.sh $RUN_USER@$IP:/tmp/multi_local_executor.sh && \
        ssh $RUN_USER@$IP -p ${SSH_PORT} \
        "SUB_CMD=${SUB_CMD} \
        bash /tmp/multi_local_executor.sh"
      ) &
    done;
    wait;
    echo "[$(date)] All nodes processed successfully"
  )
}

multi_clean_script() {
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  (
    trap 'kill 0' SIGINT;
    for IP in "${IP_ARRAY[@]}"
    do
      (
        scp -P ${SSH_PORT} multi_local_executor.sh $RUN_USER@$IP:/tmp/multi_local_executor.sh && \
        ssh $RUN_USER@$IP -p ${SSH_PORT} \
        "FT_SUB_DIR=${FT_SUB_DIR} \
        RM_PATTERN_PATH=${MODEL_TYPE^^}_* \
        SUB_CMD=${SUB_CMD} \
        bash /tmp/multi_local_executor.sh"
      ) &
    done;
    wait;
  )
}

multi_test_script() {
  IFS=',' read -r -a IP_ARRAY <<< "$IP_LISTS"
  NUM_WORKERS=${#IP_ARRAY[@]}
  EP_SIZE=${EP_SIZE:-$((TP_SIZE * DP_SIZE))}
  WORLD_SIZE=$EP_SIZE
  LOCAL_WORLD_SIZE=$((WORLD_SIZE / NUM_WORKERS))
  if ((WORLD_SIZE % NUM_WORKERS != 0)); then
    echo "WORLD_SIZE should be multiple of NUM_WORKERS"
    exit 1
  fi

  # generate GANG_CONFIG_STRING: join ip list with server ports
  GANG_CONFIG_STRING=""
  worker_rank=0
  for IP in "${IP_ARRAY[@]}"
  do
    GANG_CONFIG_STRING="${GANG_CONFIG_STRING}name:test_part${worker_rank},ip:${IP},port:${START_PORT};"
    worker_rank=$((worker_rank + 1))
  done
  # remove last ; from GANG_CONFIG_STRING
  GANG_CONFIG_STRING=${GANG_CONFIG_STRING::-1}

  # split ip list by comma and run command on each ip with list
  (
    trap 'kill 0' SIGINT;
    WORLD_RANK=0
    for IP in "${IP_ARRAY[@]}"
    do
      (
        scp -P ${SSH_PORT} multi_local_executor.sh $RUN_USER@$IP:/tmp/multi_local_executor.sh && \
        ssh $RUN_USER@$IP -p ${SSH_PORT} \
        "FT_WORK_DIR=${FT_WORK_DIR} \
        FT_SUB_DIR=${FT_SUB_DIR} \
        DIST_IP_LIST=${IP_LISTS} \
        TP_SIZE=${TP_SIZE} DP_SIZE=${DP_SIZE} EP_SIZE=${EP_SIZE} \
        WORLD_SIZE=${WORLD_SIZE} LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE} \
        WORLD_RANK=${WORLD_RANK} GANG_CONFIG_STRING='${GANG_CONFIG_STRING}' \
        FRONTEND_SERVER_COUNT=${FRONTEND_SERVER_COUNT:-16} \
        TOKENIZER_PATH=${TOKENIZER_PATH} CHECKPOINT_PATH=${CHECKPOINT_PATH} \
        MODEL_TYPE=${MODEL_TYPE} START_PORT=${START_PORT} MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192} \
        CUDA_ASAN=${CUDA_ASAN:-0} \
        NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-8} \
        WARM_UP=${WARM_UP:-1} RESERVER_RUNTIME_MEM_MB=${RESERVER_RUNTIME_MEM_MB:-128} \
        DEVICE_RESERVE_MEMORY_BYTES=${DEVICE_RESERVE_MEMORY_BYTES:-1024000000} \
        HOST_RESERVE_MEMORY_BYTES=${HOST_RESERVE_MEMORY_BYTES:-4096000000} \
        MAX_CONTEXT_BATCH_SIZE=${MAX_CONTEXT_BATCH_SIZE:-1} \
        LOAD_CKPT_NUM_PROCESS=${LOAD_CKPT_NUM_PROCESS:-64} \
        ENABLE_LAYER_MICRO_BATCH=${ENABLE_LAYER_MICRO_BATCH:-1} \
        ENABLE_COMM_OVERLAP=${ENABLE_COMM_OVERLAP:-1} \
        NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
        USE_DEEPEP_MOE=${USE_DEEPEP_MOE:-0} \
        USE_DEEPEP_LOW_LATENCY=${USE_DEEPEP_LOW_LATENCY:-0} \
        USE_DEEPEP_INTERNODE=${USE_DEEPEP_INTERNODE:-0} \
        ACT_TYPE=${ACT_TYPE:-bf16} \
        WEIGHT_TYPE=${WEIGHT_TYPE:-FP16} \
        HACK_LAYER_NUM=${HACK_LAYER_NUM:-0} TEST_LAYER_NUM=${TEST_LAYER_NUM:-}\
        DISABLE_FLASH_INFER=${DISABLE_FLASH_INFER:-0} \
        SEQ_SIZE_PER_BLOCK=${SEQ_SIZE_PER_BLOCK:-64} \
        LOG_LEVEL=${LOG_LEVEL:-INFO} \
        NVSHMEM_DEBUG=${NVSHMEM_DEBUG:-INFO} \
        NVSHMEM_IB_TRAFFIC_CLASS=${NVSHMEM_IB_TRAFFIC_CLASS:-} \
        NCCL_DEBUG=${NCCL_DEBUG:-INFO} \
        CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0} \
        DEEP_EP_NUM_SM=${DEEP_EP_NUM_SM:-0} \
        DEEP_GEMM_NUM_SM=${DEEP_GEMM_NUM_SM:-} \
        REDUNDANT_EXPERT=${REDUNDANT_EXPERT:-0} \
        ACCL_DISPATCH_NUM_WARP_GROUPS=${ACCL_DISPATCH_NUM_WARP_GROUPS:-4} \
        ACCL_COMBINE_NUM_WARP_GROUPS=${ACCL_COMBINE_NUM_WARP_GROUPS:-4} \
        ACCL_SOFT_TX_DEPTH=${ACCL_SOFT_TX_DEPTH:-} \
        ACCL_MAX_USER_MR_GB=${ACCL_MAX_USER_MR_GB:-} \
        HACK_EP_SINGLE_ENTRY=${HACK_EP_SINGLE_ENTRY:-0} \
        DISPATCH_NUM_WARP_GROUPS=${DISPATCH_NUM_WARP_GROUPS:-4} \
        COMBINE_NUM_WARP_GROUPS=${COMBINE_NUM_WARP_GROUPS:-4} \
        ENABLE_MERGE_W13=${ENABLE_MERGE_W13:-0} \
        CONCURRENCY_LIMIT=${CONCURRENCY_LIMIT:-32} \
        ACCL_LOW_LATENCY_OPTIMIZE=${ACCL_LOW_LATENCY_OPTIMIZE:-1} \
        BUILD_FROM_SCRATCH=${BUILD_FROM_SCRATCH} \
        IS_DECODE=${IS_DECODE:-1} \
        BATCH_SIZE_LIST=${BATCH_SIZE_LIST} \
        INPUT_LEN_LIST=${INPUT_LEN_LIST} \
        DECODE_TEST_LENGTH=${DECODE_TEST_LENGTH} \
        GIT_CHECKOUT_REF=${GIT_CHECKOUT_REF} \
        GIT_REPO_URL=${GIT_REPO_URL} \
        BAZEL_BUILD_ARGS=${BAZEL_BUILD_ARGS} \
        SUB_CMD=${SUB_CMD} \
        bash /tmp/multi_local_executor.sh"
      ) &
      WORLD_RANK=$((WORLD_RANK + LOCAL_WORLD_SIZE))
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

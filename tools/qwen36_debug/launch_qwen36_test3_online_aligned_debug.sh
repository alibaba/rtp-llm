#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"

BENCH_DIR="${BENCH_DIR:-/home/zhenyun.yzy/ai-search-bench}"
SRC_ROOT="${SRC_ROOT:-/home/zhenyun.yzy/.config/superpowers/worktrees/github-opensource/qwen36-cuda-graph-checksum-fix-20260709}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/zhenyun.yzy/hf/Qwen3.6-27B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${CHECKPOINT_PATH}}"
RUN_DIR="${RUN_DIR:-${BENCH_DIR}/output/qwen36_debug_service_$(date +%Y%m%d_%H%M%S)}"

START_PORT="${START_PORT:-26400}"
TP_DEVICES="${TP_DEVICES:-4,5}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda310/bin/python}"

mkdir -p "${RUN_DIR}"

emit() {
  local key="$1"
  local default_value="$2"
  local value="${!key:-${default_value}}"
  printf '%s=%q\n' "${key}" "${value}"
}

emit_if_set() {
  local key="$1"
  if [[ -n "${!key:-}" ]]; then
    printf '%s=%q\n' "${key}" "${!key}"
  fi
}

write_env() {
  local env_file="${RUN_DIR}/launch_env.main.txt"
  {
    printf 'PYTHONPATH=%q\n' "${SRC_ROOT}:${PYTHONPATH:-}"
    emit MAGA_SERVER_WORK_DIR "${SRC_ROOT}"
    emit TEST_UNDECLARED_OUTPUTS_DIR "${RUN_DIR}/main_test_outputs"
    emit LOG_PATH "${RUN_DIR}/main_rtp_logs"
    emit HIP_VISIBLE_DEVICES "${TP_DEVICES}"
    emit CUDA_VISIBLE_DEVICES "${TP_DEVICES}"

    emit ACCL_MAX_USER_MR_GB "2000"
    emit ACCL_RX_DEPTH "32"
    emit ACCL_SOFT_TX_DEPTH "8192"
    emit ACCL_TCP_TIMEOUT_MS "800"
    emit ACCL_TX_DEPTH "512"
    emit ACT_TYPE "bf16"
    emit AITER_ASM_DIR "/opt/conda310/lib/python3.10/site-packages/aiter_meta/hsa/"
    emit AITER_JIT_DIR "/home/zhenyun.yzy/.cache/rtp-llm/aiter_jit_qwen36_27b_test3_online"
    emit AUX_STRING "ea119_MI308X_2TP"
    emit BIZ_NAME "Qwen35_Agentic_SFT_AI_Summary_test3"
    emit CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS "800"
    emit CACHE_STORE_RDMA_MODE "1"
    emit CHECKPOINT_PATH "${CHECKPOINT_PATH}"
    emit COMBINE_NUM_WARP_GROUPS "4"
    emit CONCURRENCY_LIMIT "128"
    emit DECODE_RETRY_TIMEOUT_MS "120"
    emit DECODE_RETRY_TIMES "100"
    emit DEPLOYMENT_NAME "ea119_MI308X_2TP"
    emit DEVICE_RESERVE_MEMORY_BYTES "-32212254720"
    emit DISABLE_FLASH_INFER "1"
    emit DISPATCH_NUM_WARP_GROUPS "4"
    emit DP_SIZE "1"
    emit ENABLE_COMM_OVERLAP "0"
    emit ENABLE_CUDA_GRAPH "1"
    emit ENABLE_FMHA "ON"
    emit ENABLE_LAYER_MICRO_BATCH "0"
    emit ENABLE_OPENSOURCE_FMHA "ON"
    emit ENABLE_PAGED_TRT_FMHA "ON"
    emit ENABLE_TRTV1_FMHA "ON"
    emit ENABLE_TRT_FMHA "ON"
    emit EPLB_FORCE_REPACK "1"
    emit EPLB_MODE "NONE"
    emit EPLB_STATS_WINDOW_SIZE "10"
    emit FAKE_BALANCE_EXPERT "0"
    emit FAKE_GANG_ENV "1"
    emit FT_DISABLE_CUSTOM_AR "1"
    emit GEN_NUM_PER_CIRCLE "5"
    emit GEN_TIMELINE_SYNC "0"
    emit GPU_MODEL_DETAIL "AMD-Instinct-MI308X-OAM"
    emit INT8_MODE "0"
    emit KERNEL_SEQ_SIZE_PER_BLOCK "16"
    emit LD_LIBRARY_PATH "/opt/rh/gcc-toolset-12/root/usr/lib64:/opt/conda310/lib/:/opt/rocm/lib:/lib64:/usr/lib64:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"
    emit LOAD_CACHE_TIMEOUT_MS "10000"
    emit LOAD_PYTHON_MODEL "1"
    emit LOCAL_WORLD_SIZE "2"
    emit LORA_INFO "{}"
    emit MAINSE_BERT_MODULE "0"
    emit MAX_BATCH_SIZE "1024"
    emit MAX_CONTEXT_BATCH_SIZE "64"
    emit MAX_RPC_TIMEOUT_MS "1800000"
    emit MAX_SEQ_LEN "40960"
    emit MM_CACHE_ITEM_NUM "10"
    emit MODEL_TYPE "qwen35_dense"
    emit NCCL_DISABLE_ABORT "1"
    emit NCCL_IB_TC "136"
    emit NORMALIZATION_METHOD "0"
    emit NVSHMEM_IB_TRAFFIC_CLASS "136"
    emit PD_SEP_ENABLE_FALLBACK "0"
    emit PREFILL_MAX_WAIT_TIMEOUT_US "180000000"
    emit PREFILL_RETRY_TIMEOUT_MS "20"
    emit PREFILL_RETRY_TIMES "1"
    emit PY_INFERENCE_LOG_RESPONSE "1"
    emit QUANTIZATION "FP8_PER_CHANNEL_COMPRESSED"
    emit RANK_FACTOR "0"
    emit RDMA_CONNECT_RETRY_TIMES "2"
    emit REMOTE_JIT_DIR "dfs://na175dfssearch8--cn-zhangjiakou/whale/jit_cache/"
    emit RESERVER_RUNTIME_MEM_MB "51200"
    emit REUSE_CACHE "1"
    emit ROCM_DISABLE_CUSTOM_AG "1"
    emit ROLE_TYPE "PDFUSION"
    emit SEQ_SIZE_PER_BLOCK "64"
    emit SP_ACT_TYPE "AUTO"
    emit SP_INT8_MODE "0"
    emit SP_MAX_TOKEN_MATCH "2"
    emit SP_MIN_TOKEN_MATCH "2"
    emit SP_WEIGHTS_TYPE "FP16"
    emit SP_WEIGHT_TYPE "FP16"
    emit START_PORT "${START_PORT}"
    emit STOP_WORDS_LIST "[]"
    emit STOP_WORDS_STR "[]"
    emit THINK_MODE "0"
    emit TIMEOUT_KEEP_ALIVE "5"
    emit TOKENIZER_PATH "${TOKENIZER_PATH}"
    emit TP_SIZE "2"
    emit USE_AITER_PA "1"
    emit USE_ASM_PA "0"
    emit USE_GANG "TRUE"
    emit USE_LOCAL "1"
    emit USE_MLA_OPS "0"
    emit USE_SWIZZLEA "1"
    emit VIT_CACHE_LEN "10"
    emit VIT_CONCURRENCY "16"
    emit VIT_TRT "0"
    emit WARM_UP "0"
    emit WARM_UP_WITH_LOSS "0"
    emit WEIGHTS_TYPE "FP16"
    emit WEIGHT_TYPE "FP16"
    emit WORKER_INFO_PORT_NUM "10"
    emit WORLD_RANK "0"
    emit WORLD_SIZE "2"
    emit ZONE_NAME "inference"

    for optional_env in \
      RTPLLM_CUDA_GRAPH_BATCH_DEBUG \
      RTPLLM_DECODE_CHECKSUM_DEBUG \
      RTPLLM_DECODE_CHECKSUM_DIR \
      RTPLLM_DECODE_CHECKSUM_FILE \
      RTPLLM_DECODE_CHECKSUM_EVERY \
      RTPLLM_DECODE_CHECKSUM_MAX_RECORDS \
      RTPLLM_DECODE_CHECKSUM_MAX_LANES \
      RTPLLM_DECODE_CHECKSUM_TRACE_FILTER \
      RTPLLM_DECODE_CHECKSUM_SYNC_DEVICE \
      RTPLLM_DECODE_TOKEN_TRACE \
      RTPLLM_DECODE_TOKEN_TRACE_DIR \
      RTPLLM_DECODE_TOKEN_TRACE_FILE \
      RTPLLM_DECODE_TOKEN_TRACE_FILTER \
      RTPLLM_DECODE_TOKEN_TRACE_CAPTURE_PEERS \
      RTPLLM_DECODE_TOKEN_TRACE_MAX_BLOCKS_PER_GROUP \
      RTPLLM_DECODE_BAD_WATCH \
      RTPLLM_DECODE_BAD_WATCH_DIR \
      RTPLLM_DECODE_BAD_WATCH_FILE \
      RTPLLM_DECODE_BAD_WATCH_TAIL_SIZE \
      RTPLLM_DECODE_BAD_WATCH_MIN_CF \
      RTPLLM_SAMPLER_TRACE \
      RTPLLM_SAMPLER_TRACE_DIR \
      RTPLLM_SAMPLER_TRACE_FILE \
      RTPLLM_SAMPLER_TRACE_FILTER \
      RTPLLM_SAMPLER_TRACE_TOPK \
      RTPLLM_QWEN3_NEXT_TRACE_DEBUG \
      RTPLLM_QWEN3_NEXT_TRACE_DIR \
      RTPLLM_QWEN3_NEXT_TRACE_FILE \
      RTPLLM_QWEN3_NEXT_TRACE_FILTER \
      RTPLLM_QWEN3_NEXT_TRACE_LAYERS \
      RTPLLM_QWEN3_NEXT_TRACE_EVERY \
      RTPLLM_QWEN3_NEXT_TRACE_MAX_RECORDS \
      RTPLLM_QWEN3_NEXT_TRACE_MAX_LANES \
      RTPLLM_QWEN3_NEXT_TRACE_MAX_ELEMS \
      RTPLLM_QWEN3_NEXT_TRACE_TENSOR_MODE \
      RTPLLM_QWEN3_NEXT_TRACE_SYNC_DEVICE; do
      emit_if_set "${optional_env}"
    done
  } > "${env_file}"
}

start_main() {
  write_env
  local env_file="${RUN_DIR}/launch_env.main.txt"
  local log_file="${RUN_DIR}/main.stdout_stderr.log"
  local pid_file="${RUN_DIR}/main.pid"
  mkdir -p "${RUN_DIR}/main_test_outputs" "${RUN_DIR}/main_rtp_logs"

  (
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
    cd "${SRC_ROOT}"
    exec "${PYTHON_BIN}" -m rtp_llm.start_server
  ) > "${log_file}" 2>&1 &
  echo $! > "${pid_file}"
}

kill_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
    kill_tree "${child}"
  done
  kill "${pid}" 2>/dev/null || true
}

stop_run() {
  if [[ -f "${RUN_DIR}/main.pid" ]]; then
    local pid
    pid="$(cat "${RUN_DIR}/main.pid")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill_tree "${pid}"
      sleep 3
      if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    fi
  fi
}

wait_health() {
  local deadline=$((SECONDS + ${HEALTH_TIMEOUT_SECONDS:-1800}))
  while (( SECONDS < deadline )); do
    if curl -fsS --connect-timeout 2 --max-time 5 "http://127.0.0.1:${START_PORT}/health" >/dev/null 2>&1; then
      echo "health ok: http://127.0.0.1:${START_PORT}"
      return 0
    fi
    if [[ -f "${RUN_DIR}/main.pid" ]] && ! kill -0 "$(cat "${RUN_DIR}/main.pid")" 2>/dev/null; then
      echo "main launcher process exited before health" >&2
      tail -200 "${RUN_DIR}/main.stdout_stderr.log" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "health timeout after ${HEALTH_TIMEOUT_SECONDS:-1800}s" >&2
  tail -200 "${RUN_DIR}/main.stdout_stderr.log" >&2 || true
  return 1
}

case "${ACTION}" in
  start)
    echo "${RUN_DIR}" > "${BENCH_DIR}/output/.latest_qwen36_debug_service_run_dir"
    start_main
    echo "run_dir=${RUN_DIR}"
    echo "start_port=${START_PORT}"
    echo "tp_devices=${TP_DEVICES}"
    echo "enable_cuda_graph=${ENABLE_CUDA_GRAPH:-1}"
    wait_health
    ;;
  stop)
    stop_run
    ;;
  *)
    echo "usage: $0 {start|stop}" >&2
    exit 2
    ;;
esac

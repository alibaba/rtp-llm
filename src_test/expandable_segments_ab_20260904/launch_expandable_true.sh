#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?usage: launch_role.sh decode|prefill}"
RUN_DIR="$(cd "$(dirname "$0")" && pwd -P)"
MODEL_DIR="/tmp/models/DeepSeek-V4-Flash-0731"
BASE_WORK="${BASE_WORK_OVERRIDE:-/tmp/rtp_llm_prefill_unified_scr_expandable_true_20260904}"

case "$ROLE" in
  decode)
    VISIBLE_DEVICES="0,1"
    START_PORT_VALUE=18530
    REMOTE_PORT_VALUE=18630
    ROLE_ARGS=(
      --load_method fastsafetensors
      --max_seq_len 8192
      --enable_cuda_graph 1
      --decode_capture_config "1,2,4,8,16"
      --act_type BF16
      --tp_size 1
      --dp_size 2
      --ep_size 2
      --world_size 2
      --seq_size_per_block 256
      --kernel_seq_size_per_block 128
      --role_type DECODE
      --cache_store_rdma_mode 0
      --use_local 1
      --reuse_cache 1
      --enable_device_cache 1
      --enable_memory_cache 1
      --memory_cache_size_mb 1024
      --use_deepep_moe 1
      --use_deepep_low_latency 1
      --load_cache_timeout_ms 900000
      --reserver_runtime_mem_mb 10240
      --fp8_kv_cache 1
      --sp_type dspark
      --gen_num_per_cycle 3
      --sp_model_type deepseek_v4_dspark
      --sp_checkpoint_path "$MODEL_DIR"
      --sp_act_type bf16
      --cp_rotate_method PREFILL_CP
      --think_mode 1
      --enable_fp32_lm_head 0
      --frontend_server_count 1
      --start_port "$START_PORT_VALUE"
      --warm_up 1
      --model_warm_up 1
      --grammar_backend none
    )
    ;;
  prefill)
    VISIBLE_DEVICES="2,3"
    START_PORT_VALUE=18630
    REMOTE_PORT_VALUE=18530
    ROLE_ARGS=(
      --load_method fastsafetensors
      --max_seq_len 8192
      --enable_cuda_graph 0
      --act_type BF16
      --tp_size 2
      --dp_size 1
      --ep_size 2
      --world_size 2
      --seq_size_per_block 256
      --kernel_seq_size_per_block 128
      --role_type PREFILL
      --cache_store_rdma_mode 0
      --use_local 1
      --reuse_cache 1
      --enable_device_cache 1
      --enable_memory_cache 0
      --use_deepep_moe 1
      --use_deepep_low_latency 0
      --cp_rotate_method ALL_GATHER
      --reserver_runtime_mem_mb 71680
      --max_context_batch_size 1
      --concurrency_limit 1
      --fp8_kv_cache 1
      --sp_type dspark
      --gen_num_per_cycle 3
      --sp_model_type deepseek_v4_dspark
      --sp_checkpoint_path "$MODEL_DIR"
      --sp_act_type bf16
      --think_mode 1
      --enable_fp32_lm_head 0
      --frontend_server_count 1
      --start_port "$START_PORT_VALUE"
      --warm_up 1
      --model_warm_up 1
      --grammar_backend none
      --kv_cache_mem_mb 8192
    )
    ;;
  *)
    echo "unknown role: $ROLE" >&2
    exit 2
    ;;
esac

WORK_DIR="$BASE_WORK/$ROLE"
mkdir -p "$WORK_DIR" "$RUN_DIR/status" "$RUN_DIR/hook_dumps"

export SCR_PHASE=checkpoint
export SCR_ENABLE=1
export RTPLLM_ENABLE_SCR=1
export PYTHONNOUSERSITE=1
export HOST_NVIDIA_DRIVER_VERSION=580.95.05
export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export WORLD_SIZE=2
export WORLD_RANK=0
export LOCAL_WORLD_SIZE=2
# The wheel is installed in /opt/conda310, while the machine-provided
# epsilon API is installed in the operator's user site.  Keep the wheel first
# so this run exercises the rebuilt RTP-LLM code and make the Epsilon shim
# importable for the root-launched service.
export PYTHONPATH="/opt/conda310/lib/python3.10/site-packages:/home/serina.wzq/.local/lib/python3.10/site-packages"
export START_PORT="$START_PORT_VALUE"
export HIPPO_PROC_WORKDIR="$WORK_DIR"
export TIMESTAMP_STDOUT=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

export ENABLE_DSPARK=1
export SP_TYPE=dspark
export SP_MODEL_TYPE=deepseek_v4_dspark
export SP_CHECKPOINT_PATH="$MODEL_DIR"
export SP_ACT_TYPE=bf16
export GEN_NUM_PER_CIRCLE=3
export MODEL_EVAL_VARIANT=fp8_kv_cache
export DSV4_BF16_VLLM=0
export DSV4_USE_FRAMEWORK_KV=1
export DSV4_COMPRESSOR_FAST=1
export DSV4_COMPRESSOR_METADATA_TRITON=0
export DSV4_MHC_PRE_GEMM_BACKEND=tilelang_single
export DSV4_HCA_STATE_POOL_BLOCKS=200
export DSV4_FIXED_POOL_USE_MEMORY=0
export DSV4_TRAP_INVALID_KV_ACCESS=0
export DSV4_PREFILL_CP_OVERLAP=1
export DSV4_CHUNK_TOKENS=12288
export DEVICE_CACHE_MIN_FREE_BLOCKS=5000
export ENABLE_GPU_PREFIX_TREE=1
export ENABLE_DSV4_STATE_BLOCK_INDEPENDENT_EVICTION=0
export ENABLE_LEGACY_MEMORY_CONNECTOR_FALLBACK=0
export CP_FORCE_SINGLE_PREFILL=0
export PREFILL_CP_KV_CACHE_SHARDED=0
export RTP_LLM_PIN_HOST_BLOCK_POOL=0
export RTP_LLM_DEVICE_INPUT=1
export ENABLE_LAYER_MICRO_BATCH=0
export LINEAR_STEP=1
export DG_JIT_CPP_STANDARD=20
export DG_MEGA_MOE_NVLINK_BARRIER_TIMEOUT_SECS=300
# Reuse the machine-local CUDA13/SM100 DeepGEMM cache populated by the
# preceding smoke run; this avoids recompiling the ~90 prefill shapes.
export DG_JIT_CACHE_DIR="/tmp/rtp-llm/.jit_cache/v1/deep_gemm/deep_gemm-cuda-13_0-sm_100-2_6_1_09cd3ee_cu132"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:1024
export FRONTEND_SERVER_COUNT=1
export GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS=500
export THINK_MODE=1
export THINK_START_TAG='<think>'
export THINK_END_TAG='</think>'
export RTP_LLM_STREAM_ASYNC=1
export RTP_LLM_DROP_BROAD_SYNC=1
export RTP_LLM_MTP_ASYNC_PREPARE=0
export GRAMMAR_BACKEND=none
export DETERMINISTIC_GEMM=1
export DSV4_INDEXER_TOPK_CANONICALIZE=1
export NCCL_NET_PLUGIN=none
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5

export REMOTE_SERVER_PORT="$REMOTE_PORT_VALUE"
export REMOTE_RPC_SERVER_IP=localhost
export MODEL_SERVICE_CONFIG='{"service_id":"test","role_endpoints":[{"group":"default","prefill_endpoint":{"type":"Vipserver","address":"127.0.0.1:18630","protocol":"http","path":"/"},"decode_endpoint":{"type":"Vipserver","address":"127.0.0.1:18530","protocol":"http","path":"/"}}],"use_local":true}'

# The integrated worker/CPU manifest owns the Epsilon call.  Do not install
# the historical backend-only hot hook: it would call snapstart_checkpoint a
# second time with worker_num=2 and create a second, incompatible quorum.
export RTP_HOT_HOOK=0
unset RTP_HOT_HOOK_FILE RTP_HOT_HOOK_CONFIG RTP_HOT_HOOK_DUMP_DIR
export CMD_BEFORE_START='export LD_PRELOAD=/etc/scr/shadow/libnccl.so:/usr/lib64/librt.so.1'

exec /usr/bin/maga_start.sh \
  --model_type deepseek_v4 \
  --checkpoint_path "$MODEL_DIR" \
  --tokenizer_path "$MODEL_DIR" \
  "${ROLE_ARGS[@]}"

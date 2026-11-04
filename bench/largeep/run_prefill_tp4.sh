#!/bin/bash
# Prefill-only P (TP4 on 4 GPUs of a third host) that hands KV off to the EP16 decode group.
#   WS MODEL    as run_decode_ep16.sh
#   D_ADDRS     comma list of the 4 D DP leaders ip:http_port (DP leader = local rank 0 and 4 of each D host;
#               http ports PORT and PORT+36 with the default port layout, e.g. HOST0:31200,HOST0:31236,HOST1:...)
# usage: WS=... MODEL=... D_ADDRS=... [GPUS=0,1,2,3] [PORT=31100] bash run_prefill_tp4.sh
ulimit -c 0
: "${WS:?}" "${MODEL:?}" "${D_ADDRS:?}"
PORT=${PORT:-31100}
W=${RUN_DIR:-$WS/../largeep_prefill_run}
# TMPDIR holds the engine's UNIX sockets (path limit 107 bytes): keep it short
TMP=${TMP_DIR:-/tmp/largeep_${USER:-u}_$PORT}
mkdir -p $W/cache $TMP && cd $W
CUDA=${CUDA_HOME:-/usr/local/cuda}
PY=${PYTHON:-python3}
export PYTHONPATH=$WS:$WS/bazel-bin:$WS/rtp_llm/cpp/model_rpc/proto:$WS/rtp_llm/dash_sc/proto
export LD_LIBRARY_PATH=${RDMA_CORE:+$RDMA_CORE/lib:}$CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH
export TILELANG_CACHE_DIR=$W/cache/tilelang TRITON_CACHE_DIR=$W/cache/triton TMPDIR=$TMP
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
export CACHE_STORE_RDMA_MODE=${CACHE_STORE_RDMA_MODE:-0}
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=${IFNAME:-eth0}
export ENABLE_FLASHINFER_TRTLLM_GEN=0
export DSV4_HC_IMPL=tilelang DSV4_INDEXER_CACHE_MODE=fp8
export DSV4_FIXED_POOL_BLOCKS=${FIXED_POOL:-256} DSV4_HCA_STATE_POOL_BLOCKS=${HCA_POOL:-64}
export DSV4_MHC_PRE_GEMM_BACKEND=deepgemm DSV4_SM120_WOA_EINSUM=1
export DSV4_SM120_TP_MOE=1 RTP_LLM_DEVICE_INPUT=1 RTP_LLM_MTP_ASYNC_PREPARE=0
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export MODEL_SERVICE_CONFIG="{\"service_id\":\"aigc.text-generation.generation.engine_service\",\"use_local\":true,\"role_endpoints\":[{\"group\":\"largeep\",\"decode_endpoint\":{\"type\":\"domain\",\"address\":\"${D_ADDRS}\",\"protocol\":\"http\",\"path\":\"/\"}}]}"
export "DOMAIN_ADDRESS:${D_ADDRS}=${D_ADDRS}"
unset TOKENIZER_PATH CHECKPOINT_PATH MODEL_CONFIG DEPLOYMENT_CONFIG kmonitorTags role GPU_MODEL_DETAIL HIPPO_SLAVE_HTTP_PORT MODEL_TYPE DP_SIZE TP_SIZE WORLD_SIZE LOCAL_WORLD_SIZE
exec $PY -m rtp_llm.start_server \
  --model_type deepseek_v4 --checkpoint_path $MODEL --tokenizer_path $MODEL \
  --tp_size 4 --ep_size 4 --dp_size 1 --pp_size 1 --world_size 4 --local_world_size 4 \
  --role_type PREFILL --start_port ${PORT} \
  --sp_type mtp --sp_model_type deepseek_v4_mtp --sp_checkpoint_path $MODEL --gen_num_per_cycle 2 \
  --act_type BF16 --quantization FP8_PER_BLOCK --fp8_kv_cache 1 \
  --seq_size_per_block 256 --kernel_seq_size_per_block 128 \
  --use_deepep_moe 0 --use_deepep_low_latency 0 \
  --enable_cuda_graph 0 --reserver_runtime_mem_mb ${RESERVE_MB:-24576} \
  --max_seq_len ${MSL:-8192} --max_context_batch_size ${MAX_CTX_BS:-4} --concurrency_limit 64 \
  --reuse_cache 0 --enable_remote_cache 0 --warm_up 0 --model_warm_up 0 \
  --max_batch_tokens_size 8192 \
  --load_method scratch --use_batch_decode_scheduler 0 \
  --frontend_pre_stop_drain_seconds 0 --dash_sc_grpc_pre_stop_drain_seconds 0 --shutdown_timeout 60

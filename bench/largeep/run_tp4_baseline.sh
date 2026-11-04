#!/bin/bash
# TP4 baseline for the EP16 comparison: one 4-GPU TP4 group, PD fusion with prefill only at admission
# (decode_prefill_ratio 0), MTP 2+1, CUDA graph, its best collectives (FlashInfer PCIe IPC all-reduce + shared expert
# overlap). Measured with bench/decode/decode_plateau_bench.py at 16 streams (= 4 streams per GPU, the same load as
# EP16 at 64 streams); see docs/largeep/README.md "TP4 baseline".
#   WS MODEL   as run_decode_ep16.sh;   FI_PCIE   fi_up directory (setup_fi_up.sh); unset = NCCL all-reduce
# usage: WS=... MODEL=... [FI_PCIE=...] [GPUS=0,1,2,3] [PORT=27000] bash run_tp4_baseline.sh
ulimit -c 0
: "${WS:?}" "${MODEL:?}"
PORT=${PORT:-27000}
W=${RUN_DIR:-$WS/../largeep_tp4_run}
TMP=${TMP_DIR:-/tmp/largeep_${USER:-u}_$PORT}
mkdir -p $W/cache $TMP && cd $W
CUDA=${CUDA_HOME:-/usr/local/cuda}
PY=${PYTHON:-python3}
export PYTHONPATH=$WS:$WS/bazel-bin:$WS/rtp_llm/cpp/model_rpc/proto:$WS/rtp_llm/dash_sc/proto
export LD_LIBRARY_PATH=${RDMA_CORE:+$RDMA_CORE/lib:}$CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH
export TILELANG_CACHE_DIR=$W/cache/tilelang TRITON_CACHE_DIR=$W/cache/triton TMPDIR=$TMP
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
export CACHE_STORE_RDMA_MODE=0 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS
export ENABLE_FLASHINFER_TRTLLM_GEN=0
export DSV4_HC_IMPL=tilelang DSV4_SM120_PACK_DECODE_SLOTS=1 DSV4_PP_EP_ENABLE=0 DSV4_INDEXER_CACHE_MODE=fp8
export DSV4_FIXED_POOL_BLOCKS=1024 DSV4_HCA_STATE_POOL_BLOCKS=320 DSV4_MOE_CHUNK_TOKENS=4096
export DSV4_MHC_PRE_GEMM_BACKEND=deepgemm DSV4_SM120_WOA_EINSUM=1
export DSV4_SM120_TP_MOE=1 RTP_LLM_DEVICE_INPUT=1 RTP_LLM_MTP_ASYNC_PREPARE=0
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export DSV4_SHARED_EXPERT_MODE=overlap DSV4_SHARED_EXPERT_OVERLAP_IN_GRAPH=1
if [ -n "${FI_PCIE:-}" ]; then
  export DSV4_FI_PCIE_PATH=$FI_PCIE DSV4_PCIE_ONESHOT_AR=1 DSV4_PCIE_AR_BACKEND=fi
fi
unset TOKENIZER_PATH CHECKPOINT_PATH MODEL_CONFIG DEPLOYMENT_CONFIG kmonitorTags role GPU_MODEL_DETAIL HIPPO_SLAVE_HTTP_PORT MODEL_TYPE DP_SIZE TP_SIZE WORLD_SIZE LOCAL_WORLD_SIZE
exec $PY -m rtp_llm.start_server \
  --model_type deepseek_v4 --checkpoint_path $MODEL --tokenizer_path $MODEL \
  --tp_size 4 --ep_size 4 --dp_size 1 --pp_size 1 --world_size 4 --local_world_size 4 \
  --role_type PDFUSION --start_port ${PORT} \
  --sp_type mtp --sp_model_type deepseek_v4_mtp --sp_checkpoint_path $MODEL --gen_num_per_cycle 2 \
  --act_type BF16 --quantization FP8_PER_BLOCK --fp8_kv_cache 1 \
  --seq_size_per_block 256 --kernel_seq_size_per_block 128 \
  --use_deepep_moe 0 --use_deepep_low_latency 0 \
  --enable_cuda_graph 1 --reserver_runtime_mem_mb ${RESERVE_MB:-11264} \
  --max_seq_len 40960 --max_context_batch_size 1 --concurrency_limit 64 \
  --reuse_cache 0 --enable_remote_cache 0 --warm_up 0 --model_warm_up 0 \
  --max_batch_tokens_size 8192 --decode_capture_config "1,2,4,8,16,24,32,40,48,64" --prefill_capture_config 8192 \
  --load_method scratch --use_batch_decode_scheduler 0 --pdfusion_scheduler_mode ratio --decode_prefill_ratio 0 \
  --frontend_pre_stop_drain_seconds 0 --dash_sc_grpc_pre_stop_drain_seconds 0 --shutdown_timeout 60

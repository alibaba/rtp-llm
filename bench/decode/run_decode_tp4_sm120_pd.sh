#!/bin/bash
# 5kpro (SM120) TP4 MTP2 DECODE for a PD setup: receives KV from a prefill (e.g. run_prefill_cep4pp2_mtp_fastgen.sh)
# over eRDMA or TCP. Real MTP acceptance, CUDA graph on, ASYNC_PREPARE off (it drops acceptance at batch >= 2 with the
# compact draft layout). One 8-GPU host runs two instances: GPUS=0,1,2,3 PORT=21000 and GPUS=4,5,6,7 PORT=22000.
#   WS          built source tree (bazel-bin with the internal cache-store overlay)       [required]
#   DEEPGEMM    dir holding the SM120 DeepGEMM package, prepended to PYTHONPATH           [optional]
#   EXTRA_PY    extra site-packages dir appended to PYTHONPATH                           [optional]
#   RDMA_LIB    eRDMA user-space lib dir (rdma-core build-58 on SM120 hosts)              [required for RDMA]
#   GPUS PORT   default 0,1,2,3 / 21000; GPU4-7 uses erdma_1, GPU0-3 erdma_0
#   RDMA=0      KV over TCP (default 1)
#   INDEXER_MODE fp4 (default) or fp8; must match the prefill side
#   RUN         run/log dir (default ./d4-$PORT)
# The prefill must not set PD_SERVING_CP_RANK=0 for this decode (see run_prefill_cep4pp2_mtp_fastgen.sh).
set -u
unset TOKENIZER_PATH CHECKPOINT_PATH MODEL_CONFIG DEPLOYMENT_CONFIG kmonitorTags role GPU_MODEL_DETAIL HIPPO_SLAVE_HTTP_PORT MODEL_TYPE DP_SIZE TP_SIZE WORLD_SIZE LOCAL_WORLD_SIZE
ulimit -c 0
: "${WS:?set WS to the built source tree}"
PORT=${PORT:-21000}
GPUS=${GPUS:-0,1,2,3}
RUN=${RUN:-$PWD/d4-$PORT}; mkdir -p $RUN/cache/tmp; cd $RUN
NIC=erdma_0; [ "${GPUS%%,*}" -ge 4 ] && NIC=erdma_1
export PYTHONPATH=${DEEPGEMM:+$DEEPGEMM:}$WS:$WS/bazel-bin:$WS/rtp_llm/cpp/model_rpc/proto:$WS/rtp_llm/dash_sc/proto${EXTRA_PY:+:$EXTRA_PY}
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=${RDMA_LIB:+$RDMA_LIB:}/usr/local/cuda/lib64:/opt/conda310/lib
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export TILELANG_CACHE_DIR=$RUN/cache/tilelang TRITON_CACHE_DIR=$RUN/cache/triton TMPDIR=$RUN/cache/tmp
export CUDA_VISIBLE_DEVICES=$GPUS
if [ "${RDMA:-1}" = 1 ]; then
  export ACCL_SET_ERDMA=1 ACCL_USE_NICS=$NIC RTP_CACHE_STORE_RDMA_DEVICES=$NIC
  export CACHE_STORE_RDMA_QP_COUNT_PER_CONNECTION=${RDMA_QP:-8} CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS=3000
fi
export CACHE_STORE_RDMA_MODE=${RDMA:-1}
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS
export ENABLE_FLASHINFER_TRTLLM_GEN=0
export DSV4_HC_IMPL=tilelang
export DSV4_SM120_PACK_DECODE_SLOTS=1
export DSV4_PP_EP_ENABLE=0
export DSV4_INDEXER_CACHE_MODE=${INDEXER_MODE:-fp4}
[ "${INDEXER_MODE:-fp4}" = fp4 ] && export DSV4_FP4_INDEXER_ALLOW_MTP2=1 DSV4_FP4_INDEXER_ALLOW_PREFIX_REUSE=1
export DSV4_FIXED_POOL_BLOCKS=1024 DSV4_HCA_STATE_POOL_BLOCKS=320 DSV4_MOE_CHUNK_TOKENS=4096
export DSV4_MHC_PRE_GEMM_BACKEND=deepgemm DSV4_SM120_WOA_EINSUM=1
export DSV4_SM120_TP_MOE=1 RTP_LLM_DEVICE_INPUT=1 RTP_LLM_MTP_ASYNC_PREPARE=0
export LOAD_CACHE_TIMEOUT_MS=1800000 P2P_LAYER_CACHE_BUFFER_STORE_TIMEOUT_MS=1800000 DSV4_PD_SKIP_MTP_MODULE_LOAD=1
export NO_PROXY=127.0.0.1,localhost,0.0.0.0${NO_PROXY_EXTRA:+,$NO_PROXY_EXTRA} no_proxy=127.0.0.1,localhost,0.0.0.0${NO_PROXY_EXTRA:+,$NO_PROXY_EXTRA}
CKPT=${CKPT:-/mnt/fuse/deepseek-ai/DeepSeek-V4-Flash}
exec setsid /opt/conda310/bin/python3 -m rtp_llm.start_server \
  --model_type deepseek_v4 --checkpoint_path $CKPT --tokenizer_path $CKPT \
  --tp_size 4 --ep_size 4 --dp_size 1 --pp_size 1 --world_size 4 --local_world_size 4 \
  --role_type DECODE --start_port $PORT \
  --sp_type mtp --sp_model_type deepseek_v4_mtp --sp_checkpoint_path $CKPT --gen_num_per_cycle ${GEN:-2} \
  --act_type BF16 --quantization FP8_PER_BLOCK --fp8_kv_cache 1 \
  --seq_size_per_block 256 --kernel_seq_size_per_block 256 --cache_store_rdma_mode ${RDMA:-1} \
  --use_deepep_moe 0 --use_deepep_low_latency 0 \
  --enable_cuda_graph 1 --reserver_runtime_mem_mb ${RESERVE_MB:-11264} \
  --max_seq_len 40960 --max_context_batch_size 1 --concurrency_limit ${CONC:-64} \
  --reuse_cache 0 --enable_remote_cache 0 --warm_up 0 --model_warm_up 0 \
  --max_batch_tokens_size 8192 --decode_capture_config "1,2,4,8,12,16,24,32,40,48,64" --prefill_capture_config 8192 \
  --load_method scratch --use_batch_decode_scheduler 0 \
  --frontend_pre_stop_drain_seconds 0 --dash_sc_grpc_pre_stop_drain_seconds 0 --shutdown_timeout 60 \
  > $RUN/decode.stdout 2>&1

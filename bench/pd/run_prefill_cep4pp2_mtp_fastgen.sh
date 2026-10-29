#!/bin/bash
# CEP4PP2 (PP2 x CP4, EP4) PREFILL for heterogeneous PD on 8 SM120 GPUs, MTP gen3 with fastgen chunked prefill.
# Env/args = _CEP4PP2_PREFILL_ARGS/_CEP4PP2_REFERENCE_ENV (rtp_llm/test/perf_test/BUILD) + PD overrides +
# the prefill optimizations below. Paths and the decode address come from the environment:
#   WS          built source tree (bazel-bin with the internal cache-store overlay)       [required]
#   DEEPGEMM    dir holding the SM120 DeepGEMM package, prepended to PYTHONPATH           [optional]
#   EXTRA_PY    extra site-packages dir appended to PYTHONPATH                           [optional]
#   RDMA_LIB    eRDMA user-space lib dir (rdma-core build-58 on SM120 hosts)              [required for RDMA]
#   DECODE_ADDR comma list of every decode DP rank's http ip:port (list them all, or every request
#               lands on DP rank 0)                                                       [required]
#   RUN         run/log dir (default ./run)
#   MTP_CHUNKED=0 FASTGEN_ARGS=' '  non-fastgen MTP prefill (same binary) for comparison
#   CP_SINGLE=0 MAX_CTX_BS=4  batched prefill: up to 4 requests per pass (default 1 per pass)
#   GEN         MTP draft depth (gen_num_per_cycle), default 3; must match the decode side. GEN=2 also sets
#               DSV4_FP4_INDEXER_ALLOW_MTP2=1 (MTP depth 2 with the FP4 indexer is opt-in)
set -u
unset RTP_LLM_LOG_TPSYNC TOKENIZER_PATH CHECKPOINT_PATH MODEL_CONFIG DEPLOYMENT_CONFIG kmonitorTags role GPU_MODEL_DETAIL HIPPO_SLAVE_HTTP_PORT MODEL_TYPE DP_SIZE TP_SIZE WORLD_SIZE LOCAL_WORLD_SIZE
ulimit -c 0
RUN=${RUN:-$PWD/run}; mkdir -p $RUN/logs; cd $RUN
: "${WS:?set WS to the built source tree}"
: "${DECODE_ADDR:?set DECODE_ADDR to every decode DP rank ip:http_port}"
export PYTHONPATH=${DEEPGEMM:+$DEEPGEMM:}$WS:$WS/bazel-bin:$WS/rtp_llm/cpp/model_rpc/proto:$WS/rtp_llm/dash_sc/proto${EXTRA_PY:+:$EXTRA_PY}
export PYTHONNOUSERSITE=1
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${RDMA_LIB:+$RDMA_LIB:}/usr/local/cuda/lib64:/opt/conda310/lib
export MODEL_SERVICE_CONFIG="{\"service_id\":\"hetero-pd-prefill\",\"use_local\":true,\"role_endpoints\":[{\"group\":\"default\",\"decode_endpoint\":{\"type\":\"domain\",\"address\":\"${DECODE_ADDR}\",\"protocol\":\"http\",\"path\":\"/\"}}]}"
export NO_PROXY=127.0.0.1,localhost,0.0.0.0${NO_PROXY_EXTRA:+,$NO_PROXY_EXTRA} no_proxy=127.0.0.1,localhost,0.0.0.0${NO_PROXY_EXTRA:+,$NO_PROXY_EXTRA}
export ACCL_SET_ERDMA=1
export ACCL_USE_NICS=erdma_0
export CACHE_STORE_RDMA_MODE=1
export CACHE_STORE_RDMA_QP_COUNT_PER_CONNECTION=${RDMA_QP:-8}
export CACHE_STORE_RDMA_WRITE_BLOCK_CONCAT=${RDMA_CONCAT:-0}
export CP_FORCE_SINGLE_PREFILL=${CP_SINGLE:-1}
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DG_JIT_CPP_STANDARD=20
export DSV4_BF16_VLLM=0
export DSV4_CHUNK_TOKENS=12288
export DSV4_CP_COMPACT_COMPRESSOR=1
export DSV4_CP_COMPACT_TRANSPORT_FUSION=1
export DSV4_FIXED_POOL_BLOCKS=0
export DSV4_FP4_PREFILL_BACKEND=${FP4_PREFILL:-native}
export DSV4_FUSED_PREPARE=1
export DSV4_FWD_GPU=0
export DSV4_FWD_PROFILE=0
export DSV4_FWD_PROFILE_IDX=24
export DSV4_FWD_STATS=0
export DSV4_GROUPED_FP4_ASYNC=0
export DSV4_HC_IMPL=${HC_IMPL:-tilelang}
export DSV4_INDEXER_CACHE_MODE=${INDEXER_MODE:-fp4}
export DSV4_FP4_INDEXER_ALLOW_PREFIX_REUSE=1
[ "${GEN:-3}" = "2" ] && export DSV4_FP4_INDEXER_ALLOW_MTP2=1
export DSV4_INDEXER_FP8_DEEPGEMM=1
export DSV4_INDEXER_FP8_DEEPGEMM_PAGED=0
export DSV4_INDEXER_TILE=16,256,8,3
export DSV4_INDEXER_TOPK_B7_RECONSTRUCT=0
export DSV4_INDEXER_TOPK_CANONICALIZE=1
export DSV4_M1_HOST_META=0
export DSV4_MHC_PRE_GEMM_BACKEND=tilelang_single
export DSV4_MOE_ACTIVE_SCALE_FUSION=1
export DSV4_MOE_ACTIVE_SCALE_FUSION_REQUIRED=1
export DSV4_MOE_DEVICE_META=1
export DSV4_MOE_EXTENT_COUNT_FUSE=1
export DSV4_MOE_FORWARD_COUNT_PLAN=1
export DSV4_MOE_FP4_BACKEND=flashinfer
export DSV4_MOE_LOCAL_REPLAY=1
export DSV4_MOE_LOCAL_REPLAY_EXCHANGE=0
export DSV4_MOE_LOCAL_REPLAY_EXCHANGE_REQUIRED=0
export DSV4_MOE_LOCAL_REPLAY_REQUIRED=0
export DSV4_MOE_PREQUANT_INPUT=1
export DSV4_MOE_PREQUANT_INPUT_REQUIRED=1
export DSV4_MOE_TILE_N=64
export DSV4_MOE_RING_DISPATCH=${RING_DISPATCH:-0}
export DSV4_MOE_RING_COMBINE=${RING_COMBINE:-nccl}
export DSV4_MOE_RETURN_DTYPE=mxfp8
export DSV4_NCCL_EP_MXFP8_DISPATCH_PACK=1
export DSV4_PP_EP_BACKEND=fork_nccl_mxfp8
export DSV4_PP_EP_ENABLE=1
export DSV4_PP_STAGE_LAYER_COUNTS=22,21
export DSV4_PP_EP_MTP_PREFILL=1
# Prefill optimizations (2026-10-20/28 A/B on 32K c1): the async/fusion group saves ~30 ms of TTFT in fastgen
# mode; the CP plan-cache group is neutral but kept to match the measured configuration.
export DSV4_PREFILL_ASYNC_HEAD=1 DSV4_FP8_INDEXER_TAIL_STREAM=1 DSV4_FP8_PREFILL_META_SHARED_BUILD=1
export DSV4_SM120_PREFILL_SPLIT_FUSION=1 DSV4_SM120_PREFILL_INDEX_ADAPTER=ready DSV4_MOE_EAGER_COUNT_GATHER=1
export DSV4_CP_PINNED_STAGE=1 DSV4_CP_COMPACT_PLAN_CACHE=1 DSV4_CP_COMPACT_DEFERRED_VERIFY=1 DSV4_CP_CONTEXT_PLAN_CACHE=1
export DSV4_PREFILL_CAPTURE=0
export DSV4_RECORD_FUNCTION_RANGES=0
export DSV4_SHARED_EXPERT_MODE=overlap
export DSV4_SHARED_EXPERT_REPLAY=0
export DSV4_SHARED_EXPERT_REPLAY_REQUIRED=0
export DSV4_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD=4096
export DSV4_SM120_BF16_PRETRANSPOSE=0
export DSV4_SM120_NATIVE_FP8=0
export DSV4_SM120_PACK_DECODE_SLOTS=0
export DSV4_SM120_SHARED_EXPERT_FUSED=0
export DSV4_SM120_SHARED_EXPERT_NATIVE_DG=0
export DSV4_SM120_WOA_EINSUM=1
export DSV4_STARTUP_REAL_WARMUP=${REAL_WARMUP:-0}
export DSV4_USE_FRAMEWORK_KV=1
export ENABLE_FLASHINFER_TRTLLM_GEN=0
export FAKE_BALANCE_EXPERT=0
export FASTSAFETENSORS_NOGDS=1
export GEN_TIMELINE_NUM_STEPS=0
export GEN_TIMELINE_START_STEP=0
export GEN_TIMELINE_SYNC=0
export LINEAR_STEP=1
export MAX_BATCH_SIZE=1024
export MAX_BATCH_TOKENS_SIZE=1048576
export NCCL_IB_DISABLE=1
export NCCL_MAX_P2P_NCHANNELS=8
export NCCL_MIN_P2P_NCHANNELS=8
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO=LL,LL128,Simple
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RTP_CACHE_STORE_RDMA_DEVICES=erdma_0
export RTP_CACHE_STORE_RDMA_ROLE=prefill
export RTP_LLM_LOG_FASTGEN=0
export RTP_LLM_PP_CHUNK_OVERLAP=2
export RTP_LLM_MTP_CHUNKED_PREFILL=${MTP_CHUNKED:-1}
export RTP_LLM_PP_OBJ_LOG=${PP_OBJ_LOG:-0}
export RTP_LLM_PD_SERVING_CP_RANK=${PD_SERVING_CP_RANK-0}
export RTP_LLM_PD_PUBLISH_AT_END=${PD_PUBLISH_AT_END:-0}
[ -n "${PD_PUBLISH_DEBUG:-}" ] && export RTP_LLM_PD_PUBLISH_DEBUG=1
export RTP_LLM_PP_ROUND_DEVICE_SYNC=1
export RTP_LLM_PP_ROUND_FWD_EVENT_SYNC=1
export SPT_NOENV=1
export USE_BATCH_DECODE_SCHEDULER=0
export USE_LOCAL=1
export WORLD_SIZE=8
CKPT=${CKPT:-/mnt/fuse/deepseek-ai/DeepSeek-V4-Flash}
exec setsid /opt/conda310/bin/python3 -m rtp_llm.start_server --model_type deepseek_v4 --role_type PREFILL --pp_size 2 --tp_size 4 --dp_size 1 --ep_size 4 --world_size 8 --cp_rotate_method PREFILL_CP --prefill_cp_size 4 --prefill_cp_kv_cache_sharded 0 --warm_up ${WARM_UP:-0} --model_warm_up ${WARM_UP:-0} --load_method scratch --max_seq_len ${MAX_SEQ:-34816} --enable_cuda_graph 0 --act_type BF16 --seq_size_per_block 256 --kernel_seq_size_per_block ${KBLOCK:-256} --cache_store_rdma_mode 1 --cache_store_rdma_connect_timeout_ms 800 --use_local 1 --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0 --memory_cache_size_mb 256 --write_cache_sync 1 --use_deepep_moe 0 --use_deepep_low_latency 0 --reserver_runtime_mem_mb 12288 --concurrency_limit ${CONC_LIMIT:-32} --max_context_batch_size ${MAX_CTX_BS:-1} --fp8_kv_cache 1 --enable_fp32_lm_head 0 ${FASTGEN_ARGS---enable_fast_gen 1 --fast_gen_context_budget 4096} --use_batch_decode_scheduler 0 --checkpoint_path $CKPT --tokenizer_path $CKPT --start_port 14000 --sp_type mtp --sp_model_type deepseek_v4_mtp --sp_checkpoint_path $CKPT --gen_num_per_cycle ${GEN:-3} --load_cache_timeout_ms 120000 --kv_cache_mem_mb ${PREFILL_KV_MB:-16000} --shutdown_timeout 60 --frontend_pre_stop_drain_seconds 0 --dash_sc_grpc_pre_stop_drain_seconds 0 > $RUN/prefill.stdout 2>&1

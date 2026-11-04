#!/bin/bash
# EP16 decode-only D for DeepSeek-V4 over 2 hosts x 8 GPUs (TP4 x DP4, EP = world = 16), MTP 2+1, CUDA graph,
# routed rail MoE all-to-all (comm/largeep_a2a) and the measured-best settings of docs/largeep/README.md.
# Run on BOTH hosts (NODE=0 on HOST0, NODE=1 on HOST1); they rendezvous through the gang config below.
#   WS          repo root of this branch (built: bazel-bin with librtp_compute_ops / libth_transformer)
#   MODEL       DeepSeek-V4-Flash checkpoint directory
#   HOST0 HOST1 management IPs of the two hosts
#   A2A_LIB     librail_routed2.so (comm/largeep_a2a/build.sh), default $WS/comm/largeep_a2a/librail_routed2.so
#   FI_PCIE     directory of the vendored FlashInfer PCIe IPC package fi_up (docs/largeep/README.md); unset = NCCL
#   PHY2LOG     static EPLB placement json (bench/largeep/phy2log_active.json); unset = identity placement
#   RDMA_CORE   private rdma-core prefix if the system libibverbs has no provider for the NIC
#   PROXY_CPUS  comma list of 8 idle cores for the a2a proxy threads (one per local GPU)
# usage: NODE=0|1 WS=... MODEL=... HOST0=... HOST1=... [PORT=31200] bash run_decode_ep16.sh
ulimit -c 0
: "${WS:?}" "${MODEL:?}" "${HOST0:?}" "${HOST1:?}"
NODE=${NODE:?NODE=0 or 1}
PORT=${PORT:-31200}
W=${RUN_DIR:-$WS/../largeep_run}
# TMPDIR holds the engine's UNIX sockets (path limit 107 bytes): keep it short
TMP=${TMP_DIR:-/tmp/largeep_${USER:-u}_$PORT}
mkdir -p $W/cache $TMP && cd $W
CUDA=${CUDA_HOME:-/usr/local/cuda}
PY=${PYTHON:-python3}
export PYTHONPATH=$WS:$WS/bazel-bin:$WS/rtp_llm/cpp/model_rpc/proto:$WS/rtp_llm/dash_sc/proto
export LD_LIBRARY_PATH=${RDMA_CORE:+$RDMA_CORE/lib:}$CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH
export TILELANG_CACHE_DIR=$W/cache/tilelang TRITON_CACHE_DIR=$W/cache/triton TMPDIR=$TMP
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CACHE_STORE_RDMA_MODE=${CACHE_STORE_RDMA_MODE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-erdma_0,erdma_1} NCCL_DMABUF_ENABLE=0 NCCL_IB_GID_INDEX=${GID_INDEX:-1} NCCL_GIN_TYPE=1
export NCCL_SOCKET_IFNAME=${IFNAME:-eth0} GLOO_SOCKET_IFNAME=${IFNAME:-eth0} NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ENABLE_FLASHINFER_TRTLLM_GEN=0
export DSV4_HC_IMPL=tilelang DSV4_SM120_PACK_DECODE_SLOTS=1 DSV4_PP_EP_ENABLE=0 DSV4_INDEXER_CACHE_MODE=fp8
export DSV4_FIXED_POOL_BLOCKS=1024 DSV4_HCA_STATE_POOL_BLOCKS=320 DSV4_MOE_CHUNK_TOKENS=1024
export DSV4_MHC_PRE_GEMM_BACKEND=deepgemm DSV4_SM120_WOA_EINSUM=1
export RTP_LLM_DEVICE_INPUT=1 RTP_LLM_MTP_ASYNC_PREPARE=0
export LOAD_CACHE_TIMEOUT_MS=1800000 P2P_LAYER_CACHE_BUFFER_STORE_TIMEOUT_MS=1800000
export LOG_LEVEL=${LOG_LEVEL:-INFO}
# ---- decode-only DP control: equal batch across DP groups + slow-down gate (hold file, gate_agent.py) ----
export DSV4_DECODE_EQ_BS=${EQ_BS:-1}
export DSV4_DECODE_HOLD_FILE=${HOLD_FILE:-/dev/shm/dsv4_decode_hold}
# ---- MoE: routed rail all-to-all ----
export DSV4_MOE_STRATEGY=sm120_rail_routed
export DSV4_SM120_RAIL_ROUTED_LIB=${A2A_LIB:-$WS/comm/largeep_a2a/librail_routed2.so}
export DSV4_SM120_RAIL_NIC_MAP=${NIC_MAP:-0,0,0,0,1,1,1,1}
export DSV4_SM120_RAIL_NPAD_MAX=${NPAD_MAX:-192} DSV4_SM120_RAIL_FLAG_MODE=2 DSV4_SM120_RAIL_SPIN_NS=${SPIN_NS:-5000000000}
export DSV4_SM120_RAIL_STATS_S=${STATS_S:-30}
[ -n "${PROXY_CPUS:-}" ] && export DSV4_SM120_RAIL_CPU=$PROXY_CPUS
# TS_DUMP=<trigger file>: per-call device timestamps (gate_agent.py POST /ts touches it; comm/largeep_a2a/tools/)
[ -n "${TS_DUMP:-}" ] && export DSV4_SM120_RAIL_TS_DUMP=$TS_DUMP DSV4_SM120_RAIL_TS_CALLS=256 DSV4_SM120_RAIL_TS_COLS=14
# ---- measured-best decode settings (docs/largeep/README.md "Configuration") ----
export DSV4_SHARED_EXPERT_MODE=overlap DSV4_SHARED_EXPERT_OVERLAP_IN_GRAPH=1
export DSV4_MOE_MASKED_EXPECTED_M_CAP=${EXPECTED_M_CAP:-64}
[ -n "${PHY2LOG:-}" ] && export DSV4_PHY2LOG_PATH=$PHY2LOG
if [ -n "${FI_PCIE:-}" ]; then
  export DSV4_FI_PCIE_PATH=$FI_PCIE DSV4_PCIE_ONESHOT_AR=1 DSV4_PCIE_AR_BACKEND=fi DSV4_TP_RSAG=${TP_RSAG:-1}
fi
cat > $W/gang.json <<EOG
{
  "largeep_part0": {"name": "largeep_part0", "ip": "${HOST0}", "port": ${PORT}},
  "largeep_part1": {"name": "largeep_part1", "ip": "${HOST1}", "port": ${PORT}}
}
EOG
export DISTRIBUTE_CONFIG_FILE=$W/gang.json
export WORLD_RANK=$((NODE * 8))
unset TOKENIZER_PATH CHECKPOINT_PATH MODEL_CONFIG DEPLOYMENT_CONFIG kmonitorTags role GPU_MODEL_DETAIL HIPPO_SLAVE_HTTP_PORT MODEL_TYPE DP_SIZE TP_SIZE WORLD_SIZE LOCAL_WORLD_SIZE
exec $PY -m rtp_llm.start_server \
  --model_type deepseek_v4 --checkpoint_path $MODEL --tokenizer_path $MODEL \
  --tp_size 4 --ep_size 16 --dp_size 4 --pp_size 1 --world_size 16 --local_world_size 8 \
  --role_type DECODE --start_port ${PORT} \
  --sp_type mtp --sp_model_type deepseek_v4_mtp --sp_checkpoint_path $MODEL --gen_num_per_cycle 2 \
  --act_type BF16 --quantization FP8_PER_BLOCK --fp8_kv_cache 1 \
  --seq_size_per_block 256 --kernel_seq_size_per_block 128 \
  --use_deepep_moe 0 --use_deepep_low_latency 0 \
  --enable_cuda_graph ${GRAPH:-1} --reserver_runtime_mem_mb ${RESERVE_MB:-32768} \
  --max_seq_len ${MSL:-8192} --max_context_batch_size 1 --concurrency_limit ${CONC:-64} \
  --reuse_cache 0 --enable_remote_cache 0 --warm_up 0 --model_warm_up 0 \
  --max_batch_tokens_size ${MBT:-8192} --decode_capture_config "${CAPTURE:-1,2,4,8,16,24,32,40,48,64}" \
  --load_method scratch --use_batch_decode_scheduler 0 \
  --frontend_pre_stop_drain_seconds 0 --dash_sc_grpc_pre_stop_drain_seconds 0 --shutdown_timeout 60

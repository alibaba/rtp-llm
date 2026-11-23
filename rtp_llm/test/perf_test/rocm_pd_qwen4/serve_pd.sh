#!/usr/bin/env bash
set -euo pipefail
role=${1:?usage: serve_pd.sh prefill|decode}
# Opt-in profile; use ASYNC=0 for the synchronous control.
async_mode=${ASYNC:-1}
out=${WORK_DIR:-$PWD/pd_qwen4_run}
model=${MODEL_DIR:?set MODEL_DIR to the checkpoint directory}
python=${RTP_PYTHON:-python3}
if [[ "$role" == prefill ]]; then
  export CUDA_VISIBLE_DEVICES=0,1
  reuse=true
  export ROLE_TYPE=PREFILL REMOTE_SERVER_PORT=29800
  port=29700
elif [[ "$role" == decode ]]; then
  export CUDA_VISIBLE_DEVICES=2,3
  reuse=false
  export ROLE_TYPE=DECODE REMOTE_SERVER_PORT=29700
  export RTP_LLM_MTP_DRAFT_LOOP_GRAPH="${RTP_LLM_MTP_DRAFT_LOOP_GRAPH:-1}"
  port=29800
else
  echo "usage: $0 prefill|decode" >&2; exit 2
fi
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export RTP_LLM_STREAM_ASYNC="$async_mode"
export RTP_LLM_MTP_ASYNC_PREPARE="$async_mode"
export RTP_LLM_MTP_ASYNC_DRAFT_PREPARE_POST_REJECTION=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export NO_PROXY='*' no_proxy='*'
export WORLD_SIZE=2 LOCAL_WORLD_SIZE=2 FRONTEND_SERVER_COUNT=1
export ATOM_LOADER_NUM_THREADS=1
export DECODE_ENTRANCE=0 CACHE_STORE_RDMA_MODE="${CACHE_STORE_RDMA_MODE:-0}" USE_LOCAL=1
export REMOTE_RPC_SERVER_IP=127.0.0.1
export LOG_PATH="$out/$role/logs_intended"
export FLYDSL_RUNTIME_CACHE_DIR="$out/$role/jit_intended/flydsl"
export AITER_JIT_DIR="${AITER_JIT_DIR:-$out/jit/aiter}"
export ATOM_QSA_FLYDSL_ATTN=1
export MODEL_SERVICE_CONFIG='{"service_id":"rocm_pd_qwen4","role_endpoints":[{"group":"default","prefill_endpoint":{"type":"Vipserver","address":"127.0.0.1:29700","protocol":"http","path":"/"},"decode_endpoint":{"type":"Vipserver","address":"127.0.0.1:29800","protocol":"http","path":"/"}}],"use_local":true}'
mkdir -p "$LOG_PATH" "$FLYDSL_RUNTIME_CACHE_DIR"
mkdir -p "$out/$role/intended_run"; cd "$out/$role/intended_run"
exec "$python" -m rtp_llm.start_server \
  --checkpoint_path "$model" --tokenizer_path "$model" --model_type qwen4_exp \
  --external_model_packages atom.plugin.rtpllm.models \
  --tp_size 2 --ep_size 2 --dp_size 1 \
  --max_seq_len 32768 --concurrency_limit 16 --max_context_batch_size 8 \
  --max_batch_tokens_size 196608 --prefill_chunk_size 0 \
  --kv_cache_mem_mb 66570 --seq_size_per_block 64 \
  --reuse_cache "$reuse" --reuse_cache_tail_blocks 3 \
  --enable_cuda_graph true --decode_capture_config 1,2,4,8,16 \
  --sp_type mtp --sp_model_type qwen4_exp_mtp \
  --sp_checkpoint_path "$model" --gen_num_per_cycle 3 \
  --pdfusion_scheduler_mode ratio --decode_prefill_ratio 0 \
  --load_cache_timeout_ms 120000 --start_port "$port"

#!/usr/bin/env bash
# A/B runner for DSV3.2 lossy KV offload (scheme B tier-2).
#   MODE=A : no offload (baseline)   MODE=B : capacity offload + lossy third-pool
set -uo pipefail
RUNTIME=${RUNTIME:-/home/admin/rtp-hol/runtime/rtp-b-offload-20260819-ring}
MODEL=/home/admin/models/DeepSeek-V3.2-Exp
PORT=${PORT:-26100}
MODE=${MODE:-A}
LOG=/home/admin/rtp-hol/logs/ab-${MODE}-$(date +%m%d-%H%M%S).log
mkdir -p /home/admin/rtp-hol/logs
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

offload_env=()
if [[ "$MODE" == "B" ]]; then
  offload_env=(
    V32_OFFLOAD_MODE=capacity
    RTP_KV_OFFLOAD_KEEP_BLOCKS=${KEEP:-256}
    RTP_KV_OFFLOAD_MIN_SEQ=${MINSEQ:-16384}
    RTP_KV_OFFLOAD_AFTER_TOKENS=${AFTER:-16}
    RTP_KV_OFFLOAD_STAGING_BLOCKS=${STG:-32}
    V32_IDX_POOL_BLOCKS=${IDXNB:-8192}
    V32_SINGLE_WAVE=${SINGLE:-1}
    V32_LOSSY=${LOSSY:-1}
    V32_LOSSY_PREFETCH=${PREFETCH:-8}
  )
fi
[[ -n "${EXTRA_ENV:-}" ]] && offload_env+=($EXTRA_ENV)

log "MODE=$MODE log=$LOG"
[[ ${#offload_env[@]} -gt 0 ]] && log "offload env: ${offload_env[*]}"

PYTHONPATH=$RUNTIME/site-packages \
LD_LIBRARY_PATH=$RUNTIME/site-packages/rtp_llm/libs:/opt/conda310/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib64 \
MODEL_TEMPLATE_TYPE=${MODEL_TEMPLATE_TYPE:-deepseek_v31} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
env ${offload_env[@]+"${offload_env[@]}"} \
nohup /opt/conda310/bin/python -m rtp_llm.start_server \
  --checkpoint_path "$MODEL" --tokenizer_path "$MODEL" \
  --model_type deepseek_v32 --act_type bf16 \
  --start_port $PORT --use_local 1 --max_seq_len ${MAXSEQ:-139264} --warm_up 0 \
  --enable_cuda_graph ${GRAPH:-0} --seq_size_per_block 64 \
  --concurrency_limit 160 --max_context_batch_size 1 \
  --kv_cache_mem_mb ${KVMB:-20480} --reserver_runtime_mem_mb 8192 \
  --tp_size ${TP:-2} --ep_size 8 --dp_size ${DP:-4} \
  --world_size 8 --world_rank 0 --local_world_size 8 \
  >"$LOG" 2>&1 &
echo $! > /home/admin/rtp-hol/logs/ab.pid
echo "$LOG" > /home/admin/rtp-hol/logs/ab.logpath

for i in $(seq 1 360); do
  if curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "healthy after ~$((i*10))s"; exit 0
  fi
  if ! pgrep -f "rtp_llm.start_server" >/dev/null 2>&1; then
    log "server died; tail:"; tail -40 "$LOG"; exit 2
  fi
  sleep 10
done
log "timeout waiting health"; tail -40 "$LOG"; exit 1

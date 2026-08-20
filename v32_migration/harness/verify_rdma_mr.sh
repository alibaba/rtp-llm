#!/usr/bin/env bash
# Verify KV block pool RDMA MR registration (r6 cudaMalloc backing wiring).
# Strictly filters engine.log lines newer than server start time.
set -uo pipefail
RUNTIME=/home/admin/rtp-hol/runtime/rtp-b-offload-20260819-ring
MODEL=/home/admin/models/DeepSeek-V3.2-Exp
PORT=27801
LOG=/tmp/rdma-mr-verify.log
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

pkill -f "start_port $PORT" 2>/dev/null; sleep 2
START_TS=$(date "+%Y-%m-%d %H:%M:%S")
PYTHONPATH=$RUNTIME/site-packages \
LD_LIBRARY_PATH=$RUNTIME/site-packages/rtp_llm/libs:/opt/conda310/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib64 \
MODEL_TEMPLATE_TYPE=deepseek_v31 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
V32_OFFLOAD_MODE=off \
nohup /opt/conda310/bin/python -m rtp_llm.start_server \
  --checkpoint_path "$MODEL" --tokenizer_path "$MODEL" \
  --model_type deepseek_v32 --act_type bf16 \
  --role_type DECODE --cache_store_rdma_mode 1 \
  --start_port $PORT --use_local 1 \
  --remote_rpc_server_ip 127.0.0.1 --remote_server_port 28901 \
  --load_cache_timeout_ms 120000 \
  --max_seq_len 65536 --warm_up 0 --enable_cuda_graph 0 \
  --seq_size_per_block 64 --concurrency_limit 16 \
  --kv_cache_mem_mb 12288 --reserver_runtime_mem_mb 8192 \
  --max_context_batch_size 1 \
  --tp_size 1 --ep_size 8 --dp_size 8 \
  --world_size 8 --world_rank 0 --local_world_size 8 \
  >"$LOG" 2>&1 &
SRV=$!
log "server pid=$SRV started at [$START_TS], watching engine.log (timeout 30min)"

check_recent() {
  # only lines with timestamp >= START_TS
  awk -v ts="$START_TS" -F'[][]' '{ if ($2 >= ts) print }' /home/admin/logs/engine.log 2>/dev/null \
    | grep -E "cudaMalloc backing|RDMA cache store enabled|register user mr|reg user mr"
}

for i in $(seq 1 180); do
  RECENT=$(check_recent | tail -30)
  if echo "$RECENT" | grep -q "register user mr"; then
    log "=== MR log lines (since start) ==="
    echo "$RECENT" | cut -c1-230
    if echo "$RECENT" | grep -q "register user mr success"; then
      log "VERDICT: MR_REG_SUCCESS"
    elif echo "$RECENT" | grep -qE "failed"; then
      log "VERDICT: MR_REG_FAILED"
    fi
    break
  fi
  if ! kill -0 $SRV 2>/dev/null; then
    log "server died early; server log tail:"
    tail -30 "$LOG" | cut -c1-220
    RECENT=$(check_recent | tail -20)
    [ -n "$RECENT" ] && echo "$RECENT" | cut -c1-230
    break
  fi
  sleep 10
done
pkill -f "start_port $PORT" 2>/dev/null
log "verify done (server stopped)"

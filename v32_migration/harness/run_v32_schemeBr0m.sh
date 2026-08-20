#!/usr/bin/env bash
# Br0m: staging-ring admission judgment run — identical to B0m (mode=capacity,
# dataset=mixed, W48, NO admission retry / native instant-reject) plus
# RTP_KV_ADMIT_RING_BLOCKS=64. Direct pairing: B0m long 1 ok / 127 fail.
set -uo pipefail
BASE=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-06_rtp-llm-decode-routing-compare
HERE=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-12_dsv32-longctx-p1
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }
source "$HERE/v32_env.sh"

export RUNTIME_ENV="$HERE/br-runtime.env"
export FORMAL_REQUESTS=1000
export RTP_DECODE_CONCURRENCY=128
export RTP_EXPECTED_GPU_TOTAL=32
unset RTP_DECODE_MALLOC_RETRY_MS
export RTP_WORKER_EXTRA_ENV="${RTP_WORKER_EXTRA_ENV},V32_OFFLOAD_MODE=capacity,RTP_KV_OFFLOAD_KEEP_BLOCKS=256,RTP_KV_OFFLOAD_MIN_SEQ=16384,RTP_KV_OFFLOAD_AFTER_TOKENS=16,RTP_KV_OFFLOAD_STAGING_BLOCKS=32,V32_IDX_POOL_BLOCKS=8192,RTP_KV_ADMIT_RING_BLOCKS=64"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
log "=== v32 Br0m start: ring admission, ds=mixed w48 noretry"
RUN_ID="v32br0m-$STAMP-schemeBr0m-ring-noretry-w48-mixed" SCHEME=LOAD_ONLY WORKERS=48 RATIO="3:1" \
  "$BASE/run_v6_cell.sh" >>"$HERE/v32br0m-$STAMP.log" 2>&1
log "=== v32 Br0m end rc=$?"

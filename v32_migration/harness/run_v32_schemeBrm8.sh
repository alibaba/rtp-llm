#!/usr/bin/env bash
# Brm8: ring admission (16k window, keep256+ring64) + decode 8-rank fanout
# (RTP_DECODE_RANK_FANOUT via patched rtp_cluster.py: FlexLB sees one endpoint
# per DP rank instead of rank0 only). Same dataset/W/reject semantics as B0m.
set -uo pipefail
BASE=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-06_rtp-llm-decode-routing-compare
HERE=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-12_dsv32-longctx-p1
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }
source "$HERE/v32_env.sh"

export RUNTIME_ENV="$HERE/br-runtime.env"
export FORMAL_REQUESTS=1000
export RTP_DECODE_CONCURRENCY=128
export RTP_EXPECTED_GPU_TOTAL=32
export RTP_DECODE_RANK_FANOUT=1
unset RTP_DECODE_MALLOC_RETRY_MS
export RTP_WORKER_EXTRA_ENV="${RTP_WORKER_EXTRA_ENV},V32_OFFLOAD_MODE=capacity,RTP_KV_OFFLOAD_KEEP_BLOCKS=256,RTP_KV_OFFLOAD_MIN_SEQ=16384,RTP_KV_OFFLOAD_AFTER_TOKENS=16,RTP_KV_OFFLOAD_STAGING_BLOCKS=32,V32_IDX_POOL_BLOCKS=8192,RTP_KV_ADMIT_RING_BLOCKS=64"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
log "=== v32 Brm8 start: ring admission keep256 + 8-rank fanout, ds=mixed w48 noretry"
RUN_ID="v32brm8-$STAMP-schemeBrm8-ring-fanout8-w48-mixed" SCHEME=LOAD_ONLY WORKERS=48 RATIO="3:1" \
  "$BASE/run_v6_cell.sh" >>"$HERE/v32brm8-$STAMP.log" 2>&1
log "=== v32 Brm8 end rc=$?"

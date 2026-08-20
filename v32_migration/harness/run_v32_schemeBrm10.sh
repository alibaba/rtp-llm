#!/usr/bin/env bash
# Brm10: ring admission (16k window) + decode 8-rank fanout + block-aware
# FlexLB (jar flexlb-api-blockaware-895edc9f) + engine r4 (ring-pull
# concurrency gate = 1/rank + drain diagnostics). The gate is the fix for the
# CACHE_STORE_PUSH_ITEM_FAILED cascades seen in Brm8/Brm9.
# Same dataset/W48/instant-reject as B0m/Br0m/Brm8/Brm9/Am8.
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
export REMOTE_JAR_OVERRIDE=/home/admin/rtp-hol/flexlb/flexlb-api-blockaware-895edc9f.jar
export JAR_SHA256_OVERRIDE=895edc9f222a81d02d3612ec323a703f3c2b6a592a8527d32d17d8ba5ba6d70e
export FLEXLB_DECODE_OFFLOAD_RESIDENT_TOKENS=24576
export FLEXLB_DECODE_OFFLOAD_MIN_SEQ=16384
unset RTP_DECODE_MALLOC_RETRY_MS
export RTP_WORKER_EXTRA_ENV="${RTP_WORKER_EXTRA_ENV},V32_OFFLOAD_MODE=capacity,RTP_KV_OFFLOAD_KEEP_BLOCKS=256,RTP_KV_OFFLOAD_MIN_SEQ=16384,RTP_KV_OFFLOAD_AFTER_TOKENS=16,RTP_KV_OFFLOAD_STAGING_BLOCKS=32,V32_IDX_POOL_BLOCKS=8192,RTP_KV_ADMIT_RING_BLOCKS=64,RTP_KV_ADMIT_RING_CONCURRENCY=1"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
log "=== v32 Brm10 start: ring + fanout8 + block-aware LB + ring gate(r4), ds=mixed w48 noretry"
RUN_ID="v32brm10-$STAMP-schemeBrm10-ring-gate-w48-mixed" SCHEME=LOAD_ONLY WORKERS=48 RATIO="3:1" \
  "$BASE/run_v6_cell.sh" >>"$HERE/v32brm10-$STAMP.log" 2>&1
log "=== v32 Brm10 end rc=$?"

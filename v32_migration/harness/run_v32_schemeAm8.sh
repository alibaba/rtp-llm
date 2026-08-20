#!/usr/bin/env bash
# Am8: scheme A (V32_OFFLOAD_MODE=off) + decode 8-rank fanout — the aligned A
# baseline for Brm8/Brm9 (same dataset/W48/instant-reject/routing).
set -uo pipefail
BASE=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-06_rtp-llm-decode-routing-compare
HERE=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-12_dsv32-longctx-p1
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }
source "$HERE/v32_env.sh"

export RUNTIME_ENV="$HERE/b-runtime.env"
export FORMAL_REQUESTS=1000
export RTP_DECODE_CONCURRENCY=128
export RTP_EXPECTED_GPU_TOTAL=32
export RTP_DECODE_RANK_FANOUT=1
unset RTP_DECODE_MALLOC_RETRY_MS
export RTP_WORKER_EXTRA_ENV="${RTP_WORKER_EXTRA_ENV},V32_OFFLOAD_MODE=off"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
log "=== v32 Am8 start: mode=off + 8-rank fanout, ds=mixed w48 noretry"
RUN_ID="v32am8-$STAMP-schemeAm8-off-fanout8-w48-mixed" SCHEME=LOAD_ONLY WORKERS=48 RATIO="3:1" \
  "$BASE/run_v6_cell.sh" >>"$HERE/v32am8-$STAMP.log" 2>&1
log "=== v32 Am8 end rc=$?"

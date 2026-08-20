#!/usr/bin/env bash
# Brm11r: B full stack (ring admission 16k window + 8-rank fanout + block-aware
# FlexLB + ring gate + device fix) over RDMA cache store (r8: MR registration
# fixed). Aligned with Am8r.
set -uo pipefail
BASE=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-06_rtp-llm-decode-routing-compare
HERE=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-12_dsv32-longctx-p1
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }
source "$HERE/v32_env.sh"
unset RTP_CACHE_TRANSPORT_TCP   # RDMA cache store (MR fix in r8)

export RUNTIME_ENV="$HERE/br-runtime.env"
export FORMAL_REQUESTS=1000
export RTP_DECODE_CONCURRENCY=128
export RTP_EXPECTED_GPU_TOTAL=32
export RTP_DECODE_RANK_FANOUT=1
export REMOTE_JAR_OVERRIDE=/home/admin/rtp-hol/flexlb/flexlb-api-blockaware2-173c1a8f.jar
export JAR_SHA256_OVERRIDE=173c1a8f3d2cd76d0bb3686f7add7d1324f389956698b123f9e563299979d894
export FLEXLB_DECODE_OFFLOAD_RESIDENT_TOKENS=24576
export FLEXLB_DECODE_OFFLOAD_MIN_SEQ=16384
unset RTP_DECODE_MALLOC_RETRY_MS
export RTP_WORKER_EXTRA_ENV="${RTP_WORKER_EXTRA_ENV},V32_OFFLOAD_MODE=capacity,RTP_KV_OFFLOAD_KEEP_BLOCKS=256,RTP_KV_OFFLOAD_MIN_SEQ=16384,RTP_KV_OFFLOAD_AFTER_TOKENS=16,RTP_KV_OFFLOAD_STAGING_BLOCKS=32,V32_IDX_POOL_BLOCKS=8192,RTP_KV_ADMIT_RING_BLOCKS=64,RTP_KV_ADMIT_RING_CONCURRENCY=1"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
log "=== v32 Brm11r start: B full stack + RDMA, ds=mixed w48 noretry"
RUN_ID="v32brm11r-$STAMP-schemeBrm11r-ring-rdma-w48-mixed" SCHEME=LOAD_ONLY WORKERS=48 RATIO="3:1" \
  "$BASE/run_v6_cell.sh" >>"$HERE/v32brm11r-$STAMP.log" 2>&1
log "=== v32 Brm11r end rc=$?"

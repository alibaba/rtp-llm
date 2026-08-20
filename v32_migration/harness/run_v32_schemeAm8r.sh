#!/usr/bin/env bash
# Am8r: scheme A (offload off) + 8-rank fanout + RDMA cache store (r8 runtime:
# KV pool cudaMalloc backing + SingleType allocator forwarding = MR
# registration fixed). Aligned with Brm11r.
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
# blockaware2 jar with offload envs UNSET == stock LB semantics (the demand
# filter is opt-in); needed because LOCAL_JAR was rebuilt.
export REMOTE_JAR_OVERRIDE=/home/admin/rtp-hol/flexlb/flexlb-api-blockaware2-173c1a8f.jar
export JAR_SHA256_OVERRIDE=173c1a8f3d2cd76d0bb3686f7add7d1324f389956698b123f9e563299979d894
unset RTP_DECODE_MALLOC_RETRY_MS
export RTP_WORKER_EXTRA_ENV="${RTP_WORKER_EXTRA_ENV},V32_OFFLOAD_MODE=off"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
log "=== v32 Am8r start: mode=off + fanout8 + RDMA, ds=mixed w48 noretry"
RUN_ID="v32am8r-$STAMP-schemeAm8r-off-fanout8-rdma-w48-mixed" SCHEME=LOAD_ONLY WORKERS=48 RATIO="3:1" \
  "$BASE/run_v6_cell.sh" >>"$HERE/v32am8r-$STAMP.log" 2>&1
log "=== v32 Am8r end rc=$?"

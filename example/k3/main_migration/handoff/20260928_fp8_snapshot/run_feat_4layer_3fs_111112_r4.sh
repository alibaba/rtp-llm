#!/usr/bin/env bash
set -euo pipefail
umask 077

task=/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927
export PREFILL_SSH_TARGET=L20-dev-111
export DECODE_SSH_TARGET=L20-dev-112
export PREFILL_REPO_ROOT=/data6/luohaocheng.lhc/worktrees/rtp-llm-k3-feat-55641e09-20260928
export DECODE_REPO_ROOT=/data1/luohaocheng.lhc/worktrees/rtp-llm-k3-feat-55641e09-20260926
export PREFILL_CHECKPOINT_PATH=/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers
export DECODE_CHECKPOINT_PATH=/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers
export PREFILL_SP_CHECKPOINT_PATH=/data6/luohaocheng.lhc/models/kimi-k3-mtp-3fs-view-20260927
export DECODE_SP_CHECKPOINT_PATH=/data1/luohaocheng.lhc/models/kimi-k3-mtp-3fs-view-20260927
export PREFILL_ENDPOINT=11.163.39.111:26300
export DECODE_ENDPOINT=11.163.39.112:26400
export SMOKE_RUN_ID=feat-55641-4layer-3fs-111112-r4-20260928
export SMOKE_SUITE=flow
export PREFILL_SMOKE_CONTAINER=lhc_GPU_k3_3fs_20260928
export DECODE_SMOKE_CONTAINER=lhc_GPU_k3_3fs_20260927
export SMOKE_CONTAINER_USER=luohaocheng.lhc
export PREFILL_SMOKE_ARTIFACT_ROOT=/data6/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/feat-flow-111112-r4
export DECODE_SMOKE_ARTIFACT_ROOT=/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/feat-flow-111112-r4
export SMOKE_ALLOW_HF3FS_ROOT=/mnt/hf3fs/3fs/models/kimi
export PREFILL_SMOKE_CHECKPOINT_GUARD=/data6/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/weight_loader_guard.py
export DECODE_SMOKE_CHECKPOINT_GUARD=/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/weight_loader_guard.py
export PREFILL_SMOKE_ROLE_SCRIPT=/data6/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/feat_role_3fs_parallel_111112.sh
export DECODE_SMOKE_ROLE_SCRIPT=/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/feat_role_3fs_parallel_111112.sh
export PREFILL_RTP_LLM_SERVER_BINARY="$PREFILL_REPO_ROOT/bazel-bin/rtp_llm/rtp_llm_server"
export DECODE_RTP_LLM_SERVER_BINARY="$DECODE_REPO_ROOT/bazel-bin/rtp_llm/rtp_llm_server"
export RTP_LLM_SKIP_BUILD=1
export LOAD_METHOD=fastsafetensors
export FASTSAFETENSORS_NOGDS=0
export FP8_GEMM=1
export FP8_KV_CACHE=1
export FP8_MLA=1
export SP_ACT_TYPE=BF16
export SMOKE_PREFILL_TP_SIZE=8
export SMOKE_DECODE_TP_SIZE=8
export SMOKE_DECODE_DP_SIZE=1
export SMOKE_EXPECTED_LAYERS=4
export SMOKE_BLOCK_SIZE=4096
export SMOKE_KERNEL_BLOCK_SIZE=128
export SMOKE_CHUNK_TOKENS=65536
export SMOKE_MAX_TOKENS=32
export SMOKE_KEEP_CLUSTER_ON_SUCCESS=1
export SMOKE_KEEP_SERVICES=1
export SMOKE_RDMA_PREWARM_ATTEMPTS=0
export SMOKE_REQUEST_TIMEOUT_S=300
export SMOKE_STARTUP_TIMEOUT_S=3600
export SMOKE_CONTROLLER_TIMEOUT_S=7200
export FT_CORE_DUMP_ON_EXCEPTION=0

exec python3 "$task/feat_smoke_driver_3fs.py" --parallel-start --remote-detached \
  --remote-control-root /tmp/k3feat-r4-control \
  --artifact-root "$task/feat-flow-111112-r4-controller" "$@"

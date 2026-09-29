#!/usr/bin/env bash
set -euo pipefail
umask 077

task=/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PREFILL_SSH_TARGET=L20-dev-110
export DECODE_SSH_TARGET=L20-dev-112
export PREFILL_REPO_ROOT=/data7/luohaocheng.lhc/worktrees/rtp-llm-k3-feat-a9bf-nocp-r9
export DECODE_REPO_ROOT=/data1/luohaocheng.lhc/worktrees/rtp-llm-k3-feat-profile-8587b31-20260928
export PREFILL_CHECKPOINT_PATH=/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers
export DECODE_CHECKPOINT_PATH=/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers
export PREFILL_SP_CHECKPOINT_PATH=/data7/luohaocheng.lhc/models/kimi-k3-mtp-3fs-view-20260927
export DECODE_SP_CHECKPOINT_PATH=/data1/luohaocheng.lhc/models/kimi-k3-mtp-3fs-view-20260927
export PREFILL_ENDPOINT=11.163.39.110:26500
export DECODE_ENDPOINT=11.163.39.112:26600
export SMOKE_RUN_ID=feat-nocp-a9bf-4layer-3fs-110112-r9-20260929
export SMOKE_SUITE=flow
export PREFILL_SMOKE_CONTAINER=lhc_GPU_k3_3fs_20260927
export DECODE_SMOKE_CONTAINER=lhc_GPU_k3_3fs_20260927
export SMOKE_CONTAINER_USER=luohaocheng.lhc
export PREFILL_SMOKE_ARTIFACT_ROOT=/data7/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/feat-flow-110112-nocp-r9
export DECODE_SMOKE_ARTIFACT_ROOT=/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/feat-flow-110112-nocp-r9
export SMOKE_ALLOW_HF3FS_ROOT=/mnt/hf3fs/3fs/models/kimi
export PREFILL_SMOKE_CHECKPOINT_GUARD=/data7/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/weight_loader_guard.py
export DECODE_SMOKE_CHECKPOINT_GUARD=/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/weight_loader_guard.py
export PREFILL_SMOKE_ROLE_SCRIPT=/data7/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/feat_role_3fs_nocp_r9.sh
export DECODE_SMOKE_ROLE_SCRIPT=/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/feat_role_3fs_nocp_r9.sh
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
export SMOKE_MAX_TOKENS=16
export SMOKE_IDENTITY_MAX_TOKENS=16
export SMOKE_SINGLE_EXACT_MAX_TOKENS=16
export SMOKE_MTP_CHUNK_MAX_TOKENS=16
export RTP_LLM_PROFILE_MODEL_MODULES=1
export SMOKE_KEEP_CLUSTER_ON_SUCCESS=1
export SMOKE_KEEP_SERVICES=1
export SMOKE_RDMA_PREWARM_ATTEMPTS=0
export SMOKE_REQUEST_TIMEOUT_S=300
export SMOKE_STARTUP_TIMEOUT_S=3600
export SMOKE_CONTROLLER_TIMEOUT_S=7200
export FT_CORE_DUMP_ON_EXCEPTION=0

expected_sha=a9bf762e878fc54ee9176da5c34ffbe6babc8d45
for role in PREFILL DECODE; do
  ssh_target="${role}_SSH_TARGET"
  repo_root="${role}_REPO_ROOT"
  server_binary="${role}_RTP_LLM_SERVER_BINARY"
  test "$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${!ssh_target}" \
    "git -C '${!repo_root}' rev-parse HEAD")" = "$expected_sha"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${!ssh_target}" \
    "test -x '${!server_binary}'"
done

exec python3 "$script_dir/feat_smoke_driver_3fs.py" --parallel-start --remote-detached \
  --remote-control-root /tmp/k3feat-nocp-r9-control \
  --artifact-root "$task/feat-flow-110112-nocp-r9-controller" "$@"

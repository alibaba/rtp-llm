#!/usr/bin/env bash
set -euo pipefail
umask 077

host_id=${1:-}
mode=${2:-}
case "$host_id" in
  110) role=PREFILL; peer_ip=11.163.39.115; start_port=27100; peer_port=27200 ;;
  115) role=DECODE; peer_ip=11.163.39.110; start_port=27200; peer_port=27100 ;;
  *) echo 'usage: run_fp8_local_integrated_fourlayer_110115_r11_20260929.sh 110|115 config|run' >&2; exit 2 ;;
esac

if [[ "$host_id" == 110 ]]; then
  base=/data7/luohaocheng.lhc
  repo="$base/worktrees/rtp-llm-k3-integration-localms-r10-20260929"
  expected_fs=ext4
  nvcc="$base/artifacts/k3-fp8-main-20260926/nvcc-gcc13.sh"
else
  base=/data0/luohaocheng.lhc
  repo="$base/worktrees/rtp-llm-k3-fp8-kmerge-20260929"
  expected_fs=xfs
  nvcc="$base/artifacts/k3-fp8-opt-20260927/nvcc-gcc13-data0-20260929.sh"
fi
deps="$base/artifacts/k3-fp8-main-20260926"
task="$base/artifacts/k3-fp8-opt-20260927"
run="$base/k3integrated-${host_id}-fp8-4l-3fs-20260929-perf-r17"
checkpoint="$base/models/kimi-k3-4layers-ms-3fs-view-20260929"
draft="$base/models/kimi-k3-mtp-3fs-view-20260927"
server="$repo/bazel-bin/rtp_llm/rtp_llm_server"
guard="$task/weight_loader_guard.py"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
export HOME=/home/luohaocheng.lhc
export TMPDIR="$task/tmp-integrated-110115-$host_id"
export DG_JIT_CACHE_DIR="$task/jit-cache-integrated-110115-$host_id/deepgemm"
mkdir -p "$TMPDIR" "$DG_JIT_CACHE_DIR"
test -w "$HOME"
test "$(findmnt -T "$TMPDIR" -n -o FSTYPE)" = "$expected_fs"
test -x "$nvcc"
test -d "$deps/cutlass-runtime-452"
test "$(ulimit -l)" = unlimited
expected_head=97b492cc62319eab283cc2802a724dd704137524
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test "$(findmnt -T "$repo" -n -o FSTYPE)" = "$expected_fs"
test "$(findmnt -T "$checkpoint" -n -o FSTYPE | sort -u)" = "$expected_fs"
test "$(findmnt -T "$draft" -n -o FSTYPE | sort -u)" = "$expected_fs"
test -x "$server"
test -f "$deps/native-deepgemm-gcc13-elfutils/k3_native_deep_gemm/_C.cpython-310-x86_64-linux-gnu.so"
test -f "$deps/vllm-stable-lib/_C_stable_libtorch.abi3.so"
test -f "$guard"
test -d "$draft"

unset RTP_LLM_DEVICE_INPUT RTP_LLM_DEVICE_INPUT_CHECK
export RTP_LLM_JIT_CACHE_ROOT="$task/jit-cache-integrated-110115-$host_id"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FASTSAFETENSORS_DEBUG=true
export DG_JIT_CPP_STANDARD=20
export DG_JIT_NVCC_COMPILER="$nvcc"
export DG_JIT_USE_NVRTC=0
export KIMI_K3_CUTLASS_DSL_ROOT="$deps/cutlass-runtime-452"
export KIMI_K3_FLASHINFER_ROOT="$deps/native-runtime-py310"
export KIMI_K3_VLLM_STABLE_LIBRARY="$deps/vllm-stable-lib/_C_stable_libtorch.abi3.so"
export KIMI_K3_KDA_PREFILL_BACKEND=cula
export TOKENSPEED_MLA_PREFILL_BACKEND=cutedsl
export KIMI_K3_MOE_BACKEND=vllm_native
export RTP_LLM_PROFILE_MODEL_MODULES=1
export CUTE_DSL_CACHE_DIR="$RTP_LLM_JIT_CACHE_ROOT/cute"
export HF_MODULES_CACHE="$RTP_LLM_JIT_CACHE_ROOT/hf"
export K3_3FS_PREAD_THREADS=64
if [[ "$host_id" == 110 ]]; then
  pread_lib="$task/libparallel_3fs_pread.so"
else
  pread_lib="$task/libparallel_3fs_pread_20260929.so"
fi
test "$(sha256sum "$pread_lib" | cut -d' ' -f1)" = 99ae7c24fbda0e681e6876833e4cdb1f5cf0f3f1c6c0c3db8332e1e1a8435771
export LD_PRELOAD="$pread_lib:$deps/gcc13/lib/gcc/x86_64-conda-linux-gnu/13.4.0/libstdc++.so.6.0.32"
export PYTHONPATH="$repo:$deps/native-deepgemm-gcc13-elfutils:$deps/native-runtime-py310${PYTHONPATH:+:$PYTHONPATH}"
unset CUBLAS_WORKSPACE_CONFIG CP_ROTATE_METHOD QUANTIZATION SP_QUANTIZATION
mkdir -p "$CUTE_DSL_CACHE_DIR" "$HF_MODULES_CACHE"
cd "$repo"

args=(
  --role "$role" --checkpoint "$checkpoint" --draft-checkpoint "$draft"
  --peer-ip "$peer_ip" --start-port "$start_port" --peer-port "$peer_port"
  --server "$server" --guard "$guard"
  --allow-hf3fs-root /mnt/hf3fs/3fs/models/kimi
  --run-dir "$run" --min-free-gib 40 --reserve-runtime-mem-mb 14336
  --rdma-hcas mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
  --fp8-gemm --fp8-kv-cache --debug-four-layer
)
case "$mode" in
  config) python3 example/k3/main_migration/launch_bf16.py "${args[@]}" --print-config ;;
  run)
    test ! -e "$run" && test ! -e "$run.server-stdio.log"
    set +e
    python3 example/k3/main_migration/launch_bf16.py "${args[@]}" > "$run.server-stdio.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run.exit"
    exit "$status"
    ;;
  *) echo 'mode must be config or run' >&2; exit 2 ;;
esac

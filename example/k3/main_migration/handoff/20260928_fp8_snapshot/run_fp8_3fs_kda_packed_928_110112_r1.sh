#!/usr/bin/env bash
set -euo pipefail
umask 077

host_id=${1:-}
mode=${2:-}
case "$host_id" in
  110)
    base=/data7/luohaocheng.lhc
    role=PREFILL
    peer_ip=11.163.39.112
    start_port=27100
    peer_port=27200
    ;;
  112)
    base=/data1/luohaocheng.lhc
    role=DECODE
    peer_ip=11.163.39.110
    start_port=27200
    peer_port=27100
    ;;
  *) echo 'usage: run_fp8_3fs_kda_packed_928_110112_r1.sh 110|112 config|run' >&2; exit 2 ;;
esac

repo="$base/worktrees/rtp-llm-k3-fp8-opt-kda-packed-928-20260928"
deps="$base/artifacts/k3-fp8-main-20260926"
task="$base/artifacts/k3-fp8-opt-20260927"
run="$base/k3${role,,}-3fs-${host_id}-kda928-r1"
checkpoint=/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers
draft="$base/models/kimi-k3-mtp-3fs-view-20260927"
server="$repo/bazel-bin/rtp_llm/rtp_llm_server"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 928ad6a6d23e1205113334efaeeb7e6e18f9f1b1
test "$(findmnt -T "$checkpoint" -n -o FSTYPE | sort -u | head -1)" = fuse.hf3fs
test -x "$server"
test -f "$deps/native-deepgemm-gcc13-elfutils/k3_native_deep_gemm/_C.cpython-310-x86_64-linux-gnu.so"
test -f "$deps/vllm-stable-lib/_C_stable_libtorch.abi3.so"
test -f "$task/libparallel_3fs_pread.so"
test -f "$task/weight_loader_guard.py"
test -f "$task/launch_bf16_3fs_shm_diagnostic.py"
test -d "$draft"

unset RTP_LLM_DEVICE_INPUT RTP_LLM_DEVICE_INPUT_CHECK
export RTP_LLM_JIT_CACHE_ROOT="$task/jit-cache-kda928-r1-$host_id"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FASTSAFETENSORS_DEBUG=true
export DG_JIT_CPP_STANDARD=20
export DG_JIT_NVCC_COMPILER="$deps/nvcc-gcc13.sh"
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
export LD_PRELOAD="$task/libparallel_3fs_pread.so:$deps/gcc13/lib/gcc/x86_64-conda-linux-gnu/13.4.0/libstdc++.so.6.0.32"
export PYTHONPATH="$repo:$deps/native-deepgemm-gcc13-elfutils:$deps/native-runtime-py310${PYTHONPATH:+:$PYTHONPATH}"
unset CUBLAS_WORKSPACE_CONFIG CP_ROTATE_METHOD QUANTIZATION SP_QUANTIZATION
mkdir -p "$CUTE_DSL_CACHE_DIR" "$HF_MODULES_CACHE"
cd "$repo"

args=(
  --role "$role" --checkpoint "$checkpoint" --draft-checkpoint "$draft"
  --peer-ip "$peer_ip" --start-port "$start_port" --peer-port "$peer_port"
  --server "$server" --guard "$task/weight_loader_guard.py"
  --allow-hf3fs-root /mnt/hf3fs/3fs/models/kimi
  --run-dir "$run" --min-free-gib 250 --reserve-runtime-mem-mb 14336
  --rdma-hcas mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
  --fp8-gemm --fp8-kv-cache --debug-four-layer
)
case "$mode" in
  config) python3 "$task/launch_bf16_3fs_shm_diagnostic.py" "${args[@]}" --print-config ;;
  run)
    test ! -e "$run" && test ! -e "$run.server-stdio.log"
    set +e
    python3 "$task/launch_bf16_3fs_shm_diagnostic.py" "${args[@]}" > "$run.server-stdio.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run.exit"
    exit "$status"
    ;;
  *) echo 'mode must be config or run' >&2; exit 2 ;;
esac

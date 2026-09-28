#!/usr/bin/env bash
set -euo pipefail

base=/data1/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-fp8-opt-kda-packed-928-20260928"
deps="$base/artifacts/k3-fp8-main-20260926"
case_dir="$base/artifacts/k3-fp8-opt-20260927/kda-direct-output-tdd"
runfiles="$repo/bazel-bin/rtp_llm/rtp_llm_server.runfiles"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 928ad6a6d23e1205113334efaeeb7e6e18f9f1b1
test -f "$case_dir/baseline.py"
test -f "$case_dir/native_kda.py"
test -f "$case_dir/compare_native_kda_packed_gpu.py"
test -d "$runfiles/pip_gpu_cuda13_torch_cuda_linear_attention/site-packages/cula"

printf -v PYTHONPATH '%s:' "$runfiles"/pip_*/site-packages
PYTHONPATH="$deps/cutlass-runtime-452/nvidia_cutlass_dsl/python_packages:$PYTHONPATH"
export PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD="$deps/gcc13/lib/gcc/x86_64-conda-linux-gnu/13.4.0/libstdc++.so.6.0.32"
export DG_JIT_CPP_STANDARD=20
export DG_JIT_NVCC_COMPILER="$deps/nvcc-gcc13.sh"
export DG_JIT_USE_NVRTC=0
export KIMI_K3_CUTLASS_DSL_ROOT="$deps/cutlass-runtime-452"
export CUTE_DSL_CACHE_DIR="$case_dir/cute-cache"
mkdir -p "$CUTE_DSL_CACHE_DIR"

if [[ "${1:-}" == --check-import ]]; then
  python3 -c 'from cula.kda import chunk_kda; print(chunk_kda.__module__)'
  exit
fi

python3 "$case_dir/compare_native_kda_packed_gpu.py" \
  --baseline "$case_dir/baseline.py" \
  --candidate "$case_dir/native_kda.py" \
  --output "$case_dir/numerical-near-64k.json" \
  --pages 17 --heads 12 --block-size 4096

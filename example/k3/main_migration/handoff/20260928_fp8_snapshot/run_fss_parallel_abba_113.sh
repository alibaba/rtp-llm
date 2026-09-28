#!/usr/bin/env bash
set -euo pipefail
base=/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927
pkg=/data0/luohaocheng.lhc/.cache/bazel/k3-feat-55641e09/413e1568455a62f14d2ccfa3958fe07b/external/pip_gpu_cuda13_torch_fast_safetensors/site-packages
test "$(id -un)" = luohaocheng.lhc
for trial in baseline_a parallel_a parallel_b baseline_b; do
  log="$base/fss-parallel-abba-$trial.log"
  test ! -e "$log"
  preload=
  if [[ "$trial" == parallel* ]]; then
    preload="$base/libparallel_3fs_pread.so"
  fi
  docker exec -u luohaocheng.lhc \
    -e CUDA_VISIBLE_DEVICES=7 \
    -e PYTHONPATH="$pkg" \
    -e LD_PRELOAD="$preload" \
    -e K3_3FS_PREAD_THREADS=64 \
    lhc_GPU python "$base/bench_fast_safetensors_device_path_113.py" 1 \
    > "$log" 2>&1
  echo "$trial $(rg -o '"seconds": [0-9.]+' "$log" | head -1)"
done

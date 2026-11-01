#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cuda_root="${CUDA_ROOT:-/usr/local/cuda}"
nvcc_bin="${NVCC:-${cuda_root}/bin/nvcc}"
if [[ ! -x "$nvcc_bin" ]]; then
  echo "CUDA 13 nvcc not found at $nvcc_bin; set CUDA_ROOT or NVCC" >&2
  exit 1
fi
# Native code for GB200/GB300, plus compute_100 PTX. No architecture-specific 'a' instructions.
"$nvcc_bin" -std=c++17 -O2 -lineinfo \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_103,code=sm_103 \
  -gencode arch=compute_100,code=compute_100 \
  -Xcompiler=-Wall,-Wextra egm_fabric_probe.cu -o egm_fabric_probe -lcuda
echo "Built $(pwd)/egm_fabric_probe"

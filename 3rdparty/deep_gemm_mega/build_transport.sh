#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output_dir=${1:?package library output directory required}
mkdir -p "${output_dir}"
for transport in sparse signal; do
    source_name=ppu_sparse_gin.cu
    if [[ "${transport}" == signal ]]; then
        source_name=ppu_sparse_signal_gin.cu
    fi
    /usr/local/cuda/bin/nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -arch=sm_89 \
        -I/usr/local/PPU_SDK/pccl/include \
        -I/usr/local/PPU_SDK/targets/x86_64-linux/include \
        "${root_dir}/csrc/transport/${source_name}" \
        -L/usr/local/PPU_SDK/pccl/lib -lpccl \
        -o "${output_dir}/lib${transport}_gin.so"
done

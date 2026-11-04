#!/bin/bash
# Build librail_routed2.so next to this script (or into $1). Needs CUDA >= 12.8 with sm_120 and rdma-core userspace
# verbs headers + libibverbs (the system one, or a private build: RDMA_CORE=<prefix with include/ and lib/>).
# The rename is atomic, so it is safe to rebuild while a process still has the old .so loaded.
# usage: [CUDA_HOME=/usr/local/cuda] [RDMA_CORE=/opt/rdma-core] [ARCH=sm_120] bash build.sh [out_dir]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE}
CUDA=${CUDA_HOME:-/usr/local/cuda}
ARCH=${ARCH:-sm_120}
INC=() LIB=()
if [ -n "${RDMA_CORE:-}" ]; then
  INC=(-I"$RDMA_CORE/include")
  LIB=(-L"$RDMA_CORE/lib" -Xlinker -rpath,"$RDMA_CORE/lib")
fi
mkdir -p "$OUT"
"$CUDA/bin/nvcc" -O3 -std=c++17 -arch="$ARCH" -shared -Xcompiler -fPIC -lineinfo "${INC[@]}" \
  "$HERE/librail_routed2.cu" -o "$OUT/librail_routed2.so.tmp.$$" "${LIB[@]}" -l:libibverbs.so.1 \
  -L"$CUDA/lib64" -Xlinker -rpath,"$CUDA/lib64" -lcuda -lcudart -lpthread
mv -f "$OUT/librail_routed2.so.tmp.$$" "$OUT/librail_routed2.so"
ls -la "$OUT/librail_routed2.so"

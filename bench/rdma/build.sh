#!/bin/bash
# Build the RDMA / KV-transfer benchmarks in this directory into ./bin.
# Env: RDMA_INC / RDMA_LIB (rdma-core include and lib dirs with the device provider, e.g. a private rdma-core
#      build on eRDMA hosts; default: system), CUDA_HOME (default /usr/local/cuda; only vram_lat needs CUDA
#      headers, the kv_* / copy_shape tools load the CUDA driver with dlopen and need no CUDA at build time).
set -eu
cd "$(dirname "$0")"
mkdir -p bin
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
VI=${RDMA_INC:+-I$RDMA_INC}
VL=${RDMA_LIB:+-L$RDMA_LIB -Wl,-rpath,$RDMA_LIB}
gcc -O2 -pthread -o bin/tcp_probe tcp_probe.c
gcc -O2 -pthread -o bin/kvshape_xport kvshape_xport.c
gcc -O2 -pthread -o bin/kv_pipeline kv_pipeline.c -ldl
gcc -O2 -pthread -o bin/kv_pipeline_pool kv_pipeline_pool.c -ldl
gcc -O2 -pthread -o bin/kv_layerwise_xport kv_layerwise_xport.c -ldl
gcc -O2 -o bin/copy_shape_bench copy_shape_bench.c -ldl
gcc -O2 $VI -o bin/ccprobe ccprobe.c $VL -libverbs
if [ -f "$CUDA_HOME/include/cuda.h" ]; then
  g++ -std=c++17 -O2 -I"$CUDA_HOME/include" $VI -o bin/vram_lat vram_lat.cc -L"$CUDA_HOME/lib64" \
      -L"$CUDA_HOME/lib64/stubs" $VL -lcudart -lcuda -l:libibverbs.so.1
else
  echo "skip vram_lat: no $CUDA_HOME/include/cuda.h"
fi
ls bin

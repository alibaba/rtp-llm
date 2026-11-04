#!/bin/bash
# 2x4 emulation on ONE 8-GPU host with two RDMA NICs: "node 0" = GPUs 0-3 on NIC0, "node 1" = GPUs 4-7 on NIC1, the
# cross-node legs go NIC0 <-> NIC1 through the switch (loopback). Good for correctness and relative perf; absolute
# numbers are lower than a real 2-host run (shorter wire, 4 GPUs per NIC).
# usage: [NIC0=erdma_0 NIC1=erdma_1] bash run_1node_emulated.sh TAG [routed2_test.py args...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=$1; shift
NIC0=${NIC0:-erdma_0}; NIC1=${NIC1:-erdma_1}
PY=${PYTHON:-python3}
CUDA=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=${RDMA_CORE:+$RDMA_CORE/lib:}$CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export RR2_LIB=${RR2_LIB:-$HERE/librail_routed2.so}
mkdir -p "$HERE/logs/$TAG"
for r in 0 1 2 3 4 5 6 7; do
  nic=$NIC0; [ $r -ge 4 ] && nic=$NIC1
  RANK=$r WORLD_SIZE=8 LOCAL_WORLD=4 GPU=$r NIC=$nic MASTER_ADDR=127.0.0.1 MASTER_PORT=${PORT:-29821} \
    RR2_PROXY_CPU=${PROXY_CPU:--1} \
    $PY -u "$HERE/tests/routed2_test.py" --tag "$TAG" "$@" > "$HERE/logs/$TAG/r$r.log" 2>&1 &
done
wait
grep -h SUMMARY "$HERE/logs/$TAG/r0.log"

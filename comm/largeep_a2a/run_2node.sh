#!/bin/bash
# Launch the 8 local ranks of one node of a 2-node x 8-GPU run (run it on BOTH hosts, node 0 first or concurrently).
#   NODE       0 on the master host, 1 on the other
#   MASTER     IP of node 0 (gloo control plane only: wire exchange + result gather)
#   NICS       per local GPU RDMA device, comma list (default erdma_0 for GPUs 0-3, erdma_1 for 4-7; one NIC per
#              PCIe switch side: all legs of one peer pair must use the same NIC)
#   PROXY_CPU  first core of the 8 proxy threads (proxy l on PROXY_CPU+l; -1 = unpinned). Pick idle cores on the
#              NUMA node of the NICs.
#   RDMA_CORE  private rdma-core prefix if the system libibverbs has no provider for the NIC
# usage: NODE=0|1 MASTER=<ip> bash run_2node.sh TAG [routed2_test.py args...]
# logs: logs/TAG/r<rank>.log ; rank 0 prints "SUMMARY {...}" when done.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=$1; shift
NODE=${NODE:?NODE=0|1}
MASTER=${MASTER:?MASTER=<node0 ip>}
PORT=${PORT:-29831}
LW=${LOCAL_WORLD:-8}
NICS=${NICS:-erdma_0,erdma_0,erdma_0,erdma_0,erdma_1,erdma_1,erdma_1,erdma_1}
PROXY_CPU=${PROXY_CPU:-8}
PY=${PYTHON:-python3}
CUDA=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=${RDMA_CORE:+$RDMA_CORE/lib:}$CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export RR2_LIB=${RR2_LIB:-$HERE/librail_routed2.so}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$HERE/logs/$TAG"
IFS=, read -r -a NIC_ARR <<< "$NICS"
for l in $(seq 0 $((LW - 1))); do
  r=$((NODE * LW + l))
  pc=-1; [ "$PROXY_CPU" -ge 0 ] && pc=$((PROXY_CPU + l))
  RANK=$r WORLD_SIZE=$((2 * LW)) LOCAL_WORLD=$LW GPU=$l NIC=${NIC_ARR[$l]} MASTER_ADDR=$MASTER MASTER_PORT=$PORT \
    GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0} RR2_PROXY_CPU=$pc \
    $PY -u "$HERE/tests/routed2_test.py" --tag "$TAG" "$@" > "$HERE/logs/$TAG/r$r.log" 2>&1 &
done
wait
[ "$NODE" = 0 ] && grep -h SUMMARY "$HERE/logs/$TAG/r0.log"

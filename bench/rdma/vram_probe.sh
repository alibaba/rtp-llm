#!/bin/bash
# GPU VRAM <-> VRAM RDMA WRITE latency sweep between two hosts (bin/vram_lat; peer-memory MR on cudaMalloc
# buffers, no host staging). Reported one-way time = round trip / 2 and includes an 8-byte D2H flag poll.
# Run "server" first, then "client <server-mgmt-ip>".
#   usage: vram_probe.sh server|client [server-ip]
# Env: DEV (default erdma_0), GID (1), GPU (local CUDA index, default 0), SIZES ("64 4096 16384 65536"),
#      ITERS (2000), PORT (first TCP port, default 19900; one port per size), RDMA_LIB (provider lib dir).
set -u
ROLE=${1:?server|client}
SIP=${2:-}
DEV=${DEV:-erdma_0}
GID=${GID:-1}
GPU=${GPU:-0}
SIZES=${SIZES:-64 4096 16384 65536}
ITERS=${ITERS:-2000}
P=${PORT:-19900}
BIN=$(cd "$(dirname "$0")" && pwd)/bin
[ -n "${RDMA_LIB:-}" ] && export LD_LIBRARY_PATH=$RDMA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ "$ROLE" = client ] && [ -z "$SIP" ] && { echo "client needs <server-ip>"; exit 2; }
for B in $SIZES; do
  if [ "$ROLE" = server ]; then
    timeout 60 "$BIN/vram_lat" server 0 $P "$DEV" "$GPU" "$GID" $B "$ITERS"
  else
    sleep 3
    echo -n "$DEV ${B}B: "
    timeout 60 "$BIN/vram_lat" client "$SIP" $P "$DEV" "$GPU" "$GID" $B "$ITERS"
  fi
  P=$((P + 1))
done

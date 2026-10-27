#!/bin/bash
# Point-to-point RDMA latency/bandwidth at expert-parallel dispatch/combine sizes (perftest).
# Run the server side first, then the client side on the other host.
#   usage: rdma_ep_probe.sh server
#          rdma_ep_probe.sh client <server-ip>     # IP reachable over TCP (perftest out-of-band exchange)
# Env: DEV (RDMA device, default mlx5_0), GID (gid index, default 1), RDMA_LIB (optional libibverbs dir),
#      PERFTEST (dir of ib_write_lat etc., default /usr/local/bin), OUT (log dir, default ./rdma_ep_probe).
# Measures RDMA WRITE latency 64B-64KiB, 8-byte atomic latency and WRITE bandwidth with 1 and 8 QPs.
set -u
ROLE=${1:?server|client}
SIP=${2:-}
DEV=${DEV:-mlx5_0}
GID=${GID:-1}
PT=${PERFTEST:-/usr/local/bin}
OUT=${OUT:-./rdma_ep_probe}
[ -n "${RDMA_LIB:-}" ] && export LD_LIBRARY_PATH=$RDMA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ "$ROLE" = client ] && [ -z "$SIP" ] && { echo "client needs <server-ip>"; exit 2; }
mkdir -p "$OUT"
run() {
  local name=$1; shift
  echo "=== $name"
  if [ "$ROLE" = server ]; then
    timeout 60 "$@"
  else
    sleep 4
    timeout 60 "$@" "$SIP"
  fi
  echo "rc=$?"
}
{
  echo "[$(date +%T)] role=$ROLE dev=$DEV gid=$GID"
  for s in 64 1024 4096 16384 65536; do
    run write_lat_$s "$PT/ib_write_lat" -d "$DEV" -x "$GID" -s $s -n 3000
  done
  run atomic_lat "$PT/ib_atomic_lat" -d "$DEV" -x "$GID" -n 3000
  for q in 1 8; do
    for s in 8192 65536; do
      run write_bw_q${q}_$s "$PT/ib_write_bw" -d "$DEV" -x "$GID" -s $s -q $q -D 3 --report_gbits
    done
  done
  echo "[$(date +%T)] ALL_DONE"
} > "$OUT/rdma_ep_probe.$ROLE.log" 2>&1
grep -E "^===|^ [0-9]+ +[0-9]+" "$OUT/rdma_ep_probe.$ROLE.log"

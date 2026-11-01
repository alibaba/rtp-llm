#!/bin/bash
# eRDMA / RDMA link probe between two hosts with perftest and rdma-core tools (host memory).
# Run "server" on one host first, then "client <server-mgmt-ip>" on the other within ~60 s. The client
# connects to the server's TCP-reachable (management) address for the out-of-band exchange, not its GID IP.
#   usage: link_probe.sh server|client [server-ip]
# Env: DEV (default erdma_0), GID (gid index, default 1 = IPv4 GID on eRDMA), RDMA_LIB (private rdma-core lib
#      dir with the erdma provider; eRDMA hosts need it, e.g. rdma-core build-58 on SM120 hosts), PERFTEST
#      (dir of ib_write_lat / ib_write_bw, default /usr/local/bin), RDMA_BIN (dir of ibv_rc_pingpong,
#      default $RDMA_LIB/../bin), QPS (bandwidth QP counts, default "1 8 16 32"), SECS (bw duration, 10),
#      SIZES (latency sizes, "64 4096 16384 65536"), OUT (log dir, ./link_probe), PORT (perftest port, 18515).
# Prints one line per test; full tool output goes to $OUT/<test>.<role>.log.
set -u
ROLE=${1:?server|client}
SIP=${2:-}
DEV=${DEV:-erdma_0}
GID=${GID:-1}
PT=${PERFTEST:-/usr/local/bin}
RB=${RDMA_BIN:-${RDMA_LIB:+$RDMA_LIB/../bin}}
QPS=${QPS:-1 8 16 32}
SECS=${SECS:-10}
SIZES=${SIZES:-64 4096 16384 65536}
OUT=${OUT:-./link_probe}
PORT=${PORT:-18515}
[ -n "${RDMA_LIB:-}" ] && export LD_LIBRARY_PATH=$RDMA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ "$ROLE" = client ] && [ -z "$SIP" ] && { echo "client needs <server-ip>"; exit 2; }
mkdir -p "$OUT"
run() {
  local name=$1; shift
  if [ "$ROLE" = server ]; then
    timeout $((SECS + 60)) "$@" > "$OUT/$name.server.log" 2>&1
  else
    sleep 4
    timeout $((SECS + 60)) "$@" "$SIP" > "$OUT/$name.client.log" 2>&1
    echo "$name: $(grep -E '^ *[0-9]+ +[0-9]+' "$OUT/$name.client.log" | tail -1 | tr -s ' ')"
  fi
}
echo "[$(date +%T)] role=$ROLE dev=$DEV gid=$GID $(hostname)"
if [ -n "$RB" ] && [ -x "$RB/ibv_rc_pingpong" ]; then
  if [ "$ROLE" = server ]; then
    timeout 60 "$RB/ibv_rc_pingpong" -d "$DEV" -g "$GID" -s 4096 -n 2000 -p $((PORT + 1)) > "$OUT/rc_pingpong_4k.server.log" 2>&1
  else
    sleep 4
    timeout 60 "$RB/ibv_rc_pingpong" -d "$DEV" -g "$GID" -s 4096 -n 2000 -p $((PORT + 1)) "$SIP" > "$OUT/rc_pingpong_4k.client.log" 2>&1
    echo "rc_pingpong_4k (round trip): $(grep -E 'usec/iter' "$OUT/rc_pingpong_4k.client.log" | tail -1)"
  fi
fi
for s in $SIZES; do
  run write_lat_$s "$PT/ib_write_lat" -d "$DEV" -x "$GID" -s $s -n 3000 -p $PORT
done
for q in $QPS; do
  run write_bw_q${q}_1m "$PT/ib_write_bw" -d "$DEV" -x "$GID" -s 1048576 -q $q -D $SECS --report_gbits -p $PORT
done
echo "[$(date +%T)] done"

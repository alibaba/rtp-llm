#!/usr/bin/env bash
# Drive the full A / B(prefetch=0) / B(prefetch=8) / C sweep on one fixed request.
#
# A needs its own server because disabling offload is an engine-level decision made
# at startup. B0, B8 and C share a single server and are switched at runtime through
# V32_MODE_FILE, which avoids three more 700GB weight loads and, more importantly,
# keeps them on the same instance so their differences are not confounded by
# per-process variation. The prompt is pinned to a file on first use and every
# later run reads the same bytes.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-26100}
CTX=${CTX:-63000}
OUT=${OUT:-128}
REPS=${REPS:-3}
WARMUP=${WARMUP:-1}
TP=${TP:-1}
DP=${DP:-8}
KVMB=${KVMB:-12288}
MAXSEQ=${MAXSEQ:-65536}
MODE_FILE=${MODE_FILE:-/home/admin/rtp-hol/logs/v32_mode}
PROMPT=${PROMPT:-/home/admin/rtp-hol/logs/ab_prompt.txt}
JSONL=${JSONL:-/home/admin/rtp-hol/logs/abc_bench.jsonl}
SUITE=${SUITE:-A B0 B8 C}

log() { printf '[%(%Y-%m-%d %H:%M:%S)T] === %s\n' -1 "$*"; }

stop_server() {
  # Workers rename themselves to rtp_llm_rank-N / rtp_llm_backend_server, so a
  # pattern matching only the launch command leaves them holding all 8 GPUs. The
  # brackets keep the pattern from matching this script's own command line.
  local pat='rtp_llm_rank[-]|rtp_llm_backend[_]server|rtp_llm[.]start_server'
  local pids
  pids=$(pgrep -f "$pat" | grep -v "^$$\$" | tr '\n' ' ')
  [[ -z "${pids// /}" ]] && return 0
  kill $pids 2>/dev/null
  for _ in $(seq 1 30); do
    pids=$(pgrep -f "$pat" | grep -v "^$$\$" | tr '\n' ' ')
    [[ -z "${pids// /}" ]] && return 0
    sleep 2
  done
  kill -9 $pids 2>/dev/null
  sleep 10
}

bench() {  # bench <tag>
  log "bench $1"
  /opt/conda310/bin/python "$HERE/bench_ab.py" --port "$PORT" --ctx "$CTX" \
    --out "$OUT" --reps "$REPS" --warmup "$WARMUP" --tag "$1" \
    --prompt-file "$PROMPT" --jsonl "$JSONL" || log "bench $1 FAILED"
}

start_server() {  # start_server <mode A|B>
  log "starting server MODE=$1 tp=$TP dp=$DP kv=${KVMB}MB"
  MODE=$1 TP=$TP DP=$DP KVMB=$KVMB MAXSEQ=$MAXSEQ PORT=$PORT \
    EXTRA_ENV="${EXTRA_ENV:-} V32_MODE_FILE=$MODE_FILE" \
    bash "$HERE/run_ab.sh" || { log "server start FAILED"; return 1; }
}

want() { [[ " $SUITE " == *" $1 "* ]]; }

mkdir -p "$(dirname "$MODE_FILE")"

if want A; then
  stop_server
  start_server A && bench A
fi

# One instance per scheme. A shared instance saved two weight loads but let
# allocator pressure accumulate across 63k requests: after eight of them the rank
# sat at 139.38 of 139.80 GiB and the next scheme's first request died on a 492 MiB
# allocation. The mode file still selects the scheme, so B0, B8 and C keep a
# byte-identical offload environment and differ only in the scheme itself.
for tag in B0 B8 C; do
  want "$tag" || continue
  stop_server
  echo "$tag" > "$MODE_FILE"
  start_server B || continue
  bench "$tag"
done

stop_server
log "sweep done; results in $JSONL"
/opt/conda310/bin/python "$HERE/summarize_abc.py" --jsonl "$JSONL" || true

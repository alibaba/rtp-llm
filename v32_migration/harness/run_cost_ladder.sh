#!/usr/bin/env bash
# End-to-end latency attribution ladder for the DSA offload path.
# Every rung gets a fresh server and the same pinned real-corpus prompt.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-26100}
REPS=${REPS:-4}
OUT=${OUT:-512}
JSONL=${JSONL:-/home/admin/rtp-hol/logs/cost_ladder.jsonl}
PROMPT=${PROMPT:-/home/admin/rtp-hol/logs/corpus_prompt.txt}
MODE_FILE=${MODE_FILE:-/home/admin/rtp-hol/logs/v32_mode}
CORPUS=/home/admin/workspace/aop_lab/wt-dsa-offload/rtp_llm/cpp

stop_server() {
  local pat='rtp_llm_rank[-]|rtp_llm_backend[_]server|rtp_llm[.]start_server'
  local pids
  pids=$(pgrep -f "$pat" | tr '\n' ' ')
  [[ -n "${pids// /}" ]] && kill $pids 2>/dev/null
  for _ in $(seq 1 20); do
    sleep 2
    pids=$(pgrep -f "$pat" | tr '\n' ' ')
    [[ -z "${pids// /}" ]] && { sleep 8; return; }
  done
  kill -9 $pids 2>/dev/null
  sleep 10
}

start() { # mode keep extra-env
  MODE=$1 KEEP=$2 TP=1 DP=8 KVMB=12288 MAXSEQ=65536 PORT=$PORT \
    EXTRA_ENV="V32_MODE_FILE=$MODE_FILE $3" \
    timeout 2400 bash "$HERE/run_ab.sh"
}

bench() { # tag
  /opt/conda310/bin/python "$HERE/bench_ab.py" --port "$PORT" --ctx 63000 \
    --out "$OUT" --reps "$REPS" --warmup 0 --corpus "$CORPUS" --tag "$1" \
    --prompt-file "$PROMPT" --jsonl "$JSONL"
}

run_rung() { # tag mode keep mode-file-value extra-env
  local tag=$1 mode=$2 keep=$3 file_mode=$4 extra=$5
  stop_server
  printf '%s\n' "$file_mode" > "$MODE_FILE"
  printf '[%(%F %T)T] START %-18s mode=%s keep=%s env=%s\n' -1 "$tag" "$mode" "$keep" "$extra"
  start "$mode" "$keep" "$extra" || { echo "start failed: $tag"; return; }
  bench "$tag" || echo "bench failed: $tag"
}

rm -f "$JSONL"
run_rung L0_A                A 0   C  "V32_HOOK_LEVEL=0"
run_rung L1_hook             B 0   C  "V32_HOOK_LEVEL=0"
run_rung L2_scoring_only     B 0   C  "V32_MIRROR_CHUNK=0 V32_SKIP_PROCESS=1"
run_rung L3_process_nomirror B 0   C  "V32_MIRROR_CHUNK=0"
run_rung L4_B0_nomirror      B 256 B0 "V32_MIRROR_CHUNK=0"
run_rung L5_B0               B 256 B0 ""
run_rung L6_C                B 256 C  ""
stop_server

/opt/conda310/bin/python - "$JSONL" <<'PY'
import json, sys
rows = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    rows[r['tag']] = r['tpot_mean']
base = rows.get('L0_A')
print('\n=== COST LADDER ===')
last = None
for tag in ('L0_A','L1_hook','L2_scoring_only','L3_process_nomirror',
            'L4_B0_nomirror','L5_B0','L6_C'):
    v = rows.get(tag)
    if v is None:
        continue
    dbase = v-base if base is not None else 0
    dlast = v-last if last is not None else 0
    print(f'{tag:<22} {v:8.3f} ms   vs A {dbase:+7.3f}   rung {dlast:+7.3f}')
    last = v
PY

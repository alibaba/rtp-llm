#!/usr/bin/env bash
# End-to-end verification for the independent DSA indexer-K pool (#31).
# New .so invalidates all prior measurements, so this re-baselines A on the
# same runtime before running C with V32_INDEPENDENT_IDX_POOL=1.
#   A1, A2      : baseline twice (within-A spread = noise floor)
#   C_idxpool   : scheme C + engine-owned indexer pool (shadow pool stands down)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-26100}
REPS=${REPS:-4}
OUT=${OUT:-512}
RUNTIME=${RUNTIME:-/home/admin/rtp-hol/runtime/rtp-idxpool-20260901}
JSONL=${JSONL:-/home/admin/rtp-hol/logs/idxpool_verify.jsonl}
PROMPT=${PROMPT:-/home/admin/rtp-hol/logs/corpus_prompt.txt}
MODE_FILE=${MODE_FILE:-/home/admin/rtp-hol/logs/v32_mode}
CORPUS=/home/admin/workspace/aop_lab/wt-dsa-offload/rtp_llm/cpp

stop_server() {
  local pat='rtp_llm_rank[-]|rtp_llm_backend[_]server|rtp_llm[.]start_server|rtp_llm_frontend[_]server'
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

start() { # mode extra-env
  RUNTIME=$RUNTIME MODE=$1 KEEP=256 TP=1 DP=8 KVMB=12288 MAXSEQ=65536 PORT=$PORT \
    EXTRA_ENV="V32_MODE_FILE=$MODE_FILE $2" \
    timeout 2400 bash "$HERE/run_ab.sh"
}

bench() { # tag
  /opt/conda310/bin/python "$HERE/bench_ab.py" --port "$PORT" --ctx 63000 \
    --out "$OUT" --reps "$REPS" --warmup 0 --corpus "$CORPUS" --tag "$1" \
    --prompt-file "$PROMPT" --jsonl "$JSONL"
}

run_rung() { # tag mode mode-file-value extra-env
  local tag=$1 mode=$2 file_mode=$3 extra=$4
  stop_server
  printf '%s\n' "$file_mode" > "$MODE_FILE"
  printf '[%(%F %T)T] START %-12s mode=%s env=%s\n' -1 "$tag" "$mode" "$extra"
  start "$mode" "$extra" || { echo "start failed: $tag"; return 1; }
  bench "$tag" || echo "bench failed: $tag"
}

rm -f "$JSONL"
run_rung A1        A C "V32_HOOK_LEVEL=0"
run_rung A2        A C "V32_HOOK_LEVEL=0"
run_rung C_idxpool C C "V32_INDEPENDENT_IDX_POOL=1"
stop_server

echo "=== NOISE-FLOOR VERDICT (A2-vs-A1 = floor; C passes at >=0.9x floor) ==="
/opt/conda310/bin/python "$HERE/compare_noise.py" --jsonl "$JSONL" \
  --base A1 --scheme A2 --scheme C_idxpool || true

/opt/conda310/bin/python - "$JSONL" <<'PY'
import json, sys
rows = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    rows.setdefault(r['tag'], []).append(r['tpot_mean'])
print('\n=== TPOT ===')
for tag, v in rows.items():
    print(f"{tag:<12} {' '.join(f'{x:.2f}' for x in v)}")
PY

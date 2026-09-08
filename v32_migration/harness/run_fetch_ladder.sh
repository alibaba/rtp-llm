#!/usr/bin/env bash
# Hold the mask implementation and generated length constant while separating:
#   D2 -> B0-like drop, no fetch (baseline attention workload)
#   D1 -> remap, no fetch (restored attention workload, no PCIe)
#   D0 -> remap + fetch (production C)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-26100}
REPS=${REPS:-4}
OUT=${OUT:-512}
JSONL=${JSONL:-/home/admin/rtp-hol/logs/fetch_ladder.jsonl}
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

run_mode() {
  local tag=$1 diag=$2
  stop_server
  printf 'C\n' > "$MODE_FILE"
  printf '[%(%F %T)T] START %s diag=%s\n' -1 "$tag" "$diag"
  MODE=B TP=1 DP=8 KVMB=12288 MAXSEQ=65536 PORT=$PORT \
    EXTRA_ENV="V32_MODE_FILE=$MODE_FILE V32_LOSSLESS_DIAG_MODE=$diag" \
    timeout 2400 bash "$HERE/run_ab.sh" || return
  /opt/conda310/bin/python "$HERE/bench_ab.py" --port "$PORT" --ctx 63000 \
    --out "$OUT" --force-length --reps "$REPS" --warmup 0 --corpus "$CORPUS" \
    --tag "$tag" --prompt-file "$PROMPT" --jsonl "$JSONL"
}

rm -f "$JSONL"
run_mode D2_drop_no_fetch 2
run_mode D1_remap_no_fetch 1
run_mode D0_production_C 0
stop_server

/opt/conda310/bin/python - "$JSONL" <<'PY'
import json, sys
r={}
for line in open(sys.argv[1]):
    x=json.loads(line); r[x['tag']]=x['tpot_mean']
for k in ('D2_drop_no_fetch','D1_remap_no_fetch','D0_production_C'):
    print(f'{k:<22} {r[k]:.3f}ms')
print(f"restored_attention = D1-D2 = {r['D1_remap_no_fetch']-r['D2_drop_no_fetch']:+.3f}ms")
print(f"pcie_fetch         = D0-D1 = {r['D0_production_C']-r['D1_remap_no_fetch']:+.3f}ms")
PY

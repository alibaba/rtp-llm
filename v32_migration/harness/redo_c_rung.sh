#!/usr/bin/env bash
set -uo pipefail
HERE=/home/admin/workspace/aop_lab/wt-dsa-offload/v32_migration/harness
pat='rtp_llm_rank[-]|rtp_llm_backend[_]server|rtp_llm[.]start_server|rtp_llm_frontend[_]server'
pids=$(pgrep -f "$pat" | tr '\n' ' ')
[[ -n "${pids// /}" ]] && kill $pids 2>/dev/null
for _ in $(seq 1 20); do sleep 2; pids=$(pgrep -f "$pat" | tr '\n' ' '); [[ -z "${pids// /}" ]] && { sleep 8; break; }; done
pids=$(pgrep -f "$pat" | tr '\n' ' '); [[ -n "${pids// /}" ]] && { kill -9 $pids 2>/dev/null; sleep 10; }
printf 'C\n' > /home/admin/rtp-hol/logs/v32_mode
RUNTIME=/home/admin/rtp-hol/runtime/rtp-idxpool-20260901 MODE=C KEEP=256 TP=1 DP=8 KVMB=12288 MAXSEQ=65536 PORT=26100 \
  EXTRA_ENV="V32_MODE_FILE=/home/admin/rtp-hol/logs/v32_mode V32_INDEPENDENT_IDX_POOL=1" \
  timeout 2400 bash "$HERE/run_ab.sh" || { echo "start failed: C_idxpool_redo"; exit 1; }
/opt/conda310/bin/python "$HERE/bench_ab.py" --port 26100 --ctx 63000 --out 512 --reps 4 --warmup 0 \
  --corpus /home/admin/workspace/aop_lab/wt-dsa-offload/rtp_llm/cpp --tag C_noshadow \
  --prompt-file /home/admin/rtp-hol/logs/corpus_prompt.txt --jsonl /home/admin/rtp-hol/logs/idxpool_verify.jsonl
echo "=== NOISE-FLOOR VERDICT ==="
/opt/conda310/bin/python "$HERE/compare_noise.py" --jsonl /home/admin/rtp-hol/logs/idxpool_verify.jsonl \
  --base A1 --scheme A2 --scheme C_idxpool --scheme C_noshadow

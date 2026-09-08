#!/usr/bin/env bash
set -uo pipefail
cd /home/admin/workspace/aop_lab/wt-dsa-offload/v32_migration/harness
HERE=/home/admin/workspace/aop_lab/wt-dsa-offload/v32_migration/harness
pat='rtp_llm_rank[-]|rtp_llm_backend[_]server|rtp_llm[.]start_server|rtp_llm_frontend[_]server'
pids=$(pgrep -f "$pat" | tr '\n' ' ')
[[ -n "${pids// /}" ]] && kill $pids 2>/dev/null
for _ in $(seq 1 25); do sleep 2; pids=$(pgrep -f "$pat" | tr '\n' ' '); [[ -z "${pids// /}" ]] && { sleep 10; break; }; done
pids=$(pgrep -f "$pat" | tr '\n' ' '); [[ -n "${pids// /}" ]] && { kill -9 $pids 2>/dev/null; sleep 12; }
RUNTIME=/home/admin/rtp-hol/runtime/rtp-master-d4d9bf18b MODE=A TP=1 DP=8 KVMB=12288 MAXSEQ=65536 PORT=26100 \
  timeout 2400 bash "$HERE/run_ab.sh" || { echo "start failed: A_master"; exit 1; }
/opt/conda310/bin/python "$HERE/bench_ab.py" --port 26100 --ctx 63000 --out 512 --reps 4 --warmup 0 \
  --corpus /home/admin/workspace/aop_lab/wt-dsa-offload/rtp_llm/cpp --tag A_master \
  --prompt-file /home/admin/rtp-hol/logs/corpus_prompt.txt --jsonl /home/admin/rtp-hol/logs/idxpool_verify.jsonl
echo "=== A_master vs A1/A2 ==="
/opt/conda310/bin/python "$HERE/compare_noise.py" --jsonl /home/admin/rtp-hol/logs/idxpool_verify.jsonl \
  --base A_master --scheme A1 --scheme A2
/opt/conda310/bin/python - /home/admin/rtp-hol/logs/idxpool_verify.jsonl <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f"{r['tag']:<12} TPOT={r['tpot_mean']:.2f}ms")
PY

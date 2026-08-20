#!/usr/bin/env bash
# Wait for a GPU with >=200GB free, then run the three-step Flash logits
# compare (baseline -> mega -> compare) on it. Poll every 5 minutes.
set -u
CKPT=${E2E_CKPT:?set E2E_CKPT to the checkpoint dir}
SCRIPT="$(dirname "$(readlink -f "$0")")/run_e2e_logits.py"
LOG=${E2E_WATCH_LOG:-/tmp/dsv4_logits_run.log}

echo "[watch] $(date) start" > "$LOG"
while true; do
    GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2 < 40000 {print $1; exit}')
    if [ -n "${GPU:-}" ]; then
        # Double-check after a short settle to avoid racing another job's startup.
        sleep 60
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
        if [ "$USED" -lt 40000 ]; then
            echo "[watch] $(date) claiming GPU $GPU (used=${USED}MiB)" >> "$LOG"
            break
        fi
    fi
    echo "[watch] $(date) no free GPU yet" >> "$LOG"
    sleep 300
done

cd "$(dirname "$SCRIPT")"
for MODE in baseline mega compare; do
    echo "[watch] $(date) mode=$MODE" >> "$LOG"
    E2E_CKPT=$CKPT E2E_GPU=$GPU python3 "$SCRIPT" "$MODE" >> "$LOG" 2>&1
    RC=$?
    echo "[watch] mode=$MODE rc=$RC" >> "$LOG"
    if [ $RC -ne 0 ] && [ "$MODE" != "compare" ]; then
        echo "[watch] aborting after $MODE failure" >> "$LOG"
        exit $RC
    fi
done
echo "[watch] $(date) all done" >> "$LOG"

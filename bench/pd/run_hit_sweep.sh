#!/bin/bash
# Usage: run_hit_sweep.sh <tag> <set>:<qps>[:<offset>] ...   (run on the prefill host, from this directory)
# Prefix-hit sweep under constant-rate open-loop load. For every level: restart P with $RESTART_P (a command that
# restarts the prefill instance and returns when it is healthy; the prefix cache must start empty), warm it up
# (4 single requests, then WDUR s at the measured rate and output length; not counted), then run pd_uniform_load.py:
# request i is sent at exactly t0 + i/qps for DUR seconds, MAXT output tokens (ignore_eos), pinned round-robin to the
# decode ranks in D_ADDRS, with pd_wire_sampler.py recording the KV bytes. The measured requests use prompts
# [offset, offset + qps*DUR); the warm-up uses the prompts right after them, so no measured prompt was seen before.
# The warm-up must reach the steady decode concurrency at the full output length: in TCP mode a P that goes from idle
# to a high request rate needs ~10-15 s before TTFT settles (RDMA does not), so a short warm-up with short outputs
# leaves that ramp inside the measurement.
#   set: "cold" = $COLD_PROMPTS (distinct prompts, nonce at the start, no prefix reuse); otherwise a shared-prefix set
#        $PFX_DIR/<set>.json built by make_prefix_prompts.py (nonce at the end, so the prefix stays cached)
# Env: RESTART_P (required), D_ADDRS (required, comma list of decode DP-rank ip:http_port), PEER (decode host IP),
#      RDMA_DEV (erdma_0 in RDMA mode), DUR (default 60), WDUR (warm-up seconds, default 30), MAXT (default 256),
#      PFX_DIR (default .),
#      COLD_PROMPTS (default distinct_32k.json), OUT (default out)
set -u
TAG=$1; shift
OUT=${OUT:-out}; mkdir -p $OUT
DUR=${DUR:-60}
for sc in "$@"; do
  IFS=: read set qps off <<< "$sc"; off=${off:-0}
  T=$TAG-$set
  bash -c "$RESTART_P" > $OUT/$T-restart.log 2>&1 || { echo "$T: P restart failed"; continue; }
  sleep 15
  pos=suffix; pf=${PFX_DIR:-.}/$set.json
  [ "$set" = cold ] && { pos=prefix; pf=${COLD_PROMPTS:-distinct_32k.json}; }
  nm=$(python3 -c "print(round($qps * $DUR))")
  python3 pd_uniform_load.py --d-addrs $D_ADDRS --prompts-file $pf --nonce-pos $pos --qps $qps --duration ${WDUR:-30} \
    --max-tokens ${MAXT:-256} --warmup 4 --offset $((off + nm + 8)) --nonce ${T}w --out $OUT/$T-warm.jsonl > $OUT/$T-warm.summary 2>&1
  sleep 8
  python3 pd_wire_sampler.py --peer ${PEER:-127.0.0.1} --rdma-dev "${RDMA_DEV:-}" --secs $((DUR + 120)) --interval 0.01 \
    --out $OUT/wire-$T.jsonl > /dev/null 2>&1 &
  SP=$!
  python3 pd_uniform_load.py --d-addrs $D_ADDRS --prompts-file $pf --nonce-pos $pos --qps $qps --duration $DUR \
    --max-tokens ${MAXT:-256} --offset $off --nonce ${T}m --out $OUT/$T.jsonl > $OUT/$T.summary 2>&1
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  echo "$T: $(tail -1 $OUT/$T.summary | cut -c1-200)"
done

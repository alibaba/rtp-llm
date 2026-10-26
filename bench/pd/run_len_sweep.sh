#!/bin/bash
# Usage: run_len_sweep.sh <tag> [n_per_len] [lengths...]   (run on the prefill host, from this directory)
# c1 sweep over input lengths (default 1K/4K/8K/16K/32K) with one continuous wire sampler.
# Every request: distinct prompt + nonce (no prefix reuse), max_tokens 32.
# Needs prompts_<L>_x24.json (JSON list of distinct prompts) per length.
# Env: PEER (decode host IP, required), RDMA_DEV (eRDMA device for RDMA mode, e.g. erdma_0), OFF (prompt offset).
set -u
TAG=$1
N=${2:-8}
shift $(( $# >= 2 ? 2 : $# ))
LENS=${*:-1k 4k 8k 16k 32k}
python3 pd_wire_sampler.py --peer $PEER --rdma-dev "${RDMA_DEV:-}" --secs 900 --interval 0.002 --out wire-$TAG.jsonl > sampler-$TAG.log 2>&1 &
SP=$!
sleep 1
for L in $LENS; do
  python3 pd_ttft_gap_client.py --prompts-file prompts_${L}_x24.json --concurrency 1 --rounds 1 --max-tokens 32 --offset 0 \
    --nonce w-$TAG-$L --tag warm 2>&1 | grep -E " r0 s0 " | sed "s/^/[warm $L] /"
  sleep 1
  for i in $(seq 0 $((N - 1))); do
    python3 pd_ttft_gap_client.py --prompts-file prompts_${L}_x24.json --concurrency 1 --rounds 1 --max-tokens 32 \
      --offset $((i + 1 + ${OFF:-0})) --nonce s-$TAG-$L-$i --tag $TAG-$L-$i --out len-$TAG-$L-$i.json 2>&1 | grep -E " r0 s0 "
    sleep 1.2
  done
done
kill $SP 2>/dev/null
wait $SP 2>/dev/null
ls -la wire-$TAG.jsonl

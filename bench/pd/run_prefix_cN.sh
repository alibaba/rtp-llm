#!/bin/bash
# Usage: run_prefix_cN.sh <tag> <prompts.json> <conc> <rounds>   (run on the prefill host, from this directory)
# Wire sampler for the whole run + shared-prefix client (1 warm-up, then rounds x conc simultaneous).
# Env: PEER (decode host IP, required), RDMA_DEV (e.g. erdma_0), FIRST (first prompt index), GAP (s between rounds).
set -u
TAG=$1
PROMPTS=$2
C=$3
R=$4
FIRST=${FIRST:-1}
GAP=${GAP:-5}
python3 pd_wire_sampler.py --peer $PEER --rdma-dev "${RDMA_DEV:-}" --secs 900 --interval 0.002 --out wire-$TAG.jsonl > sampler-$TAG.log 2>&1 &
SP=$!
sleep 1
python3 pd_prefix_c4_client.py --prompts $PROMPTS --rounds $R --conc $C --max-tokens 32 --first $FIRST --warm 1 \
  --gap $GAP --tag $TAG > client-$TAG.log 2>&1
kill $SP 2>/dev/null
wait $SP 2>/dev/null
head -1 client-$TAG.log
grep -c "err=None" client-$TAG.log

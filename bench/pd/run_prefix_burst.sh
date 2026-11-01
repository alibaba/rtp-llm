#!/bin/bash
# Usage: run_prefix_burst.sh <tag> <prompts.json> <conc> <rounds> <first>   (run on the prefill host, from this directory)
# Gated whole-batch rounds with a connection-pool warm-up: before measuring, rounds of 8, 16, ... conc requests (8-token
# outputs, not counted) let every in-flight request get its own cache-store connection set. A cold jump to 16-32
# simultaneous requests makes all of them connect at once and the RDMA handshakes can time out on the decode side.
# Then <rounds> gated rounds of <conc> requests (pd_prefix_gated_client.py --gate 1.0) with the wire sampler running.
# Env: PEER (decode host IP, required), RDMA_DEV (e.g. erdma_0 in RDMA mode), RPC (P rank-0 RpcService, default
#      127.0.0.1:14001), WARM_FIRST (first prompt index for the warm-up rounds, default 480; use a range the
#      measured rounds do not touch), MAXT (output tokens, default 64), GAP (s between rounds, default 8).
set -u
TAG=$1
PROMPTS=$2
C=$3
R=$4
FIRST=$5
RPC=${RPC:-127.0.0.1:14001}
WF=${WARM_FIRST:-480}
mkdir -p warm out
for wc in $(seq 8 8 $C); do
  python3 pd_prefix_gated_client.py --rpc $RPC --prompts $PROMPTS --first $WF --rounds 1 --conc $wc --max-tokens 8 \
    --gate 1.0 --gap 4 --warm 0 --tag $TAG-warm-c$wc --out-dir warm > warm/$TAG-c$wc.txt 2>&1
  echo "warm c$wc: $(grep -c 'err=None' warm/$TAG-c$wc.txt)/$wc ok"
done
python3 pd_wire_sampler.py --peer $PEER --rdma-dev "${RDMA_DEV:-}" --secs 900 --interval 0.005 --out out/wire-$TAG.jsonl > /dev/null 2>&1 &
SP=$!
sleep 1
python3 pd_prefix_gated_client.py --rpc $RPC --prompts $PROMPTS --first $FIRST --rounds $R --conc $C --max-tokens ${MAXT:-64} \
  --gate 1.0 --gap ${GAP:-8} --tag $TAG --out-dir out > out/$TAG.txt 2>&1
kill $SP 2>/dev/null
wait $SP 2>/dev/null
echo "$TAG: $(grep -c 'err=None' out/$TAG.txt) ok, $(grep -c 'city_ok=True' out/$TAG.txt) correct"

#!/bin/bash
# Usage: run_prefix_gated.sh <tag> <prompts.json> <conc> <rounds>   (run on the prefill host, from this directory)
# Wire sampler for the whole run + pd_prefix_gated_client.py
# (GATE=0: natural arrival; GATE>0: pause P, fire, release after GATE seconds).
# Env: PEER (decode host IP, required), RDMA_DEV (e.g. erdma_0), RPC (P rank-0 RpcService, default 127.0.0.1:14001),
#      FIRST (first prompt index), GAP (s between rounds), GATE.
set -u
TAG=$1
PROMPTS=$2
C=$3
R=$4
FIRST=${FIRST:-1}
GAP=${GAP:-6}
GATE=${GATE:-1.0}
RPC=${RPC:-127.0.0.1:14001}
python3 pd_wire_sampler.py --peer $PEER --rdma-dev "${RDMA_DEV:-}" --secs 900 --interval 0.002 --out wire-$TAG.jsonl > sampler-$TAG.log 2>&1 &
SP=$!
sleep 1
python3 pd_prefix_gated_client.py --rpc $RPC --prompts $PROMPTS --rounds $R --conc $C --max-tokens 32 --first $FIRST \
  --warm 1 --gap $GAP --gate $GATE --tag $TAG > client-$TAG.log 2>&1
python3 -c "import grpc; grpc.insecure_channel('$RPC').unary_unary('/RpcService/SetRestart', request_serializer=lambda _: b'', response_deserializer=lambda b: b)(None, timeout=10)"
kill $SP 2>/dev/null
wait $SP 2>/dev/null
head -1 client-$TAG.log
grep -c "err=None" client-$TAG.log

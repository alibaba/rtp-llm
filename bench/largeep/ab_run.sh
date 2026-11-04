#!/bin/bash
# One EP16 decode configuration measured end to end through PD (P TP4 -> D EP16). Run on a host that reaches P, both
# D hosts and their gate agents. Waits for the 4 D DP leaders, then:
#   warm (600 tok, forced acceptance) -> forced (2400 tok, forced acceptance = step time)
#   -> real (2400 tok, real MTP acceptance) -> real1k (1024 tok) -> cc63 (63 streams: uneven DP batch, padded)
# usage: P_URL=http://<p>:31100 LEADERS=h0:31200,h0:31236,h1:31200,h1:31236 GATE_HOSTS=h0,h1 PROMPTS=prompts.json \
#        bash ab_run.sh TAG [PROFILE=0|1]
# out: $OUT/ab_<TAG>/{warm,forced,real,real1k,cc63}.log; summarise with ab_summary.py $OUT/ab_<TAG>
set -u
TAG=$1; PROFILE=${2:-0}
: "${P_URL:?}" "${LEADERS:?}" "${GATE_HOSTS:?}" "${PROMPTS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-/tmp}/ab_$TAG; mkdir -p $O; rm -f $O/*.log $O/done
PY=${PYTHON:-python3}
G="--p-url $P_URL --d-leaders $LEADERS --gate-hosts $GATE_HOSTS --prompts-file $PROMPTS --gen 2 --prefill-timeout-s 150"
for i in $(seq 1 120); do
  ok=0; for a in ${LEADERS//,/ }; do [ "$(curl -s -m2 http://$a/health)" = '"ok"' ] && ok=$((ok+1)); done
  [ $ok = 4 ] && break; sleep 5
done
echo "healthy=$ok after $((i*5))s" > $O/health
[ $ok = 4 ] || { echo FAIL > $O/done; exit 1; }
FIRST=${LEADERS%%,*}
$PY $HERE/pd_decode_gate.py $G --cc 64 --max-tokens 600 --force-accept --skip-s 2 --window-s 4 > $O/warm.log 2>&1
$PY $HERE/pd_decode_gate.py $G --cc 64 --max-tokens 2400 --force-accept --skip-s 3 --window-s 25 > $O/forced.log 2>&1
( $PY $HERE/pd_decode_gate.py $G --cc 64 --max-tokens 2400 --skip-s 3 --window-s 25 > $O/real.log 2>&1 ) &
RP=$!
for i in $(seq 1 300); do grep -q "release:" $O/real.log 2>/dev/null && break; sleep 0.5; done
sleep 14
for h in ${GATE_HOSTS//,/ }; do curl -s -m3 -X POST $h:${GATE_PORT:-31999}/ts > /dev/null; done
if [ "$PROFILE" = 1 ]; then
  sleep 3
  curl -s -m5 -X POST http://$FIRST/start_profile -d "{\"trace_name\":\"${TAG}p\",\"start_step\":3,\"num_steps\":8,\"enable_all_rank\":true}" > $O/prof.trigger
fi
wait $RP
$PY $HERE/pd_decode_gate.py $G --cc 64 --max-tokens 1024 --skip-s 2 --window-s 12 > $O/real1k.log 2>&1
$PY $HERE/pd_decode_gate.py $G --cc 63 --max-tokens 1024 --skip-s 2 --window-s 12 > $O/cc63.log 2>&1
echo OK > $O/done

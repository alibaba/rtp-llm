#!/bin/bash
# Capture a steady-state decode window of an RTP-LLM server started under `nsys launch`.
#   usage: capture_decode_nsys.sh <out-prefix> <cc> <tokens-per-step> [extra decode_plateau_bench.py args]
# The server must have been started with
#   nsys launch --session-new=$SESSION --trace=cuda,nvtx --cuda-graph-trace=node \
#        --trace-fork-before-exec=true python3 -m rtp_llm.start_server ...
# Env: SESSION (nsys session, default decode), URL (default http://127.0.0.1:8088), PY (python with aiohttp),
#      WAIT (s from bench start to capture; must cover all prefills, default 55), DUR (capture s, default 0.8),
#      NSYS (nsys binary, default nsys).
# Sends cc decode requests (force-accept, 7000 tokens), records DUR seconds once they all decode,
# then exports <out-prefix>.sqlite for dsv4_step_breakdown.py.
set -u
OUTP=${1:?out-prefix}; CC=${2:?cc}; TPS=${3:?tokens-per-step}; shift 3
SESSION=${SESSION:-decode}
URL=${URL:-http://127.0.0.1:8088}
PY=${PY:-python3}
NSYS=${NSYS:-nsys}
HERE=$(cd "$(dirname "$0")" && pwd)
$PY -u "$HERE/decode_plateau_bench.py" --url "$URL" --cc "$CC" --max-tokens 64 --plateau-s 1 > /dev/null 2>&1
$PY -u "$HERE/decode_plateau_bench.py" --url "$URL" --cc "$CC" --max-tokens 7000 \
    --tokens-per-step "$TPS" --plateau-s 6 --force-accept "$@" > "$OUTP.bench.json" 2> "$OUTP.bench.err" &
BP=$!
sleep "${WAIT:-55}"
$NSYS start --session="$SESSION" --output="$OUTP" --force-overwrite=true > "$OUTP.nsys.log" 2>&1
sleep "${DUR:-0.8}"
$NSYS stop --session="$SESSION" >> "$OUTP.nsys.log" 2>&1
kill $BP 2> /dev/null
wait $BP 2> /dev/null
sleep 5
$NSYS export --type sqlite --force-overwrite true -o "$OUTP.sqlite" "$OUTP.nsys-rep" >> "$OUTP.nsys.log" 2>&1
echo "exported $OUTP.sqlite"

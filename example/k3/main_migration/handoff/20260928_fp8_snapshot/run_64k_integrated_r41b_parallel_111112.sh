#!/usr/bin/env bash
set -euo pipefail
umask 077
base=/data6/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927
out="$base/timeline-64k-integrated-mtp-local-r41b-parallel-111112-20260928"
test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test ! -e "$out"
mkdir -p "$out"
set +e
python3 "$base/run_64k_pd_timeline_request.py" \
  --backend rtp \
  --base-url http://11.163.39.111:25100 \
  --output-dir "$out/requests" \
  --model-layers 4 \
  --decode-ip 11.163.39.112 \
  --decode-port 25200 \
  --max-tokens 8 \
  --warmups 10 \
  --max-extra-warmups 4 \
  --timeout 300 \
  --profile-prefill \
  --profile-steps 16 \
  --profile-requests 8 \
  --trace-name k3_64k_prefill_r41b \
  --no-reuse-cache \
  > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
exit "$status"

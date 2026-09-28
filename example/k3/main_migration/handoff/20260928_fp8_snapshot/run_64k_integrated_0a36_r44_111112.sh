#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data6/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-fp8-opt-0a36-20260928"
task="$base/artifacts/k3-fp8-opt-20260927"
out="$task/timeline-64k-integrated-0a36-r44-111112-20260928"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 0a36d6d24829f79e06e229ed53feeefee8913172
test ! -e "$out"
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:25100/health)" = 200
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://11.163.39.112:25200/health)" = 200
mkdir -p "$out"

set +e
python3 "$repo/example/k3/main_migration/handoff/20260928_fp8_snapshot/run_64k_pd_timeline_request.py" \
  --backend rtp \
  --base-url http://127.0.0.1:25100 \
  --output-dir "$out/requests" \
  --model-layers 4 \
  --decode-ip 11.163.39.112 --decode-port 25200 \
  --max-tokens 8 --warmups 10 --max-extra-warmups 4 --timeout 300 \
  --profile-prefill --profile-steps 16 --profile-requests 16 \
  --trace-name k3_64k_prefill_opt_0a36_r44 \
  --no-reuse-cache --disable-thinking --warmup-stability-field first-token \
  > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
exit "$status"

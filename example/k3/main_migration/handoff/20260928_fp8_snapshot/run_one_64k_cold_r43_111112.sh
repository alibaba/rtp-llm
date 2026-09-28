#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data6/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-fp8-opt-0a36-20260928"
task="$base/artifacts/k3-fp8-opt-20260927"
out="$task/diagnostic-cold-64k-0a36-r43-111112-20260928"
script="$repo/example/k3/main_migration/handoff/20260928_fp8_snapshot/run_one_64k_cold_diagnostic.py"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 0a36d6d24829f79e06e229ed53feeefee8913172
test -f "$script"
test ! -e "$out"
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:25100/health)" = 200
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://11.163.39.112:25200/health)" = 200

set +e
python3 "$script" \
  --base-url http://127.0.0.1:25100 \
  --decode-ip 11.163.39.112 --decode-port 25200 \
  --output-dir "$out" --timeout 300 > "$task/diagnostic-cold-64k-0a36-r43-111112.stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$task/diagnostic-cold-64k-0a36-r43-111112.exit"
exit "$status"

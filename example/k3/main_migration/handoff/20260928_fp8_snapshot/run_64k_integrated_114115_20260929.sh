#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data0/luohaocheng.lhc
repo="$base/integrated-worktrees-20260929/rtp-llm-k3-integration-perf-20260929"
task="$base/artifacts/k3-fp8-opt-20260927"
out="$task/timeline-64k-integrated-114115-20260929"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 869efba9652ee9cc3f4e7abcca1d372b8e31b11e
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:26500/health)" = 200
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://11.163.39.115:26600/health)" = 200
test ! -e "$out"
mkdir -p "$out"
cd "$repo"

set +e
python3 example/k3/main_migration/handoff/20260928_fp8_snapshot/run_64k_pd_timeline_request.py \
  --backend rtp --base-url http://127.0.0.1:26500 \
  --decode-ip 11.163.39.115 --decode-port 26600 \
  --output-dir "$out/requests" --model-layers 4 \
  --max-tokens 8 --warmups 10 --max-extra-warmups 4 --timeout 300 \
  --warmup-stability-field first-token \
  --profile-prefill --profile-steps 16 --profile-requests 16 \
  --trace-name k3_64k_integrated_114115 --no-reuse-cache \
  --disable-thinking > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
if [[ "$status" == 0 ]]; then
  python3 example/k3/main_migration/handoff/20260928_fp8_snapshot/audit_64k_pd_requests.py \
    "$out" --decode-ip 11.163.39.115 --decode-port 26600 \
    > "$out/audit-stdout.log" 2>&1
  status=$?
fi
exit "$status"

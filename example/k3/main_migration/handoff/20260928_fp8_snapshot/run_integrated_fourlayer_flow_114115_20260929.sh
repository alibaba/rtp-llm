#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data0/luohaocheng.lhc
repo="$base/integrated-worktrees-20260929/rtp-llm-k3-integration-perf-20260929"
task="$base/artifacts/k3-fp8-opt-20260927"
out="$task/integrated-flow-114115-20260929"
checkpoint=/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 869efba9652ee9cc3f4e7abcca1d372b8e31b11e
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:26500/health)" = 200
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://11.163.39.115:26600/health)" = 200
test ! -e "$out"
mkdir -p "$out"
printf '%s\n' 869efba9652ee9cc3f4e7abcca1d372b8e31b11e > "$out/source-commit.txt"
cd "$repo"

set +e
python3 example/k3/main_migration/text_smoke.py \
  --base-url http://127.0.0.1:26500 \
  --decode-health-url http://11.163.39.115:26600/health \
  --decode-role-addr 11.163.39.115:26600:26601 \
  --decode-dp-size 1 --output "$out/result.json" --suite flow \
  --namespace integrated-fp8-4l-114115-20260929 \
  --batch-size 4 --block-size 4096 --reuse-unit-tokens 4096 \
  --chunk-tokens 65536 --require-mtp --rdma-prewarm-attempts 0 \
  --long-prefix-checkpoint "$checkpoint" \
  --long-prefix-tp-size 8 --long-prefix-kernel-page-size 128 \
  --max-tokens 16 --timeout 300 > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
if [[ "$status" == 0 ]]; then
  python3 example/k3/main_migration/handoff/20260928_fp8_snapshot/audit_fourlayer_flow.py \
    "$out" --decode-ip 11.163.39.115 --decode-port 26600 \
    > "$out/audit-stdout.log" 2>&1
  status=$?
fi
exit "$status"

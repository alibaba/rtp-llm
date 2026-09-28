#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data6/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-fp8-opt-0a36-20260928"
out="$base/artifacts/k3-fp8-opt-20260927/smoke-4layer-fp8-opt-0a36-r42-111112-20260928"
checkpoint=/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 0a36d6d24829f79e06e229ed53feeefee8913172
test "$(findmnt -T "$checkpoint" -n -o FSTYPE | sort -u)" = fuse.hf3fs
test ! -e "$out"
mkdir -p "$out"
printf '%s\n' 0a36d6d24829f79e06e229ed53feeefee8913172 > "$out/source-commit.txt"
cd "$repo"

set +e
python3 example/k3/main_migration/text_smoke.py \
  --base-url http://11.163.39.111:25100 \
  --decode-health-url http://11.163.39.112:25200/health \
  --decode-role-addr 11.163.39.112:25200:25201 \
  --decode-dp-size 1 \
  --output "$out/result.json" \
  --suite flow \
  --namespace main-k3-fp8-opt-0a36-r42-3fs-111112-20260928 \
  --batch-size 4 --block-size 4096 --reuse-unit-tokens 4096 \
  --chunk-tokens 65536 --require-mtp --rdma-prewarm-attempts 0 \
  --long-prefix-checkpoint "$checkpoint" \
  --long-prefix-tp-size 8 --long-prefix-kernel-page-size 128 \
  --timeout 300 \
  > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
exit "$status"

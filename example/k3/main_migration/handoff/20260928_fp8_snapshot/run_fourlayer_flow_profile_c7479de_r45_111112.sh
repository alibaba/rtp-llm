#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data6/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-fp8-profile-c7479de-20260928"
out="$base/artifacts/k3-fp8-opt-20260927/smoke-4layer-fp8-profile-c7479de-r45-111112-20260928"
checkpoint=/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = c7479de2ae9c1577f6ddbb9b75dcb1ec04d5f9b2
test "$(findmnt -T "$checkpoint" -n -o FSTYPE | sort -u)" = fuse.hf3fs
test ! -e "$out"
mkdir -p "$out"
printf '%s\n' c7479de2ae9c1577f6ddbb9b75dcb1ec04d5f9b2 > "$out/source-commit.txt"
cd "$repo"

set +e
python3 example/k3/main_migration/text_smoke.py \
  --base-url http://11.163.39.111:25100 \
  --decode-health-url http://11.163.39.112:25200/health \
  --decode-role-addr 11.163.39.112:25200:25201 \
  --decode-dp-size 1 \
  --output "$out/result.json" \
  --suite flow \
  --namespace main-k3-fp8-profile-c7479de-r45-3fs-111112-20260928 \
  --batch-size 4 --block-size 4096 --reuse-unit-tokens 4096 \
  --chunk-tokens 65536 --require-mtp --rdma-prewarm-attempts 0 \
  --long-prefix-checkpoint "$checkpoint" \
  --long-prefix-tp-size 8 --long-prefix-kernel-page-size 128 \
  --timeout 300 \
  > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
exit "$status"

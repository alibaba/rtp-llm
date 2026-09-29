#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data0/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-fp8-kmerge-20260929"
out="$base/artifacts/k3-fp8-opt-20260927/smoke-93layer-fp8-integrated-f67-114115-20260929"
checkpoint=/mnt/hf3fs/3fs/models/kimi/kimi-k3

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = f67f903e09e7db113820325f775ac21d783b0037
test "$(findmnt -T "$checkpoint" -n -o FSTYPE | sort -u)" = fuse.hf3fs
test ! -e "$out"
mkdir -p "$out"
cd "$repo"

set +e
python3 example/k3/main_migration/text_smoke.py \
  --base-url http://11.163.39.114:27100 \
  --decode-health-url http://11.163.39.115:27200/health \
  --decode-role-addr 11.163.39.115:27200:27201 \
  --decode-dp-size 1 \
  --output "$out/result.json" \
  --suite main-text-64k-capped \
  --namespace main-k3-fp8-integrated-f67-114115-20260929 \
  --batch-size 4 \
  --block-size 4096 \
  --reuse-unit-tokens 4096 \
  --chunk-tokens 65536 \
  --require-mtp \
  --rdma-prewarm-attempts 0 \
  --long-prefix-checkpoint "$checkpoint" \
  --long-prefix-tp-size 8 \
  --long-prefix-kernel-page-size 128 \
  --timeout 300 \
  > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
exit "$status"

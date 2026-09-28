#!/usr/bin/env bash
set -euo pipefail
umask 077

mode=${1:-}
base=/data6/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-fp8-pagedmeta-b24-20260929"
task="$base/artifacts/k3-fp8-opt-20260927"
checkpoint=/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers
commit=b24b6cd889093908923a0d62cb1ac3890438b5d4

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
git -C "$repo" merge-base --is-ancestor "$commit" HEAD
git -C "$repo" diff --quiet "$commit" -- rtp_llm
test "$(findmnt -T "$checkpoint" -n -o FSTYPE | sort -u)" = fuse.hf3fs
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:28100/health)" = 200
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://11.163.39.112:28200/health)" = 200
cd "$repo"

case "$mode" in
  flow)
    out="$task/smoke-4layer-fp8-pagedmeta-b24-r2-111112-20260929"
    test ! -e "$out"
    mkdir -p "$out"
    printf '%s\n' "$commit" > "$out/source-commit.txt"
    set +e
    python3 example/k3/main_migration/text_smoke.py \
      --base-url http://11.163.39.111:28100 \
      --decode-health-url http://11.163.39.112:28200/health \
      --decode-role-addr 11.163.39.112:28200:28201 \
      --decode-dp-size 1 --output "$out/result.json" \
      --suite flow --namespace main-k3-fp8-pagedmeta-b24-r2-3fs-111112-20260929 \
      --batch-size 4 --block-size 4096 --reuse-unit-tokens 4096 \
      --chunk-tokens 65536 --require-mtp --rdma-prewarm-attempts 0 \
      --long-prefix-checkpoint "$checkpoint" \
      --long-prefix-tp-size 8 --long-prefix-kernel-page-size 128 \
      --max-tokens 16 --timeout 300 > "$out/stdout.log" 2>&1
    ;;
  timeline)
    out="$task/timeline-64k-integrated-pagedmeta-b24-r2-111112-20260929"
    test ! -e "$out"
    mkdir -p "$out"
    printf '%s\n' "$commit" > "$out/source-commit.txt"
    set +e
    python3 example/k3/main_migration/handoff/20260928_fp8_snapshot/run_64k_pd_timeline_request.py \
      --backend rtp --base-url http://127.0.0.1:28100 \
      --decode-ip 11.163.39.112 --decode-port 28200 \
      --output-dir "$out/requests" --model-layers 4 \
      --max-tokens 8 --warmups 10 --max-extra-warmups 4 --timeout 300 \
      --warmup-stability-field http \
      --profile-prefill --profile-steps 16 --profile-requests 16 \
      --trace-name k3_64k_integrated_pagedmeta_b24_r2 --no-reuse-cache \
      --disable-thinking > "$out/stdout.log" 2>&1
    ;;
  *) echo 'usage: run_pagedmeta_b24_checks_111112_r2.sh flow|timeline' >&2; exit 2 ;;
esac
status=$?
printf '%s\n' "$status" > "$out/exit"
exit "$status"

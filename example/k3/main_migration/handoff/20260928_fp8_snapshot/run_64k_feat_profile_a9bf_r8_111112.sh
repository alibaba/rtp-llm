#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data6/luohaocheng.lhc
feat="$base/worktrees/rtp-llm-k3-feat-profile-8587b31-20260928"
runner="$base/worktrees/rtp-llm-k3-fp8-opt-0a36-20260928/example/k3/main_migration/handoff/20260928_fp8_snapshot/run_64k_pd_timeline_request.py"
out="$base/artifacts/k3-fp8-opt-20260927/timeline-64k-feat-profile-a9bf-r8-111112-20260929"

test "$(id -un)" = luohaocheng.lhc
test "$(git -C "$feat" rev-parse HEAD)" = a9bf762e878fc54ee9176da5c34ffbe6babc8d45
test "$(sha256sum "$runner" | cut -d' ' -f1)" = 7f2af6f574530e1a83a05e3dabb35615c95d6106f160418b3794aec80d1ba0cb
test ! -e "$out"
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:26300/health)" = 200
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://11.163.39.112:26400/health)" = 200
mkdir -p "$out"

set +e
python3 "$runner" \
  --backend rtp --base-url http://127.0.0.1:26300 \
  --decode-ip 11.163.39.112 --decode-port 26400 \
  --output-dir "$out/requests" --model-layers 4 \
  --max-tokens 8 --warmups 10 --max-extra-warmups 4 --timeout 300 \
  --warmup-stability-field first-token \
  --profile-prefill --profile-steps 16 --profile-requests 16 \
  --trace-name k3_64k_feat_profile_a9bf_r8 --no-reuse-cache \
  --disable-thinking --legacy-feat-aux \
  > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
exit "$status"

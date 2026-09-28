#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data7/luohaocheng.lhc
feat="$base/worktrees/rtp-llm-k3-feat-a9bf-nocp-r9"
runner="$base/artifacts/k3-fp8-opt-20260927/run_64k_pd_timeline_request_r9.py"
out="$base/artifacts/k3-fp8-opt-20260927/timeline-64k-feat-nocp-a9bf-r11-110112-20260929"

test "$(id -un)" = luohaocheng.lhc
test "$(git -C "$feat" rev-parse HEAD)" = a9bf762e878fc54ee9176da5c34ffbe6babc8d45
test "$(sha256sum "$runner" | cut -d' ' -f1)" = 02eb9ed6f66f9cbb54710e5abe2d9e887f6067ab34844e865c2c69e7896756ce
test ! -e "$out"
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:26500/health)" = 200
test "$(curl --noproxy '*' -sS -m 3 -o /dev/null -w '%{http_code}' http://11.163.39.112:26600/health)" = 200
mkdir -p "$out"

set +e
python3 "$runner" \
  --backend rtp --base-url http://127.0.0.1:26500 \
  --decode-ip 11.163.39.112 --decode-port 26600 \
  --output-dir "$out/requests" --model-layers 4 \
  --max-tokens 8 --warmups 10 --max-extra-warmups 4 --timeout 300 \
  --warmup-stability-field first-token \
  --profile-prefill --profile-steps 16 --profile-requests 16 \
  --trace-name k3_64k_feat_nocp_a9bf_r11_110112 --no-reuse-cache \
  --disable-thinking --legacy-feat-aux \
  > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
exit "$status"

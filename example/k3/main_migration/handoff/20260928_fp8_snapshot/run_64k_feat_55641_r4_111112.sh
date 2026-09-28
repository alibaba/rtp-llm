#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data6/luohaocheng.lhc
feat="$base/worktrees/rtp-llm-k3-feat-55641e09-20260928"
runner="$base/worktrees/rtp-llm-k3-fp8-opt-0a36-20260928/example/k3/main_migration/handoff/20260928_fp8_snapshot/run_64k_pd_timeline_request.py"
out="$base/artifacts/k3-fp8-opt-20260927/timeline-64k-feat-55641-r4-111112-20260928"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$feat" rev-parse HEAD)" = 55641e09bc09cdafcf8f31b28aa55b18bc66d24b
test -f "$runner"
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
  --trace-name k3_64k_feat_55641_r4 --no-reuse-cache \
  --disable-thinking --legacy-feat-aux \
  > "$out/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit"
exit "$status"

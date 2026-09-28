#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data1/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-feat-profile-8587b31-20260928"
artifact="$base/artifacts/k3-fp8-main-20260926"
output="$base/.cache/bazel/k3-feat-profile-a9bf762e"
expected=a9bf762e878fc54ee9176da5c34ffbe6babc8d45

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = "$expected"
test "$(findmnt -T "$repo" -n -o FSTYPE)" = ext4
test "$(findmnt -T "$output" -n -o FSTYPE)" = ext4
test -f "$artifact/feat-external-git-names.txt"
cd "$repo"

cmd=(bazelisk "--output_user_root=$output" build
  --config=cuda13 --config=sm10x --define=use_accl_ep=0
  "--distdir=$artifact"
  "--override_repository=arch_config=$artifact/rdma-build-overlay/arch_config"
  "--override_repository=rtp_deps=$artifact/rdma-build-overlay/rtp_deps"
  "--override_repository=xgrammar=$artifact/xgrammar-384264-source"
  --jobs=24)
while IFS= read -r name; do
  test -d "$artifact/feat-external-git-sources/$name"
  cmd+=("--override_repository=$name=$artifact/feat-external-git-sources/$name")
done < "$artifact/feat-external-git-names.txt"
cmd+=(//rtp_llm:rtp_llm_server)

printf 'host=112 container=lhc_GPU user=%s source=%s source_fs=ext4 output_root=%s output_fs=ext4 command:' \
  "$(id -un)" "$repo" "$output"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
test -x "$repo/bazel-bin/rtp_llm/rtp_llm_server"

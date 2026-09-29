#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data6/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-feat-55641e09-20260928"
artifact="$base/artifacts/k3-fp8-main-20260926"
task="$base/artifacts/k3-fp8-opt-20260927"
output="$base/.cache/bazel/k3-feat-55641e09-20260928"
log="$task/build-feat-55641-111.log"
status_file="$task/build-feat-55641-111.exit"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 55641e09bc09cdafcf8f31b28aa55b18bc66d24b
test "$(findmnt -T "$repo" -n -o FSTYPE)" = ext4
test "$(readlink -f "$repo/internal_source")" = "$artifact/internal-feat-f6b258d7/internal_source"
test -f "$artifact/feat-external-git-names.txt"
test ! -e "$log" && test ! -e "$status_file"
mkdir -p "$output"
test "$(findmnt -T "$output" -n -o FSTYPE)" = ext4

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

cd "$repo"
{
  printf 'container=lhc_GPU user=%s source=%s source_fs=ext4 output_root=%s output_fs=ext4 command:' "$(id -un)" "$repo" "$output"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  set +e
  "${cmd[@]}"
  status=$?
  printf '%s\n' "$status" > "$status_file"
  exit "$status"
} > "$log" 2>&1

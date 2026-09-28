#!/usr/bin/env bash
# Rebuild the fixed feat/k3_dev diagnostic worktree in each host's own container.
set -euo pipefail
umask 077

base=${1:?personal data root required}
expected_sha=${2:?diagnostic commit SHA required}
repo="$base/worktrees/rtp-llm-k3-feat-profile-8587b31-20260928"
deps="$base/artifacts/k3-fp8-main-20260926"
task="$base/artifacts/k3-fp8-opt-20260927"
output="$base/.cache/bazel/k3-feat-profile-${expected_sha:0:8}"
log="$task/build-feat-profile-${expected_sha:0:8}.log"
status_file="$task/build-feat-profile-${expected_sha:0:8}.exit"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_sha"
test "$(findmnt -T "$repo" -n -o FSTYPE)" = ext4
test -d "$(readlink -f "$repo/internal_source")"
test -r "$deps/feat-external-git-names.txt"
test ! -e "$log" && test ! -e "$status_file"
mkdir -p "$output"
test "$(findmnt -T "$output" -n -o FSTYPE)" = ext4

cmd=(bazelisk "--output_user_root=$output" build
  --config=cuda13 --config=sm10x --define=use_accl_ep=0
  "--distdir=$deps"
  "--override_repository=arch_config=$deps/rdma-build-overlay/arch_config"
  "--override_repository=rtp_deps=$deps/rdma-build-overlay/rtp_deps"
  "--override_repository=xgrammar=$deps/xgrammar-384264-source"
  --jobs=24)
while IFS= read -r name; do
  test -d "$deps/feat-external-git-sources/$name"
  cmd+=("--override_repository=$name=$deps/feat-external-git-sources/$name")
done < "$deps/feat-external-git-names.txt"
cmd+=(//rtp_llm:rtp_llm_server)

cd "$repo"
{
  printf 'container=lhc_GPU user=%s source=%s source_fs=ext4 output_root=%s output_fs=ext4 command:' \
    "$(id -un)" "$repo" "$output"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  set +e
  "${cmd[@]}"
  result=$?
  printf '%s\n' "$result" > "$status_file"
  exit "$result"
} > "$log" 2>&1

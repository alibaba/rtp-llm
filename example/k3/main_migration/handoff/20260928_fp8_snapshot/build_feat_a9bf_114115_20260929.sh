#!/usr/bin/env bash
set -euo pipefail
umask 077

host_id=${1:?usage: build_feat_a9bf_114115_20260929.sh 114\|115}
case "$host_id" in
  114|115) ;;
  *) echo 'only 114 and 115 are configured' >&2; exit 2 ;;
esac

base=/data0/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-feat-a9bf-perf-20260929"
distdir="$base/artifacts/k3-fp8-main-20260926"
deps="$distdir"
if [[ "$host_id" == 114 ]]; then
  deps="$base/artifacts/k3-fp8-opt-20260927/feat-a9bf-build-deps-20260929"
fi
output="$base/.cache/bazel/k3-feat-a9bf-perf-20260929"
export HOME=/home/luohaocheng.lhc
export TMPDIR="$base/tmp/k3-feat-a9bf-perf-20260929"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = a9bf762e878fc54ee9176da5c34ffbe6babc8d45
test "$(findmnt -T "$repo" -n -o FSTYPE)" = xfs
test -f "$repo/internal_source/deps/git.bzl"
test "$(sha256sum "$repo/internal_source/deps/git.bzl" | cut -d' ' -f1)" = 57ff5a594f44d2e966de2bd4d4b38a3fa0cbfdcd8faa2464c891cd04a68b7d71
test "$(sha256sum "$deps/feat-external-git-names.txt" | cut -d' ' -f1)" = b532c02022a73a00ef9674088f5753bae2328b5e2c91b4a2a9542c76828b49b0
test -d "$deps/rdma-build-overlay/arch_config"
test -d "$deps/rdma-build-overlay/rtp_deps"
test -d "$deps/xgrammar-384264-source"
mkdir -p "$output" "$TMPDIR"
test "$(findmnt -T "$output" -n -o FSTYPE)" = xfs
test "$(findmnt -T "$TMPDIR" -n -o FSTYPE)" = xfs
cd "$repo"

cmd=(bazelisk "--output_user_root=$output" build
  --config=cuda13 --config=sm10x --define=use_accl_ep=0
  "--distdir=$distdir"
  "--override_repository=arch_config=$deps/rdma-build-overlay/arch_config"
  "--override_repository=rtp_deps=$deps/rdma-build-overlay/rtp_deps"
  "--override_repository=xgrammar=$deps/xgrammar-384264-source"
  --jobs=24)
while IFS= read -r name; do
  test -d "$deps/feat-external-git-sources/$name"
  cmd+=("--override_repository=$name=$deps/feat-external-git-sources/$name")
done < "$deps/feat-external-git-names.txt"
cmd+=(//rtp_llm:rtp_llm_server)

printf 'host=%s container=lhc_GPU user=%s source=%s source_fs=xfs output_root=%s output_fs=xfs command:' \
  "$host_id" "$(id -un)" "$repo" "$output"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
test -x "$repo/bazel-bin/rtp_llm/rtp_llm_server"

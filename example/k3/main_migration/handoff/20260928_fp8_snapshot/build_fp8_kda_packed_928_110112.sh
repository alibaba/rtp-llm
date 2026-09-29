#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  110)
    base=/data7/luohaocheng.lhc
    pip_repos="$base/.cache/bazel/k3-main-20260926/14e92cbfe48a8a41a60e44f9c348e502/external"
    bazel=/usr/local/bin/bazelisk
    ;;
  112)
    base=/data1/luohaocheng.lhc
    pip_repos="$base/.cache/bazel/k3-integrated-20260928-112/dbf6ebb01707840f616367bd9b046564/external"
    bazel="$base/tools/bazel-6.4.0"
    ;;
  *) echo 'usage: build_fp8_kda_packed_928_110112.sh 110|112' >&2; exit 2 ;;
esac

repo="$base/worktrees/rtp-llm-k3-fp8-opt-kda-packed-928-20260928"
deps="$base/artifacts/k3-fp8-main-20260926"
root="$base/.cache/bazel/k3-kda-packed-928-20260928-$1"
export TMPDIR="$base/tmp/k3-kda-packed-928-build-$1"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 928ad6a6d23e1205113334efaeeb7e6e18f9f1b1
test "$(readlink -f "$repo/internal_source")" = "$deps/internal_source_main"
test "$(findmnt -T "$repo" -n -o FSTYPE)" = ext4
test -d "$pip_repos"
test -x "$bazel"
test -d "$deps/rdma-build-overlay/arch_config"
test -d "$deps/rdma-build-overlay/rtp_deps"
test -d "$deps/xgrammar-384264-source"
mkdir -p "$root" "$TMPDIR"
test "$(findmnt -T "$root" -n -o FSTYPE)" = ext4
test "$(findmnt -T "$TMPDIR" -n -o FSTYPE)" = ext4

cd "$repo"
out="$($bazel "--output_user_root=$root" info --config=cuda13 --config=sm10x output_base)"
test "$(findmnt -T "$out" -n -o FSTYPE)" = ext4
case "$out" in "$base"/*) ;; *) echo "output outside personal data disk: $out" >&2; exit 2;; esac

cmd=("$bazel" "--output_user_root=$root" build
  --config=cuda13 --config=sm10x
  --define=use_accl_ep=0 "--distdir=$deps"
  "--override_repository=arch_config=$deps/rdma-build-overlay/arch_config"
  "--override_repository=rtp_deps=$deps/rdma-build-overlay/rtp_deps"
  "--override_repository=xgrammar=$deps/xgrammar-384264-source"
  --jobs=32 //rtp_llm:rtp_llm_server)
for source in "$deps/feat-external-git-sources"/*; do
  test -d "$source"
  cmd+=("--override_repository=${source##*/}=$source")
done
count=0
for source in "$pip_repos"/pip_*; do
  test -f "$source/WORKSPACE"
  test -f "$source/BUILD.bazel"
  cmd+=("--override_repository=${source##*/}=$source")
  count=$((count + 1))
done
test "$count" -ge 170

printf 'container=lhc_GPU_k3_3fs_20260927 user=%s source=%s source_fs=ext4 output=%s output_fs=ext4 pip_overrides=%s command:' \
  "$(id -un)" "$repo" "$out" "$count"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

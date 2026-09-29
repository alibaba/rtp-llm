#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  111) base=/data6/luohaocheng.lhc ;;
  112) base=/data1/luohaocheng.lhc ;;
  *) echo 'usage: build_fp8_opt_0a36_111112.sh 111|112' >&2; exit 2 ;;
esac

repo="$base/worktrees/rtp-llm-k3-fp8-opt-0a36-20260928"
deps="$base/artifacts/k3-fp8-main-20260926"
root="$base/.cache/bazel/k3-integrated-0a36-20260928"
bazel="$base/tools/bazel-6.4.0"
export TMPDIR="$base/tmp/k3-fp8-opt-0a36-build"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 0a36d6d24829f79e06e229ed53feeefee8913172
test "$(readlink -f "$repo/internal_source")" = "$deps/internal_source_main"
test "$(findmnt -T "$repo" -n -o FSTYPE)" = ext4
mkdir -p "$root" "$TMPDIR"
test "$(findmnt -T "$root" -n -o FSTYPE)" = ext4
test "$(findmnt -T "$TMPDIR" -n -o FSTYPE)" = ext4
test -x "$bazel"
test -d "$deps/rdma-build-overlay/arch_config"
test -d "$deps/rdma-build-overlay/rtp_deps"
test -d "$deps/xgrammar-384264-source"

cd "$repo"
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

printf 'container=lhc_GPU user=%s source=%s source_fs=ext4 output=%s output_fs=ext4 command:' "$(id -un)" "$repo" "$root"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

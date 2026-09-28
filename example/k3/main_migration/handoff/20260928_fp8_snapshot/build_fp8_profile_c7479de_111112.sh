#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  111)
    base=/data6/luohaocheng.lhc
    pip_repos="$base/artifacts/k3-fp8-main-20260926/pip-repositories"
    ;;
  112)
    base=/data1/luohaocheng.lhc
    pip_repos="$base/.cache/bazel/k3-integrated-20260928-112/dbf6ebb01707840f616367bd9b046564/external"
    ;;
  *) echo 'usage: build_fp8_profile_c7479de_111112.sh 111|112' >&2; exit 2 ;;
esac

repo="$base/worktrees/rtp-llm-k3-fp8-profile-c7479de-20260928"
deps="$base/artifacts/k3-fp8-main-20260926"
root="$base/.cache/bazel/k3-integrated-profile-c7479de-20260928"
bazel="$base/tools/bazel-6.4.0"
export TMPDIR="$base/tmp/k3-fp8-profile-c7479de-build"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = c7479de2ae9c1577f6ddbb9b75dcb1ec04d5f9b2
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

# The previous build already verified the same locked CUDA13 wheels. Reuse its
# repository trees, which are local to this host, to avoid downloading them
# again at roughly 0.5–0.7 MB/s. Bazel reads these trees without changing them.
count=0
for source in "$pip_repos"/pip_*; do
  test -f "$source/WORKSPACE"
  test -f "$source/BUILD.bazel"
  cmd+=("--override_repository=${source##*/}=$source")
  count=$((count + 1))
done
test "$count" -ge 170

printf 'container=lhc_GPU user=%s source=%s pip_cache=%s pip_overrides=%s command:' \
  "$(id -un)" "$repo" "$pip_repos" "$count"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

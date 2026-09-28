#!/usr/bin/env bash
set -euo pipefail

base=/data7/luohaocheng.lhc
repo="$base/worktrees/rtp-llm-k3-mla-kmerge-redgreen-20260929"
deps="$base/artifacts/k3-fp8-main-20260926"
pip_repos="$base/.cache/bazel/k3-main-20260926/14e92cbfe48a8a41a60e44f9c348e502/external"
root="$base/.cache/bazel/k3-mla-kmerge-20260929"
export TMPDIR="$base/tmp/k3-mla-kmerge-build-20260929"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(findmnt -T "$repo" -n -o FSTYPE)" = ext4
test "$(readlink -f "$repo/internal_source")" = "$deps/internal_source_main"
test -d "$pip_repos"
test -d "$deps/rdma-build-overlay/arch_config"
test -d "$deps/rdma-build-overlay/rtp_deps"
test -d "$deps/xgrammar-384264-source"
mkdir -p "$root" "$TMPDIR"
test "$(findmnt -T "$root" -n -o FSTYPE)" = ext4
test "$(findmnt -T "$TMPDIR" -n -o FSTYPE)" = ext4

cd "$repo"
bazel=/usr/local/bin/bazelisk
out="$("$bazel" "--output_user_root=$root" info --config=cuda13 --config=sm10x output_base)"
test "$(findmnt -T "$out" -n -o FSTYPE)" = ext4

cmd=("$bazel" "--output_user_root=$root" test
  --config=cuda13 --config=sm10x
  --define=use_accl_ep=0 "--distdir=$deps"
  "--override_repository=arch_config=$deps/rdma-build-overlay/arch_config"
  "--override_repository=rtp_deps=$deps/rdma-build-overlay/rtp_deps"
  "--override_repository=xgrammar=$deps/xgrammar-384264-source"
  --jobs=24 --test_env=CUDA_VISIBLE_DEVICES=1
  --test_output=errors --cache_test_results=no
  //rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:mla_k_merge_non_power_two_heads_test)
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

printf 'user=%s source=%s source_fs=ext4 output=%s output_fs=ext4 pip_overrides=%s command:' \
  "$(id -un)" "$repo" "$out" "$count"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

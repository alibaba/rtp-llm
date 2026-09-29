#!/usr/bin/env bash
set -euo pipefail
umask 077

host_id=${1:?usage: build_integrated_114115_20260929.sh 114\|115}
case "$host_id" in
  114)
    deps=/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/feat-a9bf-build-deps-20260929
    main_internal=/data0/luohaocheng.lhc/artifacts/k3-fp8-main-20260926/internal_source
    existing_bazel_root=/data0/luohaocheng.lhc/.cache/bazel/k3-feat-a9bf-perf-20260929
    bazel=bazelisk
    ;;
  115)
    deps=/data0/luohaocheng.lhc/artifacts/k3-fp8-main-20260926
    main_internal="$deps/internal_source_main"
    existing_bazel_root=/data0/luohaocheng.lhc/.cache/bazel/k3-main-20260926
    bazel=/data0/luohaocheng.lhc/tools/bazel-6.4.0
    ;;
  *) echo 'only 114 and 115 are configured' >&2; exit 2 ;;
esac

base=/data0/luohaocheng.lhc
repo="$base/integrated-worktrees-20260929/rtp-llm-k3-integration-perf-20260929"
output="$base/.cache/bazel/k3-integrated-perf-20260929"
export TMPDIR="$base/tmp/k3-integrated-perf-build-20260929"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 869efba9652ee9cc3f4e7abcca1d372b8e31b11e
test "$(readlink -f "$repo/internal_source")" = "$main_internal"
test "$(sha256sum "$main_internal/deps/git.bzl" | cut -d' ' -f1)" = 51c749ad063a6106e42f1f02d8f01334d45d9a14c4c488cea4fca05a3b71c159
test "$(findmnt -T "$repo" -n -o FSTYPE)" = xfs
test -d "$deps/rdma-build-overlay/arch_config"
test -d "$deps/rdma-build-overlay/rtp_deps"
test -d "$deps/xgrammar-384264-source"
test -d "$deps/feat-external-git-sources"
test -d "$existing_bazel_root/cache/repos/v1"
command -v "$bazel" >/dev/null
mkdir -p "$output" "$TMPDIR"
test "$(findmnt -T "$output" -n -o FSTYPE)" = xfs
test "$(findmnt -T "$TMPDIR" -n -o FSTYPE)" = xfs

cd "$repo"
# The first analysis of this exact source/output pair generated the current
# pip lock metadata. Match every reused wheel against that metadata before
# borrowing this host's own previously populated main cache. The new packages
# stay on the normal pinned download path.
old_meta=("$base/.cache/bazel/k3-main-20260926"/*/external/pip_gpu_cuda13_torch/requirements.bzl)
new_meta=("$output"/*/external/pip_gpu_cuda13_torch/requirements.bzl)
test "${#old_meta[@]}" -eq 1 && test -f "${old_meta[0]}"
test "${#new_meta[@]}" -eq 1 && test -f "${new_meta[0]}"
old_external=${old_meta[0]%/pip_gpu_cuda13_torch/requirements.bzl}
pip_repos=$(python3 - "${old_meta[0]}" "${new_meta[0]}" "$old_external" <<'PY'
import re
from pathlib import Path
import sys

pattern = re.compile(r'^\s*\("([^"]+)", "([^"]+)"\),?$')

def locked_wheels(path):
    return {
        match.group(1): match.group(2)
        for line in Path(path).read_text().splitlines()
        if (match := pattern.match(line))
    }

old, current = map(locked_wheels, sys.argv[1:3])
expected_added = {
    "pip_gpu_cuda13_torch_cuda_linear_attention",
    "pip_gpu_cuda13_torch_flash_linear_attention",
}
if len(current) != 180 or len(old) not in {178, 180} or not set(current) - set(old) <= expected_added:
    raise SystemExit("CUDA13 pip lock set differs from the audited source")
if set(old) - set(current) or any(old[name] != current[name] for name in old):
    raise SystemExit("A cached CUDA13 wheel differs in version or hash")
external = Path(sys.argv[3])
reused = 0
for name in sorted(old):
    path = external / name
    if not (path / "WORKSPACE").is_file() or not (path / "BUILD.bazel").is_file():
        # Bazel never materialized this wheel for the previous server target.
        # Leave it on the new build's hash-pinned download path if needed.
        continue
    print(name)
    reused += 1
print(f"CUDA13 pip lock matches; reusing {reused} host-local wheel trees", file=sys.stderr)
PY
)

cmd=("$bazel" "--output_user_root=$output" build
  --config=cuda13 --config=sm10x --define=use_accl_ep=0
  "--distdir=$base/artifacts/k3-fp8-main-20260926"
  "--repository_cache=$existing_bazel_root/cache/repos/v1"
  "--override_repository=arch_config=$deps/rdma-build-overlay/arch_config"
  "--override_repository=rtp_deps=$deps/rdma-build-overlay/rtp_deps"
  "--override_repository=xgrammar=$deps/xgrammar-384264-source"
  --jobs=24)
while IFS= read -r name; do
  test -d "$deps/feat-external-git-sources/$name"
  cmd+=("--override_repository=$name=$deps/feat-external-git-sources/$name")
done < "$deps/feat-external-git-names.txt"
while IFS= read -r name; do
  cmd+=("--override_repository=$name=$old_external/$name")
done <<< "$pip_repos"
cmd+=(//rtp_llm:rtp_llm_server)

printf 'host=%s container=lhc_GPU user=%s source=%s source_fs=xfs output_root=%s output_fs=xfs command:' \
  "$host_id" "$(id -un)" "$repo" "$output"
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${K3_BUILD_PRINT_ONLY:-0}" == 1 ]]; then
  exit 0
fi
"${cmd[@]}"
test -x "$repo/bazel-bin/rtp_llm/rtp_llm_server"

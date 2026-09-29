#!/usr/bin/env bash
set -euo pipefail
umask 077

base=/data1/luohaocheng.lhc
repo="$base/integrated-worktrees-20260929/rtp-llm-k3-integration-perf-20260929"
deps="$base/artifacts/k3-fp8-main-20260926"
cached="$base/.cache/bazel/k3-fp8-pagedmeta-b24-20260929-112"
output="$base/.cache/bazel/k3-integrated-perf-20260929-112"
bazel="$base/tools/bazel-6.4.0"
export TMPDIR="$base/tmp/k3-integrated-perf-build-20260929"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test "$(git -C "$repo" rev-parse HEAD)" = 869efba9652ee9cc3f4e7abcca1d372b8e31b11e
test "$(readlink -f "$repo/internal_source")" = "$deps/internal_source_main"
test "$(sha256sum "$deps/internal_source_main/deps/git.bzl" | cut -d' ' -f1)" = 51c749ad063a6106e42f1f02d8f01334d45d9a14c4c488cea4fca05a3b71c159
test "$(sha256sum "$deps/rdma-build-overlay/rtp_deps/requirements_lock_torch_gpu_cuda13.txt" | cut -d' ' -f1)" = f321c9c055399fff9cdea3c5b5af4b09ac502fe92acc911085e7b1422750dd01
test "$(findmnt -T "$repo" -n -o FSTYPE)" = ext4
test -d "$cached/cache/repos/v1"
test -d "$deps/rdma-build-overlay/arch_config"
test -d "$deps/rdma-build-overlay/rtp_deps"
test -d "$deps/xgrammar-384264-source"
test -d "$deps/feat-external-git-sources"
test -x "$bazel"
mkdir -p "$output" "$TMPDIR"
test "$(findmnt -T "$output" -n -o FSTYPE)" = ext4
test "$(findmnt -T "$TMPDIR" -n -o FSTYPE)" = ext4

# This generated Bazel metadata SHA matches the current 114/115 integrated
# source and the exact CUDA13 lock above. Only materialized, host-local wheel
# trees are reused; missing packages retain Bazel's pinned download path.
cached_meta=("$cached"/*/external/pip_gpu_cuda13_torch/requirements.bzl)
test "${#cached_meta[@]}" -eq 1 && test -f "${cached_meta[0]}"
test "$(sha256sum "${cached_meta[0]}" | cut -d' ' -f1)" = b04c8cce6b628659cd7661aaef82724a75917c335f38cbbe3116825b537d8058
cached_external=${cached_meta[0]%/pip_gpu_cuda13_torch/requirements.bzl}
pip_repos=$(python3 - "${cached_meta[0]}" "$cached_external" <<'PY'
import re
from pathlib import Path
import sys

pattern = re.compile(r'^\s*\("([^"]+)", "([^"]+)"\),?$')
names = []
for line in Path(sys.argv[1]).read_text().splitlines():
    match = pattern.match(line)
    if match is not None:
        names.append(match.group(1))
if len(names) != 180 or len(set(names)) != len(names):
    raise SystemExit("CUDA13 wheel lock differs from audited integrated source")
external = Path(sys.argv[2])
reused = 0
for name in sorted(names):
    path = external / name
    if (path / "WORKSPACE").is_file() and (path / "BUILD.bazel").is_file():
        print(name)
        reused += 1
print(f"CUDA13 pip lock matches; reusing {reused} host-local wheel trees", file=sys.stderr)
PY
)

cd "$repo"
cmd=("$bazel" "--output_user_root=$output" build
  --config=cuda13 --config=sm10x --define=use_accl_ep=0
  "--distdir=$deps" "--repository_cache=$cached/cache/repos/v1"
  "--override_repository=arch_config=$deps/rdma-build-overlay/arch_config"
  "--override_repository=rtp_deps=$deps/rdma-build-overlay/rtp_deps"
  "--override_repository=xgrammar=$deps/xgrammar-384264-source"
  --jobs=24)
while IFS= read -r name; do
  test -d "$deps/feat-external-git-sources/$name"
  cmd+=("--override_repository=$name=$deps/feat-external-git-sources/$name")
done < "$deps/feat-external-git-names.txt"
while IFS= read -r name; do
  cmd+=("--override_repository=$name=$cached_external/$name")
done <<< "$pip_repos"
cmd+=(//rtp_llm:rtp_llm_server)

printf 'host=112 container=lhc_GPU user=%s source=%s source_fs=ext4 output_root=%s output_fs=ext4 command:' \
  "$(id -un)" "$repo" "$output"
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${K3_BUILD_PRINT_ONLY:-0}" == 1 ]]; then
  exit 0
fi
"${cmd[@]}"
test -x "$repo/bazel-bin/rtp_llm/rtp_llm_server"

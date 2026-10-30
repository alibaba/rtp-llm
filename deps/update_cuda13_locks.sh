#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <x86|arm> [--check|--print-command]" >&2
    exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
platform=$1
mode=${2:-update}
[[ "$mode" == update || "$mode" == --check || "$mode" == --print-command ]] || usage
workspace=${BUILD_WORKSPACE_DIRECTORY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
overrides="$workspace/deps/requirements_cuda13_overrides.txt"

case "$platform" in
    x86)
        input="$workspace/deps/requirements_torch_gpu_cuda13.txt"
        output="$workspace/deps/requirements_lock_torch_gpu_cuda13.txt"
        python_platform=x86_64-manylinux_2_28
        target=requirements_torch_gpu_cuda13.update
        # Indexes are caller-supplied: public regeneration uses the defaults
        # below; internal regeneration (where artlab carries the aarch64+cu13
        # artifacts and mirrors public PyPI) sets UV_INDEX/UV_EXTRA_INDEX to
        # the artlab endpoints. Locks themselves carry no index lines.
        indexes=(
            --default-index "${UV_INDEX:-https://pypi.org/simple/}"
            --index "${UV_EXTRA_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
        )
        ;;
    arm)
        input="$workspace/deps/requirements_cuda13_arm.txt"
        output="$workspace/deps/requirements_lock_cuda13_arm.txt"
        python_platform=aarch64-manylinux_2_28
        target=requirements_cuda13_arm.update
        # Same env-driven index selection as x86; internal runners export the
        # artlab set (aarch64+cu13 artifacts live there) before regenerating.
        indexes=(
            --default-index "${UV_INDEX:-https://pypi.org/simple/}"
            --index "${UV_EXTRA_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
        )
        ;;
    *) usage ;;
esac

command=(
    "${UV_BIN:-uv}" --no-config pip compile "$input"
    --overrides "$overrides"
    --python-version 3.10
    --python-platform "$python_platform"
    --generate-hashes
    --no-strip-extras
    "${indexes[@]}"
    --index-strategy unsafe-best-match
    --custom-compile-command "bazelisk run //deps:$target"
    # Locks carry no index lines: --emit-index-url is deliberately omitted so
    # install-time index selection stays with the caller (pip.bzl envsubst).
)

if [[ "$mode" == --print-command ]]; then
    printf '%q ' "${command[@]}"
    printf '%q\n' --output-file "$output"
    exit 0
fi

required_uv_version=${UV_REQUIRED_VERSION:-0.11.1}
actual_uv_version=$("${command[0]}" --version | awk '{print $2}')
[[ "$actual_uv_version" == "$required_uv_version" ]] || {
    echo "uv $required_uv_version is required, found $actual_uv_version" >&2
    exit 1
}

if [[ "$mode" == --check ]]; then
    temporary=$(mktemp)
    trap 'rm -f "$temporary"' EXIT
    # uv pip compile reuses pins found in an existing --output-file; seed the
    # temp with the committed lock so --check compares like-for-like with
    # update mode.  (An unseeded temp re-resolves each package to the latest
    # release and makes --check fail by construction -- verified with uv
    # 0.11.1: seeded check passes, unseeded diffs on e.g. aiohappyeyeballs.)
    cp "$output" "$temporary"
    "${command[@]}" --output-file "$temporary"
    cmp "$temporary" "$output"
else
    "${command[@]}" --output-file "$output"
fi

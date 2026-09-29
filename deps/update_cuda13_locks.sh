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
        ;;
    arm)
        input="$workspace/deps/requirements_cuda13_arm.txt"
        output="$workspace/deps/requirements_lock_cuda13_arm.txt"
        python_platform=aarch64-manylinux_2_28
        target=requirements_cuda13_arm.update
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
    --default-index https://pypi.org/simple/
    --index https://mirrors.aliyun.com/pypi/simple/
    --index-strategy unsafe-best-match
    --emit-index-url
    --custom-compile-command "bazelisk run //deps:$target"
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
    "${command[@]}" --output-file "$temporary"
    cmp "$temporary" "$output"
else
    "${command[@]}" --output-file "$output"
fi

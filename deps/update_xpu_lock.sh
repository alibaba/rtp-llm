#!/usr/bin/env bash
# Regenerate requirements_lock_xpu.txt (Intel XPU, --config=xpu) with uv. Same
# flags and modes as update_cuda13_locks.sh.
set -euo pipefail

usage() {
    echo "Usage: $0 [--check|--print-command]" >&2
    exit 2
}

[[ $# -le 1 ]] || usage
mode=${1:-update}
[[ "$mode" == update || "$mode" == --check || "$mode" == --print-command ]] || usage

workspace=${BUILD_WORKSPACE_DIRECTORY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
input="$workspace/deps/requirements_xpu.txt"
overrides="$workspace/deps/requirements_xpu_overrides.txt"
output="$workspace/deps/requirements_lock_xpu.txt"
target=requirements_xpu.update
# Indexes are caller-supplied, as for the cuda13 locks: torch, torchvision and
# triton-xpu built for XPU come from the PyTorch XPU index, the rest from PyPI.
# Locks themselves carry no index lines.
indexes=(
    --default-index "${UV_INDEX:-https://pypi.org/simple/}"
    --index "${UV_XPU_INDEX:-https://download.pytorch.org/whl/xpu}"
)

command=(
    "${UV_BIN:-uv}" --no-config pip compile "$input"
    --overrides "$overrides"
    --python-version 3.10
    --python-platform x86_64-manylinux_2_28
    --generate-hashes
    --no-strip-extras
    "${indexes[@]}"
    --index-strategy unsafe-best-match
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
    # Seed with the committed lock, as update_cuda13_locks.sh does: uv reuses
    # pins found in an existing --output-file.
    cp "$output" "$temporary"
    "${command[@]}" --output-file "$temporary"
    cmp "$temporary" "$output"
else
    "${command[@]}" --output-file "$output"
fi

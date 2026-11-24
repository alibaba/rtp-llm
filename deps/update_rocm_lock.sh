#!/usr/bin/env bash
set -euo pipefail

# Resolve from the checkout for bazel run, or from the adjacent input files
# in the runfiles tree for the read-only lock test.
if [[ -n ${BUILD_WORKSPACE_DIRECTORY:-} && -d "$BUILD_WORKSPACE_DIRECTORY/deps" ]]; then
    cd "$BUILD_WORKSPACE_DIRECTORY/deps"
else
    cd "$(dirname "${BASH_SOURCE[0]}")"
fi
lock=requirements_lock_rocm.txt
output="$lock"
if [[ ${1:-} == --check ]]; then
    output=$(mktemp)
    trap 'rm -f "$output"' EXIT
    cp "$lock" "$output"
fi
uv=${UV_BIN:-uv}
[[ $("$uv" --version | awk '{print $2}') == 0.11.15 ]] || { echo "ROCm lock requires uv 0.11.15" >&2; exit 1; }
"$uv" --no-config pip compile requirements_rocm.txt --overrides requirements_rocm_overrides.txt \
    --python-version 3.10 --python-platform x86_64-manylinux_2_28 --generate-hashes \
    --default-index "${PIP_INDEX_URL:-https://pypi.org/simple/}" \
    --index "${PIP_EXTRA_INDEX_URL:-https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/simple/}" \
    --index-strategy unsafe-best-match --custom-compile-command 'bazel run @rtp_deps//:requirements_rocm.update' \
    --output-file "$output"
if [[ ${1:-} == --check ]]; then
    cmp "$lock" "$output"
fi

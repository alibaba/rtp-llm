#!/usr/bin/env bash
set -euo pipefail

readonly FLASHINFER_REPOSITORY="https://github.com/flashinfer-ai/flashinfer.git"
readonly FLASHINFER_COMMIT="a1aa676196f798435248d9ea205c67674476f473"
readonly SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PATCH_FILE="${SCRIPT_DIR}/flashinfer-v0.6.9-direct-per-token-kv-scale.patch"

if (( $# > 1 )); then
  echo "usage: $0 [wheel-output-directory]" >&2
  exit 2
fi

OUTPUT_DIR="${1:-/tmp}"
PYTHON="${PYTHON:-python3}"
: "${FLASHINFER_LOCAL_VERSION:=rtp.dynamicfp8.1}"
export FLASHINFER_LOCAL_VERSION

[[ -f "${PATCH_FILE}" ]] || { echo "missing patch: ${PATCH_FILE}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd -P)"

BUILD_ROOT="$(mktemp -d /tmp/flashinfer-wheel.XXXXXX)"
trap 'rm -rf "${BUILD_ROOT}"' EXIT
SOURCE_DIR="${BUILD_ROOT}/flashinfer"

# Keep all build/JIT caches inside the disposable checkout and never install the wheel.
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${BUILD_ROOT}/cache"
export FLASHINFER_WORKSPACE_BASE="${BUILD_ROOT}/flashinfer-workspace"
export TORCH_EXTENSIONS_DIR="${BUILD_ROOT}/torch-extensions"

git clone --quiet --filter=blob:none "${FLASHINFER_REPOSITORY}" "${SOURCE_DIR}"
git -C "${SOURCE_DIR}" checkout --quiet --detach "${FLASHINFER_COMMIT}"
[[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${FLASHINFER_COMMIT}" ]] || {
  echo "unexpected FlashInfer commit" >&2
  exit 1
}
git -C "${SOURCE_DIR}" submodule update --init --recursive

git -C "${SOURCE_DIR}" apply --check "${PATCH_FILE}"
git -C "${SOURCE_DIR}" apply "${PATCH_FILE}"
git -C "${SOURCE_DIR}" diff --check -- \
  include/flashinfer/attention/variant_helper.cuh \
  include/flashinfer/attention/prefill.cuh

"${PYTHON}" -m pip wheel --no-deps --no-cache-dir --wheel-dir "${OUTPUT_DIR}" "${SOURCE_DIR}"
echo "FlashInfer wheel written to ${OUTPUT_DIR}"

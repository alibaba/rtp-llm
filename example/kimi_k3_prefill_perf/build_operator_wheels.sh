#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
work_parent="$(realpath -m -- "${K3_PERF_OPS_BUILD_ROOT:-${TMPDIR:-/tmp}}")"
wheel_dir="$(realpath -m -- "${K3_PERF_WHEEL_OUTPUT:-${script_dir}/wheels-built}")"
python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"

flash_kda_repo="https://github.com/MoonshotAI/FlashKDA.git"
flash_kda_ref="fa7eb894824a"
deep_gemm_repo="https://github.com/deepseek-ai/DeepGEMM.git"
deep_gemm_ref="f5a76426fa084087169693fd0cd815223576d6e9"
expected_wheels=(
  "deep_gemm-2.6.1-cp310-cp310-linux_x86_64.whl"
  "flash_kda-0.0.1-cp310-cp310-linux_x86_64.whl"
)

if [[ ! -x "${python_bin}" ]]; then
  echo "Python not found or not executable: ${python_bin}" >&2
  exit 2
fi
"${python_bin}" - <<'PY'
import sys

import torch

if sys.version_info[:2] != (3, 10):
    raise RuntimeError(f"operator wheels require Python 3.10, got {sys.version.split()[0]}")
if torch.__version__ != "2.11.0+cu130" or torch.version.cuda != "13.0":
    raise RuntimeError(
        "operator wheels require torch 2.11.0+cu130/CUDA 13.0, got "
        f"{torch.__version__}/{torch.version.cuda}"
    )
PY

# K3_PERF_OPS_BUILD_ROOT is a parent directory, never a disposable target.
# Allocate an exact private child so cleanup cannot erase a caller-owned path.
if [[ "${repo_root}/" == "${work_parent}/"* \
      || "${work_parent}/" == "${repo_root}/"* ]]; then
  echo "K3_PERF_OPS_BUILD_ROOT must not overlap the RTP checkout" >&2
  exit 2
fi
mkdir -p "${work_parent}" "${wheel_dir}"

# Refuse a stale mixed-version bundle without deleting caller-owned files.
shopt -s nullglob
existing_wheels=("${wheel_dir}"/*.whl)
shopt -u nullglob
for existing_wheel in "${existing_wheels[@]}"; do
  existing_name="${existing_wheel##*/}"
  expected=0
  for expected_wheel in "${expected_wheels[@]}"; do
    if [[ "${existing_name}" == "${expected_wheel}" ]]; then
      expected=1
      break
    fi
  done
  if [[ "${expected}" -ne 1 ]]; then
    echo "unexpected wheel already exists in output: ${existing_wheel}" >&2
    echo "remove it or select an empty K3_PERF_WHEEL_OUTPUT" >&2
    exit 2
  fi
done

work_root="$(mktemp -d "${work_parent}/kimi_k3_perf_ops_build.XXXXXX")"
cleanup() {
  case "${work_root}" in
    "${work_parent}"/kimi_k3_perf_ops_build.*)
      rm -rf -- "${work_root}"
      ;;
    *)
      echo "refusing to clean unexpected build path: ${work_root}" >&2
      ;;
  esac
}
trap cleanup EXIT

build_wheel_dir="${work_root}/wheelhouse"
mkdir -p "${build_wheel_dir}"

git clone --recursive "${flash_kda_repo}" "${work_root}/FlashKDA"
git -C "${work_root}/FlashKDA" checkout "${flash_kda_ref}"
git -C "${work_root}/FlashKDA" submodule update --init --recursive

git clone --recursive "${deep_gemm_repo}" "${work_root}/DeepGEMM"
git -C "${work_root}/DeepGEMM" checkout "${deep_gemm_ref}"
git -C "${work_root}/DeepGEMM" submodule update --init --recursive
git -C "${work_root}/DeepGEMM" apply --unidiff-zero \
  "${script_dir}/patches/deepgemm_cuda13_float_nttp.patch"

FLASH_KDA_CUDA_ARCHS=103a \
  "${python_bin}" -m pip wheel \
  --no-build-isolation --no-deps \
  --wheel-dir "${build_wheel_dir}" \
  "${work_root}/FlashKDA"

DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
  "${python_bin}" -m pip wheel \
  --no-build-isolation --no-deps \
  --wheel-dir "${build_wheel_dir}" \
  "${work_root}/DeepGEMM"

published_wheels=()
for wheel_name in "${expected_wheels[@]}"; do
  built_wheel="${build_wheel_dir}/${wheel_name}"
  if [[ ! -f "${built_wheel}" ]]; then
    echo "expected wheel was not produced: ${built_wheel}" >&2
    exit 1
  fi
  published_wheel="${wheel_dir}/${wheel_name}"
  install -m 0644 "${built_wheel}" "${published_wheel}"
  published_wheels+=("${published_wheel}")
done

sha256sum "${published_wheels[@]}"

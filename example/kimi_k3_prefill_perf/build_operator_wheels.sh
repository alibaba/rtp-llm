#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_root="${K3_PERF_OPS_BUILD_ROOT:-${TMPDIR:-/tmp}/kimi_k3_perf_ops_build}"
wheel_dir="${K3_PERF_WHEEL_OUTPUT:-${script_dir}/wheels-built}"
python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"

flash_kda_repo="https://github.com/MoonshotAI/FlashKDA.git"
flash_kda_ref="fa7eb894824a"
deep_gemm_repo="https://github.com/deepseek-ai/DeepGEMM.git"
deep_gemm_ref="f5a76426fa084087169693fd0cd815223576d6e9"

rm -rf "${work_root}"
mkdir -p "${work_root}" "${wheel_dir}"

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
  --wheel-dir "${wheel_dir}" \
  "${work_root}/FlashKDA"

DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
  "${python_bin}" -m pip wheel \
  --no-build-isolation --no-deps \
  --wheel-dir "${wheel_dir}" \
  "${work_root}/DeepGEMM"

sha256sum "${wheel_dir}"/*.whl

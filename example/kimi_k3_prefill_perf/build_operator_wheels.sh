#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_root="${K3_PERF_OPS_BUILD_ROOT:-${TMPDIR:-/tmp}/kimi_k3_perf_ops_build}"
wheel_dir="${K3_PERF_WHEEL_OUTPUT:-${script_dir}/wheels-built}"
python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"

flash_kda_repo="https://github.com/MoonshotAI/FlashKDA.git"
flash_kda_ref="fa7eb894824a"
cula_repo="git@gitlab.alibaba-inc.com:foundation_models/cuLA.git"
cula_ref="18543238473028425b81482e9e569161453bf2d6"
cula_base_wheel="cuda_linear_attention-0.1.2+rtp.aec3546.1-cp310-cp310-linux_x86_64.whl"
cula_base_url="https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp_llm/cula/cuda13_sm103/aec3546edb40b71032896cfd507202c73221c999/cuda_linear_attention-0.1.2%2Brtp.aec3546.1-cp310-cp310-linux_x86_64.whl"
cula_base_sha256="849c39d9b36d5c0ddd649435a1e4a59041fde7815db8814abfaebe3783b0f423"
deep_gemm_repo="https://github.com/deepseek-ai/DeepGEMM.git"
deep_gemm_ref="f5a76426fa084087169693fd0cd815223576d6e9"

rm -rf "${work_root}"
mkdir -p "${work_root}" "${wheel_dir}"

git clone --recursive "${flash_kda_repo}" "${work_root}/FlashKDA"
git -C "${work_root}/FlashKDA" checkout "${flash_kda_ref}"
git -C "${work_root}/FlashKDA" submodule update --init --recursive

git clone --recursive "${cula_repo}" "${work_root}/cuLA"
git -C "${work_root}/cuLA" checkout "${cula_ref}"
git -C "${work_root}/cuLA" submodule update --init --recursive
git -C "${work_root}/cuLA" apply \
  "${script_dir}/patches/cula_dynamic_checkpoint_pointer.patch"

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

OUTPUT_DIR="${wheel_dir}" PYTHON_BIN="${python_bin}" \
  "${work_root}/cuLA/scripts/build_fla_wheel.sh"

curl --fail --location --retry 3 \
  --output "${work_root}/${cula_base_wheel}" "${cula_base_url}"
printf '%s  %s\n' \
  "${cula_base_sha256}" "${work_root}/${cula_base_wheel}" |
  sha256sum --check -
OUTPUT_DIR="${wheel_dir}" PYTHON_BIN="${python_bin}" \
  "${work_root}/cuLA/scripts/build_prebuilt_sm103_wheel.sh" \
  "${work_root}/${cula_base_wheel}"

DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
  "${python_bin}" -m pip wheel \
  --no-build-isolation --no-deps \
  --wheel-dir "${wheel_dir}" \
  "${work_root}/DeepGEMM"

sha256sum "${wheel_dir}"/*.whl

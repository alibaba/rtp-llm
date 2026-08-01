#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
work_parent="$(realpath -m -- "${K3_PERF_OPS_BUILD_ROOT:-${TMPDIR:-/tmp}}")"
wheel_dir="$(realpath -m -- "${K3_PERF_WHEEL_OUTPUT:-${script_dir}/wheels-built}")"
python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"

flash_kda_repo="https://github.com/MoonshotAI/FlashKDA.git"
flash_kda_ref="fa7eb894824a"
cula_repo="git@gitlab.alibaba-inc.com:foundation_models/cuLA.git"
cula_ref="4db9fb97b791ace6b8c7709b9ead8016b9c0c72a"
cula_version="0.1.2+rtp.4db9fb9.1"
deep_gemm_repo="https://github.com/deepseek-ai/DeepGEMM.git"
deep_gemm_ref="f5a76426fa084087169693fd0cd815223576d6e9"

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

python_dir="$(cd -- "$(dirname -- "${python_bin}")" && pwd)"
if [[ ( -n "${CULA_CC:-}" && -z "${CULA_CXX:-}" ) \
      || ( -z "${CULA_CC:-}" && -n "${CULA_CXX:-}" ) ]]; then
  echo "CULA_CC and CULA_CXX must be set together" >&2
  exit 2
fi
if [[ -n "${CULA_CC:-}" ]]; then
  cula_cc="${CULA_CC}"
  cula_cxx="${CULA_CXX}"
elif [[ -x /opt/rh/gcc-toolset-13/root/usr/bin/gcc \
      && -x /opt/rh/gcc-toolset-13/root/usr/bin/g++ ]]; then
  cula_cc=/opt/rh/gcc-toolset-13/root/usr/bin/gcc
  cula_cxx=/opt/rh/gcc-toolset-13/root/usr/bin/g++
else
  cula_cc="$(command -v gcc)"
  cula_cxx="$(command -v g++)"
fi
cula_cuda_host_cxx="${CULA_CUDAHOSTCXX:-${cula_cxx}}"
for compiler in "${cula_cc}" "${cula_cxx}" "${cula_cuda_host_cxx}"; do
  if ! command -v "${compiler}" >/dev/null 2>&1; then
    echo "compiler not found: ${compiler}" >&2
    exit 2
  fi
done

# K3_PERF_OPS_BUILD_ROOT is a parent directory, never a disposable target.
# Allocate an exact private child so cleanup cannot erase a caller-owned path.
if [[ "${repo_root}/" == "${work_parent}/"* \
      || "${work_parent}/" == "${repo_root}/"* ]]; then
  echo "K3_PERF_OPS_BUILD_ROOT must not overlap the RTP checkout" >&2
  exit 2
fi
mkdir -p "${work_parent}" "${wheel_dir}"
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
  --wheel-dir "${build_wheel_dir}" \
  "${work_root}/FlashKDA"

OUTPUT_DIR="${build_wheel_dir}" PYTHON_BIN="${python_bin}" \
  "${work_root}/cuLA/scripts/build_fla_wheel.sh"

(
  cd "${work_root}/cuLA"
  # The SM103 template instantiations exceed the default 8 MiB compiler stack
  # on some build hosts.  A fresh native build is mandatory because b900f78
  # changes the C++ W/U address calculation; an older prebuilt .so is invalid.
  ulimit -s unlimited
  PATH="${python_dir}:${PATH}" \
    CC="${cula_cc}" \
    CXX="${cula_cxx}" \
    CUDAHOSTCXX="${cula_cuda_host_cxx}" \
    CULA_DISABLE_SM90=1 \
    CULA_DISABLE_SM100=1 \
    CULA_DISABLE_SM103=0 \
    MAX_JOBS="${CULA_MAX_JOBS:-2}" \
    NVCC_THREADS="${CULA_NVCC_THREADS:-4}" \
    "${python_bin}" setup.py build_ext --inplace

  source_epoch="$(git show -s --format=%ct HEAD)"
  PYTHONHASHSEED=0 \
    SOURCE_DATE_EPOCH="${source_epoch}" \
    CULA_PREBUILT_VERSION="${cula_version}" \
    "${python_bin}" scripts/setup_prebuilt_wheel.py \
      bdist_wheel --dist-dir "${build_wheel_dir}"
)

DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
  "${python_bin}" -m pip wheel \
  --no-build-isolation --no-deps \
  --wheel-dir "${build_wheel_dir}" \
  "${work_root}/DeepGEMM"

expected_wheels=(
  "cuda_linear_attention-${cula_version}-cp310-cp310-linux_x86_64.whl"
  "deep_gemm-2.6.1-cp310-cp310-linux_x86_64.whl"
  "flash_kda-0.0.1-cp310-cp310-linux_x86_64.whl"
  "flash_linear_attention-0.5.0+rtp.3a9ce1c.3-py3-none-any.whl"
)
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

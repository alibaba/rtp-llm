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
artlab_pypi_base_url="${K3_PERF_ARTLAB_PYPI_BASE_URL:-https://artlab.alibaba-inc.com/1/pypi/rtp_llm}"
cula_version="0.1.dev1+ed777e01.cu132"
cula_wheel="cuda_linear_attention-${cula_version}-cp310-cp310-linux_x86_64.whl"
cula_sha256="672f97b32469cd9c6a57bb91ed8a2cf59b9cddfea6b267ccd6a8a5d8af25489f"
fla_version="0.5.0+rtp.3a9ce1c.3"
fla_wheel="flash_linear_attention-${fla_version}-py3-none-any.whl"
fla_sha256="57f6f406f8f6125a760ca3de2e5e58a5feb2ee593de9c4de2208f0e13de098ed"
expected_wheels=(
  "${cula_wheel}"
  "deep_gemm-2.6.1-cp310-cp310-linux_x86_64.whl"
  "flash_kda-0.0.1-cp310-cp310-linux_x86_64.whl"
  "${fla_wheel}"
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

fetch_artlab_wheel() {
  local package_name="$1"
  local wheel_name="$2"
  local expected_sha256="$3"
  local wheel_path="${build_wheel_dir}/${wheel_name}"
  local download_path="${wheel_path}.download"
  local actual_sha256

  curl --fail --location --retry 3 --retry-delay 2 \
    --output "${download_path}" \
    "${artlab_pypi_base_url}/${package_name}/${wheel_name}"
  actual_sha256="$(sha256sum "${download_path}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "ArtLab wheel checksum mismatch: ${wheel_name}" >&2
    echo "expected ${expected_sha256}, got ${actual_sha256}" >&2
    exit 1
  fi
  mv -- "${download_path}" "${wheel_path}"
}

# cuLA and its matching FLA runtime are canonical, already-published artifacts.
# Pinning their exact hashes avoids rebuilding a different wheel under the same
# package version and makes this operator bundle reproducible.
fetch_artlab_wheel cuda_linear_attention "${cula_wheel}" "${cula_sha256}"
fetch_artlab_wheel flash_linear_attention "${fla_wheel}" "${fla_sha256}"

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

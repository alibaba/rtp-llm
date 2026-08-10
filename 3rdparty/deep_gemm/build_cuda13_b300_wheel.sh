#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"
work_parent="$(realpath -m -- "${DEEP_GEMM_BUILD_ROOT:-${TMPDIR:-/tmp}}")"
wheel_dir="$(realpath -m -- "${DEEP_GEMM_WHEEL_OUTPUT:-${work_parent}/rtp-deep-gemm-wheelhouse}")"
deep_gemm_repo="https://github.com/deepseek-ai/DeepGEMM.git"
deep_gemm_ref="f5a76426fa084087169693fd0cd815223576d6e9"
expected_wheel="deep_gemm-2.6.1-cp310-cp310-linux_x86_64.whl"

[[ "$(hostname)" == "e01-cn-qiz4s5sfe01" ]] || {
  echo "DeepGEMM must be built on L20-dev-115 (e01-cn-qiz4s5sfe01)" >&2
  exit 2
}
[[ "$(whoami)" == "luohaocheng.lhc" ]] || {
  echo "DeepGEMM must be built as luohaocheng.lhc" >&2
  exit 2
}
[[ -f /.dockerenv || -r /proc/1/cgroup ]] || {
  echo "run this build inside lhc_GPU" >&2
  exit 2
}
[[ -x "${python_bin}" ]] || {
  echo "Python not found or not executable: ${python_bin}" >&2
  exit 2
}
case "${repo_root}" in
  /data[0-9]*/* | /data/* | /ssd/*) ;;
  *) echo "refusing non-local source path: ${repo_root}" >&2; exit 2 ;;
esac
repo_fs="$(findmnt -T "${repo_root}" -n -o FSTYPE)"
case "${repo_fs}" in
  nfs* | cifs | smb* | fuse.*)
    echo "refusing network source filesystem: ${repo_fs}" >&2
    exit 2
    ;;
esac
if [[ "${repo_root}/" == "${work_parent}/"* \
      || "${work_parent}/" == "${repo_root}/"* \
      || "${repo_root}/" == "${wheel_dir}/"* ]]; then
  echo "build and wheel output directories must not overlap the RTP checkout" >&2
  exit 2
fi

"${python_bin}" - <<'PY'
import sys

import torch

if sys.version_info[:2] != (3, 10):
    raise RuntimeError(f"DeepGEMM wheel requires Python 3.10, got {sys.version.split()[0]}")
if torch.__version__ != "2.11.0+cu130" or torch.version.cuda != "13.0":
    raise RuntimeError(
        "DeepGEMM wheel requires torch 2.11.0+cu130/CUDA 13.0, got "
        f"{torch.__version__}/{torch.version.cuda}"
    )
PY

mkdir -p "${work_parent}" "${wheel_dir}"
work_root="$(mktemp -d "${work_parent}/rtp_deep_gemm_build.XXXXXX")"
cleanup() {
  case "${work_root}" in
    "${work_parent}"/rtp_deep_gemm_build.*) rm -rf -- "${work_root}" ;;
    *) echo "refusing to clean unexpected build path: ${work_root}" >&2 ;;
  esac
}
trap cleanup EXIT

git clone --recursive "${deep_gemm_repo}" "${work_root}/DeepGEMM"
git -C "${work_root}/DeepGEMM" checkout "${deep_gemm_ref}"
git -C "${work_root}/DeepGEMM" submodule update --init --recursive
git -C "${work_root}/DeepGEMM" apply --unidiff-zero \
  "${script_dir}/0003-k3-cuda13-float-nttp.patch"

DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
  "${python_bin}" -m pip wheel \
  --no-build-isolation --no-deps \
  --wheel-dir "${work_root}/wheelhouse" \
  "${work_root}/DeepGEMM"

built_wheel="${work_root}/wheelhouse/${expected_wheel}"
[[ -f "${built_wheel}" ]] || {
  echo "expected wheel was not produced: ${built_wheel}" >&2
  exit 1
}
published_wheel="${wheel_dir}/${expected_wheel}"
install -m 0644 "${built_wheel}" "${published_wheel}"
echo "source=${deep_gemm_repo}@${deep_gemm_ref}"
sha256sum "${published_wheel}"
echo "Update internal_source/deps and the CUDA13 lock if this artifact changes."

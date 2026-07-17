#!/usr/bin/env bash
# Reproducible, fail-closed install of the FA2 (vllm-xpu-kernels) kernel that
# XPU decode attention hard-requires.
#
# vllm-xpu-kernels is not on PyPI or the PyTorch XPU wheel index, so it
# cannot be pinned via deps/requirements_xpu.txt / requirements_lock_xpu.txt
# (see the comment block in deps/requirements_xpu.txt). This script is the
# single enforced install step build/CI/image pipelines must call instead of
# relying on developers to read a comment: it installs a pinned build and
# immediately re-runs the runtime preflight (check_fa2_requirements) so a
# missing/incompatible FA2 fails the BUILD, not the first decode request in
# production.
#
# Usage:
#   VLLM_XPU_KERNELS_INDEX_URL=<url> deps/install_xpu_fa2.sh
#   deps/install_xpu_fa2.sh --wheel /path/to/vllm_xpu_kernels-*.whl
#
# Env vars:
#   VLLM_XPU_KERNELS_VERSION    Pinned version spec (default: see below).
#   VLLM_XPU_KERNELS_INDEX_URL  Off-index find-links URL hosting the wheel.
#   PYTHON_BIN                  Python to install into (default: python3).
set -euo pipefail

# Keep in sync with the floor enforced at runtime in
# rtp_llm/models_py/modules/base/xpu/vllm_xpu_ops.py (_FA2_MIN_VERSION) and
# documented in deps/requirements_xpu.txt.
VLLM_XPU_KERNELS_VERSION="${VLLM_XPU_KERNELS_VERSION:->=0.1.10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

WHEEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheel)
      WHEEL="$2"
      shift 2
      ;;
    *)
      echo "install_xpu_fa2.sh: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$WHEEL" ]]; then
  echo "install_xpu_fa2.sh: installing pinned wheel: $WHEEL"
  "$PYTHON_BIN" -m pip install --no-deps "$WHEEL"
elif [[ -n "${VLLM_XPU_KERNELS_INDEX_URL:-}" ]]; then
  echo "install_xpu_fa2.sh: installing vllm-xpu-kernels${VLLM_XPU_KERNELS_VERSION} from ${VLLM_XPU_KERNELS_INDEX_URL}"
  "$PYTHON_BIN" -m pip install --no-deps \
    "vllm-xpu-kernels${VLLM_XPU_KERNELS_VERSION}" \
    --find-links "${VLLM_XPU_KERNELS_INDEX_URL}"
else
  echo "install_xpu_fa2.sh: no --wheel and no VLLM_XPU_KERNELS_INDEX_URL set." >&2
  echo "Set one of them to a build matching the installed torch-xpu wheel." >&2
  exit 2
fi

echo "install_xpu_fa2.sh: verifying FA2 preflight (check_fa2_requirements)..."
"$PYTHON_BIN" -c '
import sys
from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import check_fa2_requirements
err = check_fa2_requirements()
if err:
    print(f"install_xpu_fa2.sh: FA2 preflight FAILED after install: {err}", file=sys.stderr)
    sys.exit(1)
print("install_xpu_fa2.sh: FA2 preflight OK")
'

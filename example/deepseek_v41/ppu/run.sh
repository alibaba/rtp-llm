#!/usr/bin/env bash
set -euo pipefail

# Source in this shell, before Python imports any accelerator packages.
v41_sdk_root="${PPU_HOME:-/usr/local/PPU_SDK}"
source "$v41_sdk_root/envsetup.sh" >/dev/null
export PPU_HOME="$v41_sdk_root"
export CUDA_HOME="$v41_sdk_root/CUDA_SDK"
export CUDA_PATH="$CUDA_HOME"
export PATH="/opt/conda312/bin:$v41_sdk_root/ppu-smi/bin:$PATH"
export LD_LIBRARY_PATH="$v41_sdk_root/pccl/nccl_wrapper/lib:$v41_sdk_root/pccl/lib:/opt/conda312/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec /opt/conda312/bin/python -m rtp_llm.tools.v41.launch "$@"

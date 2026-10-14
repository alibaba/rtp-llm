#!/usr/bin/env bash
# Run after installing RTP's Python dependencies, using the service's Python
# environment (PYTHON=/path/to/python if needed). Existing dependency locks
# contain older FlashMLA wheels; install this pinned source build last.
# Requires a CUDA-enabled PyTorch, CUDA >=12.9, a C++20 compiler, git and ninja.
# Builds SM100a/SM103a without allocating a GPU or downloading model weights.
set -euo pipefail

mega_python="${PYTHON:-python3}"
mega_revision=063a9f05aaf86d25c3da213a6e79cd5c879f99fe
export FLASH_MLA_DISABLE_SM90=1
export FLASH_MLA_DISABLE_SM100=0
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-4}"

"${mega_python}" -m pip install --no-deps --no-build-isolation --force-reinstall \
    "flash-mla @ git+https://github.com/deepseek-ai/FlashMLA.git@${mega_revision}"

"${mega_python}" - <<'PY'
from flash_mla import cuda, fused_norm_rope_attn_rope_cast

for name in (
    "fused_norm_rope_attn_rope_cast_fwd",
    "fused_norm_rope_attn_rope_cast_decode",
    "permute_q_b_proj",
    "permute_wv_proj",
):
    if not callable(getattr(cuda, name, None)):
        raise ImportError(f"FlashMLA Mega Attention API missing: {name}")
print("FlashMLA Mega Attention prefill/decode APIs verified")
PY

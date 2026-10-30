# DeepGEMM MegaMoE for PPU

This directory builds the PPU-only `deep-gemm-mega` distribution. It starts
from the public Alibaba/T-Head DeepGEMM-for-sail source pinned in `UPSTREAM`
and applies the local MegaMoE patch.

The resulting Python package is `deep_gemm_mega`; the regular `deep_gemm`
package remains the official PPU wheel used by HC and other PPU kernels.

The patch is intentionally applied at source-build time. A prebuilt
`deep_gemm` wheel cannot gain the MegaMoE native extension by changing its
metadata or replacing a header file.

Build requirements are the PPU SDK, the matching CUDA-ABI PyTorch, git, and
the Python build frontend available in the target PPU image.

Build in the target PPU environment (including `setuptools-scm`, `wheel`,
`ninja`, and `packaging`):

```bash
PYTHON_BIN=python3 DG_TORCH_CUDA_ABI=1 bash 3rdparty/deep_gemm_mega/build_wheel.sh
python3 -m pip install --no-deps --force-reinstall 3rdparty/deep_gemm_mega/dist/deep_gemm_mega-*.whl
python3 3rdparty/deep_gemm_mega/check_install.py
```

The build runs source/package contracts before native compilation. Install the
wheel alongside the official PPU `deep_gemm` wheel in PPU images only. The install
check verifies separate file ownership, both import orders, MegaMoE entry points,
and the packaged LUT. It does not validate M890P kernel numerics or performance.

The upstream pin and MegaMoE patch are taken from PR #1561 at
`53bf97072085d124879554e19512647c14825f56`. Build-time adjustments keep Python
imports and package-data paths consistent after renaming, and preserve separate
JIT caches (`DG_MEGA_JIT_CACHE_DIR`) and LUT selection (`RTP_PPU_MEGA_LUT_PATH`).

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

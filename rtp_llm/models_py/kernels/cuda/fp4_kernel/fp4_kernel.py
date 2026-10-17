import logging
from typing import Tuple

import torch

from rtp_llm.models_py.utils.arch import get_sm, is_cuda

nvfp4_available = False
if is_cuda() and get_sm()[0] >= 10:
    try:
        from rtp_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant

        nvfp4_available = True
    except ImportError:
        # The rtp-kernel build in this environment has no nvfp4 module; keep
        # the rest of the CUDA linear package importable and skip FP4.
        logging.info("rtp_kernel.nvfp4 unavailable; disabling NVFP4 kernels")
else:
    logging.info("skip import fp4 kernel from rtp_kernel.nvfp4 for non cuda platform")

logger = logging.getLogger(__name__)


def _require_nvfp4() -> None:
    if not nvfp4_available:
        raise RuntimeError("rtp_kernel.nvfp4 is not available in this environment")


def cutlass_scaled_fp4_mm_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    _require_nvfp4()
    return cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, out_dtype)


def scaled_fp4_quant_wrapper(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dim to NVFP4 with swizzled UE4M3 scales."""
    _require_nvfp4()
    return scaled_fp4_quant(input, input_global_scale)

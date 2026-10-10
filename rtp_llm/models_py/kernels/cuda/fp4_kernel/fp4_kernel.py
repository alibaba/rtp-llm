import logging
from typing import Tuple

import torch

from rtp_llm.models_py.utils.arch import get_sm, is_cuda

if is_cuda() and get_sm()[0] >= 10:
    from rtp_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant
else:
    logging.info("skip import fp4 kernel from rtp_kernel.nvfp4 for non cuda platform")

logger = logging.getLogger(__name__)


def cutlass_scaled_fp4_mm_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, out_dtype)


def scaled_fp4_quant_wrapper(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dim to NVFP4 with swizzled UE4M3 scales."""
    return scaled_fp4_quant(input, input_global_scale)

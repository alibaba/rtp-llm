import logging
from typing import Tuple

import torch

from rtp_llm.models_py.utils.arch import get_sm, is_cuda

cutlass_scaled_fp4_mm = None
scaled_fp4_quant = None

if is_cuda() and torch.cuda.is_available() and get_sm()[0] >= 10:
    from rtp_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant
else:
    logging.info("skip import fp4 kernel from rtp_kernel.nvfp4 for non cuda platform")


def is_legacy_cutlass_fp4_available() -> bool:
    """Whether the CUTLASS FP4 provider used by sgl_cutlass is available."""
    return callable(cutlass_scaled_fp4_mm)


def cutlass_scaled_fp4_mm_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if cutlass_scaled_fp4_mm is None:
        raise RuntimeError(
            "CUTLASS FP4 GEMM requires the rtp_kernel.nvfp4 CUDA provider"
        )
    return cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, out_dtype)


def scaled_fp4_quant_wrapper(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dim to NVFP4 with swizzled UE4M3 scales."""
    if scaled_fp4_quant is None:
        raise RuntimeError(
            "FP4 quantization requires the rtp_kernel.nvfp4 CUDA provider"
        )
    return scaled_fp4_quant(input, input_global_scale)

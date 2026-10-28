"""Fuse a dense SwiGLU producer into FlashInfer's NVFP4 quantizer."""

import os

import torch
from rtp_llm.models_py.utils.arch import get_sm

try:
    from flashinfer import silu_and_mul_scaled_nvfp4_experts_quantize
except ImportError:
    silu_and_mul_scaled_nvfp4_experts_quantize = None

_FUSED_PRODUCER_ENABLED = os.getenv("RTP_LLM_SM120_FUSION", "0") == "1"


def silu_fp4_quant(input, scale, mask):
    """Return dense packed data/scales, or None to retain the old producer.

    A single expert is a dense matrix. Undo only the grouped GEMM views: the
    physical 128x4 scale layout already matches the dense FP4 GEMM contract.
    FlashInfer rounds the SiLU product to BF16 before computing block scales.
    """
    if (
        not _FUSED_PRODUCER_ENABLED
        or silu_and_mul_scaled_nvfp4_experts_quantize is None
        or not input.is_cuda
        or input.ndim != 2
        or input.dtype != torch.bfloat16
        or input.shape[0] == 0
        or input.numel() >= 2**31
        or input.shape[1] not in (17408, 34816)
        or not input.is_contiguous()
        or get_sm(input.device) != (12, 0)
        or scale.device != input.device
        or scale.dtype != torch.float32
        or scale.numel() != 1
        or not scale.is_contiguous()
        or mask is None
        or mask.device != input.device
        or mask.dtype != torch.int32
        or mask.shape != (1,)
    ):
        return None
    m, two_k = input.shape
    k = two_k // 2
    q, scales = silu_and_mul_scaled_nvfp4_experts_quantize(
        input.unsqueeze(0), mask, scale.reshape(1)
    )
    # FlashInfer 0.6.9 grouped output; retain the original producer if changed.
    if q.shape != (m, k // 2, 1) or scales.shape != (
        32,
        4,
        (m + 127) // 128,
        4,
        (k // 16 + 3) // 4,
        1,
    ):
        return None
    q = q.permute(2, 0, 1).reshape(m, k // 2)
    scales = scales.permute(5, 2, 4, 0, 1, 3).reshape(-1, k // 16)
    return q, scales.view(torch.uint8)

import os

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


def prepare_bf16_fp32_router_weight(weight: torch.Tensor) -> torch.Tensor | None:
    """Pack immutable [K,E] weights before capture; reject any BF16 rounding loss.

    Validation synchronizes once at initialization. Replacing or mutating the
    original weight requires preparing a new packed tensor before the next call.
    """
    if os.environ.get("RTP_LLM_HY4_ROUTER_BF16_FP32", "1") != "1":
        return None
    if not (
        weight.is_cuda
        and torch.version.hip is None
        and weight.ndim == 2
        and weight.dtype in (torch.float32, torch.bfloat16)
        and weight.is_contiguous()
        and not weight.requires_grad
    ):
        return None
    if not torch.isfinite(weight).all().item():
        return None
    packed = weight.t().to(torch.bfloat16).contiguous()
    if weight.dtype == torch.float32 and not torch.equal(weight, packed.t().float()):
        return None
    return packed


def bf16_fp32_router_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    packed_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Use lossless BF16 operands with FP32 output; reduction order may differ."""
    if (
        packed_weight is not None
        and torch.version.hip is None
        and hidden.is_cuda
        and hidden.ndim == 2
        and hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and not hidden.requires_grad
        and not weight.requires_grad
        and not packed_weight.requires_grad
        and hidden.device == weight.device == packed_weight.device
        and weight.ndim == 2
        and weight.dtype in (torch.float32, torch.bfloat16)
        and packed_weight.dtype == torch.bfloat16
        and packed_weight.is_contiguous()
        and packed_weight.shape == (weight.shape[1], hidden.shape[1])
        and hidden.shape[1] == weight.shape[0]
        and 0 < hidden.shape[0] <= 128
    ):
        return rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(hidden, packed_weight)
    fallback_weight = weight.float() if weight.dtype == torch.bfloat16 else weight
    return torch.matmul(hidden.float(), fallback_weight)

"""Qwen4 BF16 MoE decode specializations and their kernel support domain."""

import inspect
from dataclasses import dataclass

import torch
import triton.language as tl


@dataclass(frozen=True)
class DecodeCapabilities:
    alignment: bool
    tuned_tiles: bool
    small_gemv: bool


def masked_histogram_available() -> bool:
    try:
        return "mask" in inspect.signature(getattr(tl, "histogram", None)).parameters
    except (TypeError, ValueError):
        return False


def decode_capabilities(
    weight, top_k: int, ep_size: int, model_type: str = ""
) -> DecodeCapabilities:
    if model_type not in ("qwen4_exp", "qwen4_exp_mtp"):
        return DecodeCapabilities(False, False, False)
    cuda_bf16 = (
        weight.dtype == torch.bfloat16 and weight.is_cuda and torch.version.hip is None
    )
    arch = torch.cuda.get_device_capability(weight.device) if cuda_bf16 else None
    alignment = cuda_bf16 and top_k == 10 and masked_histogram_available() and (
        weight.shape[0] == 512
        or (arch == (9, 0) and ep_size == 2 and weight.shape[0] == 256)
    )
    tuned = (
        cuda_bf16
        and tuple(weight.shape) == (512, 160, 2560)
        and top_k == 10
        and arch == (12, 0)
    )
    return DecodeCapabilities(alignment, tuned, tuned and ep_size == 1)

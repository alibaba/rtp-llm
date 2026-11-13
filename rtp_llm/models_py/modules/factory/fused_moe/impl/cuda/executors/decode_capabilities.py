"""BF16 MoE decode specializations selected only by their kernel support domain."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecodeCapabilities:
    alignment: bool
    tuned_tiles: bool
    small_gemv: bool


def decode_capabilities(weight, top_k: int, ep_size: int) -> DecodeCapabilities:
    cuda_bf16 = (
        weight.dtype == torch.bfloat16 and weight.is_cuda and torch.version.hip is None
    )
    alignment = cuda_bf16 and weight.shape[0] == 512 and top_k == 10
    tuned = (
        cuda_bf16
        and tuple(weight.shape) == (512, 160, 2560)
        and top_k == 10
        and torch.cuda.get_device_capability(weight.device) == (12, 0)
    )
    return DecodeCapabilities(alignment, tuned, tuned and ep_size == 1)

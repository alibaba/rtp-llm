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
    arch = torch.cuda.get_device_capability(weight.device) if cuda_bf16 else None
    alignment = cuda_bf16 and top_k == 10 and (
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

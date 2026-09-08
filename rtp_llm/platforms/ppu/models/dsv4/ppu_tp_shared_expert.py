"""Private TP slices of the loader's replicated block-FP8 shared expert."""

import torch

from .ppu_shared_expert import PpuSharedExpert


def shard_shared_weights(weights, *, dim, inter_dim, tp_size, tp_rank):
    if tp_size != 4 or not 0 <= tp_rank < tp_size:
        raise ValueError("PPU shared expert requires TP4 and rank in [0, 4)")
    if inter_dim <= 0 or inter_dim % (128 * tp_size) or dim <= 0 or dim % 128:
        raise ValueError("Shared expert dimensions must preserve block-128 scales")
    expected = {
        "w13_w": ((2 * inter_dim, dim), torch.float8_e4m3fn),
        "w13_s": ((2 * inter_dim // 128, dim // 128), torch.float8_e8m0fnu),
        "w2_w": ((dim, inter_dim), torch.float8_e4m3fn),
        "w2_s": ((dim // 128, inter_dim // 128), torch.float8_e8m0fnu),
    }
    device = weights["w13_w"].device
    for key, (shape, dtype) in expected.items():
        tensor = weights[key]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{key} requires {shape}/{dtype}, got {tensor.shape}/{tensor.dtype}"
            )
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(
                "Shared expert weights/scales must be contiguous on one device"
            )
    local = inter_dim // tp_size
    result = {}
    # Copy only this rank's quarter. Byte views preserve both FP8 encodings,
    # including scale payloads; never modify the loader's source tensors.
    for key, block in (("w13_w", 1), ("w13_s", 128)):
        src = weights[key]
        width, half = local // block, inter_dim // block
        start = tp_rank * width
        result[key] = torch.cat(
            (
                src[start : start + width].view(torch.uint8),
                src[half + start : half + start + width].view(torch.uint8),
            ),
            dim=0,
        ).view(src.dtype)
    for key, block in (("w2_w", 1), ("w2_s", 128)):
        src = weights[key]
        width = local // block
        result[key] = (
            src[:, tp_rank * width : (tp_rank + 1) * width]
            .view(torch.uint8)
            .contiguous()
            .view(src.dtype)
        )
    return result


class PpuTPSharedExpert(PpuSharedExpert):
    def __init__(
        self,
        dim,
        inter_dim,
        expert_weights,
        *,
        tp_size,
        tp_rank,
        swiglu_limit=0.0,
        platform_provider,
    ):
        local_weights = shard_shared_weights(
            expert_weights,
            dim=dim,
            inter_dim=inter_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        super().__init__(
            dim,
            inter_dim // tp_size,
            local_weights,
            swiglu_limit=swiglu_limit,
            platform_provider=platform_provider,
            sglang_moe=True,
        )
        self.tp_size, self.tp_rank = tp_size, tp_rank

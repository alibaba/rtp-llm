"""Module-owned ModelOpt NVFP4 reference quantization (no loader dependency)."""

import torch


def quantize_expert_to_nvfp4(weight: torch.Tensor, *, amax=None):
    """Quantize [N,K] with 16-element blocks and round-to-nearest-even E2M1.

    An optional full-checkpoint amax preserves quantize-before-TP semantics
    without retaining a full expert on every tensor-parallel rank.
    """
    if weight.ndim != 2 or weight.shape[1] % 16:
        raise ValueError("NVFP4 requires a 2D weight with K divisible by 16")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Online NVFP4 requires floating-point checkpoint weights")
    w = weight.float()
    if not torch.isfinite(w).all():
        raise ValueError("Online NVFP4 weights must be finite")
    maximum = w.abs().amax() if amax is None else amax.to(w.device)
    scale_2 = maximum.clamp_min(1e-8).float() / (6.0 * 448.0)
    blocks = w.reshape(w.shape[0], w.shape[1] // 16, 16)
    scale = blocks.abs().amax(-1) / (6.0 * scale_2)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    scale = scale.to(torch.float8_e4m3fn)
    values = (blocks / (scale.float() * scale_2).unsqueeze(-1)).clamp(-6, 6)
    values = values.reshape_as(w)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=w.device)
    magnitude = values.abs().contiguous()
    ordinal = torch.searchsorted(bounds, magnitude, out_int32=True).to(torch.uint8)
    ties = (magnitude.unsqueeze(-1) == bounds[[1, 3, 5]]).any(-1).to(torch.uint8)
    ordinal = ((values < 0).to(torch.uint8) << 3) + ordinal + ties
    packed = (ordinal[:, 1::2] << 4) | ordinal[:, ::2]
    return packed.contiguous(), scale.contiguous(), scale_2

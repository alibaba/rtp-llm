"""Portable copy of the RTP NVFP4 block16 reference for the matched baseline."""

import torch

FLOAT4_E2M1_MAX = 6.0

FLOAT8_E4M3_MAX = 448.0

NVFP4_BLOCK_SIZE = 16

_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

_E2M1_ODD_BOUNDS = _E2M1_BOUNDS[[1, 3, 5]]


def _cast_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Map float values in [-6, 6] to packed E2M1 ordinals (uint8 in [0, 15])."""
    device = x.device
    sign_bit = (x < 0).to(torch.uint8)
    abs_x = x.abs()
    bounds = _E2M1_BOUNDS.to(device=device, dtype=abs_x.dtype)
    ord_ = torch.searchsorted(bounds, abs_x, out_int32=True).to(torch.uint8)
    odd_bounds = _E2M1_ODD_BOUNDS.to(device=device, dtype=abs_x.dtype)
    equals_odd = torch.any(abs_x.unsqueeze(-1) == odd_bounds, dim=-1).to(torch.uint8)
    return (sign_bit << 3) + ord_ + equals_odd


def quantize_expert_to_nvfp4(
    weight: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
):
    """Quantize a 2D BF16/FP16 weight tensor to NVFP4 (per-block fp8 + per-tensor fp32).

    Args:
        weight: shape ``[N, K]`` with K divisible by ``block_size``.

    Returns:
        Tuple of (packed uint8 ``[N, K // 2]``, per-block fp8 scale ``[N, K // block_size]``,
        per-tensor fp32 scale_2 scalar).
    """
    assert weight.dim() == 2, f"expected 2D, got {tuple(weight.shape)}"
    N, K = weight.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"

    w_fp32 = weight.to(torch.float32)
    amax = w_fp32.abs().amax().clamp_min(1e-8)
    weight_scale_2 = (amax / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)).to(torch.float32)

    blocked = w_fp32.view(N, K // block_size, block_size)
    per_block_amax = blocked.abs().amax(dim=-1)  # [N, K//block_size]
    per_block_scale = per_block_amax / (FLOAT4_E2M1_MAX * weight_scale_2)
    per_block_scale = torch.where(
        per_block_scale == 0,
        torch.ones_like(per_block_scale),
        per_block_scale,
    )
    per_block_scale_fp8 = per_block_scale.to(torch.float8_e4m3fn)

    # Re-dequant fp8 scale to use the same value the kernel will see.
    effective_block_scale = per_block_scale_fp8.to(torch.float32)
    combined = (effective_block_scale * weight_scale_2).unsqueeze(-1)  # [N, K//bs, 1]
    scaled = blocked / combined
    scaled = scaled.clamp(-FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(N, K)

    q = _cast_to_e2m1(scaled)  # uint8 [N, K]
    packed = (q[..., 1::2].to(torch.uint8) << 4) | q[..., 0::2].to(torch.uint8)
    packed = packed.contiguous()  # [N, K // 2]
    return packed, per_block_scale_fp8.contiguous(), weight_scale_2

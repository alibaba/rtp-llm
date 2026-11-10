"""Indexer K RoPE and Hadamard without producing an unused query tensor."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_hadamard_k(
    KEY,
    POSITIONS,
    COS_SIN,
    OUTPUT,
    KEY_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.arange(0, 128)
    position = tl.load(POSITIONS + row)
    value = tl.load(KEY + row * KEY_STRIDE + index).to(tl.float32)
    if IS_NEOX:
        partner_index = index ^ 32
        trig_index = index & 31
        upper = index >= 32
    else:
        partner_index = index ^ 1
        trig_index = (index // 2) & 31
        upper = (index & 1) != 0
    partner = tl.load(KEY + row * KEY_STRIDE + partner_index).to(tl.float32)
    cos = tl.load(COS_SIN + position * CACHE_STRIDE + trig_index).to(tl.float32)
    sin = tl.load(COS_SIN + position * CACHE_STRIDE + trig_index + 32).to(tl.float32)
    rotated = tl.fma(value, cos, tl.where(upper, partner * sin, -partner * sin))
    # Preserve the BF16 rounding between RoPE and Hadamard in fused QK.
    transformed = tl.where(index < 64, rotated.to(tl.bfloat16).to(tl.float32), value)
    for shift in tl.static_range(7):
        stride = 1 << shift
        partner = tl.gather(transformed, index ^ stride, 0)
        transformed = tl.where(
            (index & stride) != 0, partner - transformed, transformed + partner
        )
    tl.store(
        OUTPUT + row * 128 + index,
        (transformed * (128**-0.5)).to(tl.bfloat16),
    )


def fused_indexer_k_rope(key, positions, cos_sin_cache, is_neox_style=False):
    """Match the 128-dim BF16 K output of the existing fused QK kernel."""
    assert key.ndim == 2 and key.shape[1] == 128 and key.stride(1) == 1
    assert key.dtype == torch.bfloat16 and cos_sin_cache.dtype == torch.float32
    assert cos_sin_cache.shape[-1] == 64 and cos_sin_cache.stride(-1) == 1
    assert positions.ndim == 1 and positions.numel() == key.shape[0]
    assert positions.is_contiguous()
    output = torch.empty_like(key, memory_format=torch.contiguous_format)
    if key.shape[0]:
        _rope_hadamard_k[(key.shape[0],)](
            key,
            positions,
            cos_sin_cache,
            output,
            key.stride(0),
            cos_sin_cache.stride(0),
            is_neox_style,
            num_warps=4,
        )
    return output

"""Explicit mRoPE for kernel versions that do not implement the current layout."""

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_mrope_qk_inplace(
    QKV,
    POSITIONS,
    QKV_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    POS_AXIS_STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAIRS: tl.constexpr,
    ROPE_BASE: tl.constexpr,
    ROPE_SCALE: tl.constexpr,
    T_PAIRS: tl.constexpr,
    H_PAIRS: tl.constexpr,
    W_PAIRS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    pair = tl.arange(0, BLOCK)
    valid = pair < PAIRS
    if INTERLEAVED:
        axis = tl.where((pair % 3 == 1) & (pair < 3 * H_PAIRS), 1, 0)
        axis = tl.where((pair % 3 == 2) & (pair < 3 * W_PAIRS), 2, axis)
    else:
        axis = tl.where(pair < T_PAIRS, 0, tl.where(pair < T_PAIRS + H_PAIRS, 1, 2))
    position = tl.load(
        POSITIONS + token * POS_STRIDE + axis * POS_AXIS_STRIDE,
        mask=valid,
        other=0,
    ).to(tl.float32)
    inv_freq = tl.exp(-tl.log(ROPE_BASE) * pair.to(tl.float32) / PAIRS)
    angle = position * inv_freq / ROPE_SCALE
    cos, sin = tl.cos(angle), tl.sin(angle)
    offset = token * QKV_STRIDE + head * HEAD_DIM + pair
    low = tl.load(QKV + offset, mask=valid, other=0).to(tl.float32)
    high = tl.load(QKV + offset + PAIRS, mask=valid, other=0).to(tl.float32)
    tl.store(QKV + offset, low * cos - high * sin, mask=valid)
    tl.store(QKV + offset + PAIRS, high * cos + low * sin, mask=valid)


def is_supported(qkv: torch.Tensor, head_dim: int, rope_dim: int) -> bool:
    return (
        qkv.is_cuda
        and qkv.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and qkv.ndim == 2
        and qkv.stride(1) == 1
        and 0 < rope_dim <= head_dim
        and rope_dim % 2 == 0
    )


def apply_mrope_qk_inplace(
    qkv: torch.Tensor,
    position_ids: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    head_dim: int,
    rope_config,
) -> torch.Tensor:
    """Rotate packed Q/K using explicit T/H/W positions; leave V and tail intact."""
    rope_dim = rope_config.dim
    if not is_supported(qkv, head_dim, rope_dim):
        raise ValueError("unsupported QKV layout or rotary dimension for legacy mRoPE")
    if qkv.size(1) != (head_num + 2 * kv_head_num) * head_dim:
        raise ValueError("QKV width does not match attention head dimensions")
    if position_ids.numel() != qkv.size(0) * 3:
        raise ValueError("mRoPE requires three position IDs for every QKV token")
    sections = (
        rope_config.mrope_dim1,
        rope_config.mrope_dim2,
        rope_config.mrope_dim3,
    )
    pairs = rope_dim // 2
    if any(section < 0 for section in sections) or sum(sections) != pairs:
        raise ValueError("mRoPE sections must sum to half the rotary dimension")
    interleaved = rope_config.mrope_interleaved
    if interleaved and (sections[1] > (pairs + 1) // 3 or sections[2] > pairs // 3):
        raise ValueError("mRoPE sections exceed interleaved H/W capacity")
    if rope_config.base <= 0 or rope_config.scale <= 0:
        raise ValueError("mRoPE base and scale must be positive")
    positions = position_ids.to(device=qkv.device, non_blocking=True).reshape(-1, 3)
    _apply_mrope_qk_inplace[(qkv.size(0), head_num + kv_head_num)](
        qkv,
        positions,
        qkv.stride(0),
        positions.stride(0),
        positions.stride(1),
        head_dim,
        pairs,
        float(rope_config.base),
        float(rope_config.scale),
        *sections,
        interleaved,
        triton.next_power_of_2(pairs),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return qkv

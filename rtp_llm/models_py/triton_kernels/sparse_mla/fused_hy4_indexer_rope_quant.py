"""HY4 Indexer RoPE, UE8M0 quantization, and paged-cache fusion.

The HY4 checkpoint is adapted at load time from ``[NoPE, RoPE]`` to RTP's
``[RoPE, NoPE]`` Indexer layout.  This kernel therefore rotates the leading
64 channels, quantizes every 128-channel Q head, and writes the single K head
directly into the existing FP8 Indexer cache.

This is deliberately a HY4-only decode kernel.  The shared DSV3.2/GLM5
Hadamard path and ordinary prefill path remain unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_HY4_HEAD_DIM = 128
_HY4_ROPE_DIM = 64
_HY4_INDEX_HEADS = 32
_Q_HEAD_TILE = 8


@triton.jit
def _ieee_rn_div_f32(x, y):
    """IEEE round-to-nearest-even FP32 division, matching the CUDA baseline."""
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _round_up_ue8m0(s):
    """Return the FP32 value represented by ceil(log2(s)) in UE8M0."""
    bits = s.to(tl.int32, bitcast=True)
    mantissa_nonzero = (bits & 0x7FFFFF) != 0
    exponent = ((bits >> 23) & 0xFF) + tl.where(mantissa_nonzero, 1, 0)
    exponent = tl.minimum(tl.maximum(exponent, 0), 255)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _fused_hy4_indexer_rope_quant_kernel(
    Q,
    K,
    POSITIONS,
    COS_SIN_CACHE,
    SLOT_MAPPING,
    Q_FP8,
    Q_SCALE,
    K_CACHE_FP8,
    K_CACHE_FP32,
    RAW_HEAD_GATE,
    HEAD_WEIGHTS,
    head_scale,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_cos_sin,
    stride_q_fp8_token,
    stride_q_fp8_head,
    stride_q_scale_token,
    stride_cache_fp8_page,
    stride_cache_fp32_page,
    CACHE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT_DIM: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    HEAD_TILES: tl.constexpr,
    TILE_OFFSET: tl.constexpr,
    FOLD_HEAD_GATE: tl.constexpr,
    ROUND_HEAD_GATE_BF16: tl.constexpr,
):
    """Tiled Q programs plus one K-cache program, all in a single launch."""
    token = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1) + TILE_OFFSET
    dims = tl.arange(0, HEAD_DIM)
    position = tl.load(POSITIONS + token).to(tl.int64)
    half_rope: tl.constexpr = ROT_DIM // 2
    rope_mask = dims < ROT_DIM

    if IS_NEOX:
        first_half = dims < half_rope
        partner_dims = tl.where(first_half, dims + half_rope, dims - half_rope)
        cache_dims = dims % half_rope
        rope_sign = tl.where(first_half, -1.0, 1.0)
    else:
        even = (dims & 1) == 0
        partner_dims = tl.where(even, dims + 1, dims - 1)
        cache_dims = dims // 2
        rope_sign = tl.where(even, -1.0, 1.0)

    cos = tl.load(
        COS_SIN_CACHE + position * stride_cos_sin + cache_dims,
        mask=rope_mask,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        COS_SIN_CACHE + position * stride_cos_sin + half_rope + cache_dims,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)

    if tile < HEAD_TILES:
        heads = tile * HEAD_TILE + tl.arange(0, HEAD_TILE)
        q_offsets = (
            token * stride_q_token + heads[:, None] * stride_q_head + dims[None, :]
        )
        q = tl.load(Q + q_offsets).to(tl.float32)
        q_partner = tl.load(
            Q
            + token * stride_q_token
            + heads[:, None] * stride_q_head
            + partner_dims[None, :],
            mask=rope_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_rotated = tl.extra.libdevice.fma_rn(
            q,
            cos[None, :],
            rope_sign[None, :] * q_partner * sin[None, :],
        )
        # FlashInfer stores RoPE to BF16 before quantization reads it.
        q_rotated = q_rotated.to(tl.bfloat16).to(tl.float32)
        q = tl.where(rope_mask[None, :], q_rotated, q)

        q_amax = tl.maximum(tl.max(tl.abs(q), axis=1), 1.0e-10)
        q_scale_init = _ieee_rn_div_f32(q_amax, 448.0)
        q_scale = _round_up_ue8m0(tl.maximum(q_scale_init, 1.0e-10))
        q_quant = tl.clamp(
            _ieee_rn_div_f32(q, tl.broadcast_to(q_scale[:, None], q.shape)),
            -448.0,
            448.0,
        ).to(Q_FP8.dtype.element_ty)
        tl.store(
            Q_FP8
            + token * stride_q_fp8_token
            + heads[:, None] * stride_q_fp8_head
            + dims[None, :],
            q_quant,
        )
        tl.store(
            Q_SCALE + token * stride_q_scale_token + heads,
            q_scale,
        )
        if FOLD_HEAD_GATE:
            raw = tl.load(RAW_HEAD_GATE + token * 32 + heads)
            if ROUND_HEAD_GATE_BF16:
                raw = raw.to(tl.bfloat16).to(tl.float32)
            tl.store(HEAD_WEIGHTS + token * 32 + heads, (raw * q_scale) * head_scale)
    else:
        slot = tl.load(SLOT_MAPPING + token).to(tl.int64)
        slot_valid = slot >= 0
        safe_slot = tl.where(slot_valid, slot, 0)
        page = safe_slot // CACHE_BLOCK_SIZE
        page_offset = safe_slot % CACHE_BLOCK_SIZE

        k_base = token * stride_k_token
        k = tl.load(K + k_base + dims).to(tl.float32)
        k_partner = tl.load(
            K + k_base + partner_dims,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        k_rotated = tl.extra.libdevice.fma_rn(k, cos, rope_sign * k_partner * sin)
        k_rotated = k_rotated.to(tl.bfloat16).to(tl.float32)
        k = tl.where(rope_mask, k_rotated, k)
        k_amax = tl.maximum(tl.max(tl.abs(k)), 1.0e-4)
        k_scale_init = _ieee_rn_div_f32(k_amax, 448.0)
        k_scale = _round_up_ue8m0(k_scale_init)
        k_quant = tl.clamp(
            _ieee_rn_div_f32(k, tl.full(k.shape, k_scale, tl.float32)),
            -448.0,
            448.0,
        ).to(K_CACHE_FP8.dtype.element_ty)

        # Indexer cache layout is page-major with all K bytes first, followed
        # by one float scale per page slot.  It is not token-strided by 132B.
        cache_k_base = page * stride_cache_fp8_page + page_offset * HEAD_DIM
        tl.store(
            K_CACHE_FP8 + cache_k_base + dims,
            k_quant,
            mask=slot_valid,
        )
        cache_scale_offset = (
            page * stride_cache_fp32_page
            + CACHE_BLOCK_SIZE * (HEAD_DIM // 4)
            + page_offset
        )
        tl.store(K_CACHE_FP32 + cache_scale_offset, k_scale, mask=slot_valid)


def can_fuse_hy4_indexer_rope_quant_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    is_neox_style: bool,
) -> bool:
    """Check metadata before CMP submits either independent branch."""
    if not q.is_cuda:
        return False
    if not isinstance(cos_sin_cache, torch.Tensor):
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False
    if q.dim() != 3 or tuple(q.shape[1:]) != (
        _HY4_INDEX_HEADS,
        _HY4_HEAD_DIM,
    ):
        return False
    if k.dim() != 2 or k.shape != (q.shape[0], _HY4_HEAD_DIM):
        return False
    if not q.is_contiguous() or not k.is_contiguous():
        return False
    if (
        positions.dim() != 1
        or positions.numel() != q.shape[0]
        or positions.dtype not in (torch.int32, torch.int64)
        or not positions.is_contiguous()
    ):
        return False
    if slot_mapping.dim() != 1 or slot_mapping.numel() != q.shape[0]:
        return False
    if slot_mapping.dtype != torch.int64 or not slot_mapping.is_contiguous():
        return False
    if (
        cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.dim() != 2
        or cos_sin_cache.stride(1) != 1
    ):
        return False
    if cos_sin_cache.shape[1] < _HY4_ROPE_DIM:
        return False
    if kv_cache.dtype != torch.uint8 or kv_cache.dim() != 3:
        return False
    if not kv_cache.is_contiguous() or kv_cache.shape[2] != _HY4_HEAD_DIM + 4:
        return False
    if any(
        tensor.device != q.device
        for tensor in (k, positions, cos_sin_cache, slot_mapping, kv_cache)
    ):
        return False

    return True


def fused_hy4_indexer_rope_quant_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    is_neox_style: bool,
    branch: str = "qk",
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    raw_head_gate: Optional[torch.Tensor] = None,
    head_weights: Optional[torch.Tensor] = None,
    head_scale: float = 1.0,
    round_head_gate_bf16: bool = False,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Run both branches, or only Q/K for CMP, with the same numerical kernel.

    In Q-only mode K is metadata-only; in K-only mode Q and its outputs are
    metadata-only. Their storage may be awaiting a producer on another stream.
    Caller-owned ``out`` buffers avoid side-stream allocations.
    """
    if branch not in ("q", "k", "qk"):
        raise ValueError("Indexer branch must be q, k, or qk")
    if not can_fuse_hy4_indexer_rope_quant_cache(
        q,
        k,
        positions,
        cos_sin_cache,
        slot_mapping,
        kv_cache,
        is_neox_style=is_neox_style,
    ):
        return None
    num_tokens = q.shape[0]
    if out is None:
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        q_scale = torch.empty(
            (num_tokens, _HY4_INDEX_HEADS),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        q_fp8, q_scale = out
        q_scale = q_scale.squeeze(-1) if q_scale.dim() == 3 else q_scale
        if (
            q_fp8.shape != q.shape
            or q_fp8.dtype != torch.float8_e4m3fn
            or q_fp8.device != q.device
            or not q_fp8.is_contiguous()
            or q_scale.shape != (num_tokens, _HY4_INDEX_HEADS)
            or q_scale.dtype != torch.float32
            or q_scale.device != q.device
            or not q_scale.is_contiguous()
        ):
            raise ValueError("invalid HY4 Indexer Q output buffers")
    if (raw_head_gate is None) != (head_weights is None):
        raise ValueError("raw head gate and head-weight output must be paired")
    if raw_head_gate is not None and any(
        tensor.shape != (num_tokens, _HY4_INDEX_HEADS)
        or tensor.dtype != torch.float32
        or tensor.device != q.device
        or not tensor.is_contiguous()
        for tensor in (raw_head_gate, head_weights)
    ):
        raise ValueError("invalid HY4 Indexer head-gate buffers")
    if num_tokens == 0:
        return q_fp8, q_scale.unsqueeze(-1)

    cache_fp8 = kv_cache.view(torch.float8_e4m3fn)
    cache_fp32 = kv_cache.view(torch.float32)
    head_tiles = (_HY4_INDEX_HEADS + _Q_HEAD_TILE - 1) // _Q_HEAD_TILE
    grid = (num_tokens, 1 if branch == "k" else head_tiles + (branch == "qk"))
    _fused_hy4_indexer_rope_quant_kernel[grid](
        q,
        k,
        positions,
        cos_sin_cache,
        slot_mapping,
        q_fp8,
        q_scale,
        cache_fp8,
        cache_fp32,
        raw_head_gate if raw_head_gate is not None else q_scale,
        head_weights if head_weights is not None else q_scale,
        float(head_scale),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        cos_sin_cache.stride(0),
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_scale.stride(0),
        cache_fp8.stride(0),
        cache_fp32.stride(0),
        CACHE_BLOCK_SIZE=kv_cache.shape[1],
        HEAD_DIM=_HY4_HEAD_DIM,
        ROT_DIM=_HY4_ROPE_DIM,
        IS_NEOX=is_neox_style,
        HEAD_TILE=_Q_HEAD_TILE,
        HEAD_TILES=head_tiles,
        TILE_OFFSET=head_tiles if branch == "k" else 0,
        FOLD_HEAD_GATE=raw_head_gate is not None,
        ROUND_HEAD_GATE_BF16=round_head_gate_bf16,
        num_warps=4,
        num_stages=2,
    )
    return q_fp8, q_scale.unsqueeze(-1)


__all__ = [
    "can_fuse_hy4_indexer_rope_quant_cache",
    "fused_hy4_indexer_rope_quant_cache",
]

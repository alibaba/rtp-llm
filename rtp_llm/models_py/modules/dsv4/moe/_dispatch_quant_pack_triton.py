"""Fuse FP8 quant + dispatch pad/pack for SM120 fixed-EP AllGather.

Writes one rank-local row-major payload so dispatch is a single
``all_gather_into_tensor``. Each row is::

    [fp8 x | packed UE8M0 scale int32 | fp32 weights | int32 expert ids]

Pad rows (``n_valid .. n_pad``) get zero activations / weights and ``-1`` ids.
Scale layout is packed-int32 row-major ``[T, ceil(D/128/4)]``; ``ep_scatter_v2``
copies it by stride into the DeepGEMM TMA buffer, so the post-AG view does not
need a TMA repack.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..quant_layouts import FP8_BLOCK

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)


def dispatch_payload_layout(
    hidden: int, topk: int, group_size: int = FP8_BLOCK
) -> tuple[int, int, int, int, int]:
    """Return ``(payload_bytes, packed, x_bytes, scale_bytes, w_bytes)``."""
    if hidden % group_size != 0:
        raise ValueError(f"hidden {hidden} is not divisible by {group_size}")
    if hidden % 4 != 0:
        raise ValueError(f"hidden {hidden} must be 4-byte aligned for payload views")
    groups = hidden // group_size
    packed = (groups + 3) // 4
    x_bytes = hidden
    scale_bytes = packed * 4
    w_bytes = topk * 4
    i_bytes = topk * 4
    payload_bytes = x_bytes + scale_bytes + w_bytes + i_bytes
    return payload_bytes, packed, x_bytes, scale_bytes, w_bytes


def view_dispatch_payload(
    payload: torch.Tensor,
    hidden: int,
    topk: int,
    packed: int,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strided views of a ``[T, payload_bytes]`` uint8 payload."""
    if weight_dtype != torch.float32:
        raise TypeError(f"dispatch payload weights must be fp32, got {weight_dtype}")
    rows, payload_bytes = payload.shape
    u8 = payload.reshape(-1)
    all_x = torch.as_strided(
        u8.view(torch.float8_e4m3fn), (rows, hidden), (payload_bytes, 1)
    )
    i32 = u8.view(torch.int32)
    row_i32 = payload_bytes // 4
    scale_off = hidden // 4
    all_scale = torch.as_strided(i32[scale_off:], (rows, packed), (row_i32, 1))
    w_off = scale_off + packed
    all_w = torch.as_strided(
        i32[w_off:].view(torch.float32), (rows, topk), (row_i32, 1)
    )
    all_i = torch.as_strided(i32[w_off + topk :], (rows, topk), (row_i32, 1))
    return all_x, all_scale, all_w, all_i


@triton.jit
def _quant_pack_group_kernel(
    x_ptr,
    x_stride0,
    x_stride1,
    w_ptr,
    w_stride0,
    w_stride1,
    ids_ptr,
    ids_stride0,
    ids_stride1,
    x_out_ptr,
    x_out_stride0,
    scale_u8_ptr,
    scale_u8_stride0,
    w_out_ptr,
    w_out_stride0,
    w_out_stride1,
    ids_out_ptr,
    ids_out_stride0,
    ids_out_stride1,
    n_valid,
    fp8_min,
    fp8_max,
    eps,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_PAD: tl.constexpr,
    CONTIG_X: tl.constexpr,
):
    """One CTA per ``(token, group)``. Decode n_pad is tiny; this launches
    enough CTAs to beat a (token, packed-scale) grid that serializes 4 groups."""
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    offs = tl.max_contiguous(tl.multiple_of(tl.arange(0, GROUP_SIZE), 16), 16)
    col = group * GROUP_SIZE + offs
    x_row = x_ptr + token * x_stride0
    x_out_row = x_out_ptr + token * x_out_stride0
    valid = (not HAS_PAD) or (token < n_valid)
    if valid:
        if CONTIG_X:
            val = tl.load(x_row + col).to(tl.float32)
        else:
            val = tl.load(x_row + col * x_stride1).to(tl.float32)
        absmax = tl.max(tl.abs(val))
        scale_raw = tl.maximum(absmax / fp8_max, eps)
        exponent = tl.ceil(tl.log2(scale_raw))
        scale = tl.math.exp2(exponent)
        q = tl.clamp(val / scale, fp8_min, fp8_max)
        tl.store(x_out_row + col, q.to(x_out_ptr.dtype.element_ty))
        exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
    else:
        tl.store(
            x_out_row + col,
            tl.zeros([GROUP_SIZE], dtype=x_out_ptr.dtype.element_ty),
        )
        exponent_biased = 0
    tl.store(
        scale_u8_ptr + token * scale_u8_stride0 + group,
        exponent_biased.to(tl.uint8),
    )
    if group == 0:
        offs_k = tl.arange(0, TOPK_PAD)
        mask_k = offs_k < TOPK
        if valid:
            w = tl.load(
                w_ptr + token * w_stride0 + offs_k * w_stride1,
                mask=mask_k,
                other=0.0,
            )
            expert = tl.load(
                ids_ptr + token * ids_stride0 + offs_k * ids_stride1,
                mask=mask_k,
                other=-1,
            )
            ids32 = expert.to(tl.int32)
        else:
            w = tl.zeros([TOPK_PAD], dtype=w_out_ptr.dtype.element_ty)
            ids32 = tl.full([TOPK_PAD], -1, dtype=tl.int32)
        tl.store(
            w_out_ptr + token * w_out_stride0 + offs_k * w_out_stride1,
            w,
            mask=mask_k,
        )
        tl.store(
            ids_out_ptr + token * ids_out_stride0 + offs_k * ids_out_stride1,
            ids32,
            mask=mask_k,
        )


@triton.jit
def _quant_pack_dispatch_payload_kernel(
    x_ptr,
    x_stride0,
    x_stride1,
    w_ptr,
    w_stride0,
    w_stride1,
    ids_ptr,
    ids_stride0,
    ids_stride1,
    x_out_ptr,
    x_out_stride0,
    scale_out_ptr,
    scale_out_stride0,
    scale_out_stride1,
    w_out_ptr,
    w_out_stride0,
    w_out_stride1,
    ids_out_ptr,
    ids_out_stride0,
    ids_out_stride1,
    n_valid,
    fp8_min,
    fp8_max,
    eps,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_PAD: tl.constexpr,
    CONTIG_X: tl.constexpr,
):
    """One CTA per ``(token, packed-scale)``. Decode n_pad is small; pack-parallel
    beats a single row CTA walking all groups."""
    token = tl.program_id(0).to(tl.int64)
    pack = tl.program_id(1).to(tl.int64)
    offs = tl.max_contiguous(tl.multiple_of(tl.arange(0, GROUP_SIZE), 16), 16)
    x_row = x_ptr + token * x_stride0
    x_out_row = x_out_ptr + token * x_out_stride0
    valid = (not HAS_PAD) or (token < n_valid)
    packed_scale: tl.int32 = 0
    base = pack * (4 * GROUP_SIZE)
    if valid:
        for g in tl.static_range(4):
            col = base + g * GROUP_SIZE + offs
            if CONTIG_X:
                val = tl.load(x_row + col).to(tl.float32)
            else:
                val = tl.load(x_row + col * x_stride1).to(tl.float32)
            absmax = tl.max(tl.abs(val))
            scale_raw = tl.maximum(absmax / fp8_max, eps)
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)
            q = tl.clamp(val / scale, fp8_min, fp8_max)
            tl.store(x_out_row + col, q.to(x_out_ptr.dtype.element_ty))
            exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
            packed_scale = packed_scale | (exponent_biased << (g * 8))
    else:
        zero_q = tl.zeros([GROUP_SIZE], dtype=tl.float32).to(x_out_ptr.dtype.element_ty)
        for g in tl.static_range(4):
            tl.store(x_out_row + base + g * GROUP_SIZE + offs, zero_q)
    tl.store(
        scale_out_ptr + token * scale_out_stride0 + pack * scale_out_stride1,
        packed_scale,
    )
    if pack == 0:
        offs_k = tl.arange(0, TOPK_PAD)
        mask_k = offs_k < TOPK
        if valid:
            w = tl.load(
                w_ptr + token * w_stride0 + offs_k * w_stride1,
                mask=mask_k,
                other=0.0,
            )
            expert = tl.load(
                ids_ptr + token * ids_stride0 + offs_k * ids_stride1,
                mask=mask_k,
                other=-1,
            )
            ids32 = expert.to(tl.int32)
        else:
            w = tl.zeros([TOPK_PAD], dtype=w_out_ptr.dtype.element_ty)
            ids32 = tl.full([TOPK_PAD], -1, dtype=tl.int32)
        tl.store(
            w_out_ptr + token * w_out_stride0 + offs_k * w_out_stride1,
            w,
            mask=mask_k,
        )
        tl.store(
            ids_out_ptr + token * ids_out_stride0 + offs_k * ids_out_stride1,
            ids32,
            mask=mask_k,
        )


def quant_pack_dispatch_payload(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    payload: torch.Tensor,
    n_valid: int,
    group_size: int = FP8_BLOCK,
    eps: float = 1e-4,
    views: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``x[:n_valid]`` and pack/pad into ``payload`` in one launch."""
    assert payload.dtype == torch.uint8 and payload.is_contiguous()
    assert weights.dtype == torch.float32
    n_pad, hidden = payload.size(0), x.size(1)
    topk = int(weights.size(1))
    if views is None:
        payload_bytes, packed, _, _, _ = dispatch_payload_layout(
            hidden, topk, group_size
        )
        if payload.size(1) != payload_bytes:
            raise ValueError(
                f"payload width {payload.size(1)} != layout {payload_bytes}"
            )
        x_out, scale_out, w_out, i_out = view_dispatch_payload(
            payload, hidden, topk, packed, weights.dtype
        )
    else:
        x_out, scale_out, w_out, i_out = views
        packed = int(scale_out.size(1))
    if n_pad == 0:
        return x_out, scale_out, w_out, i_out
    topk_pad = triton.next_power_of_2(max(topk, 1))
    has_pad = int(n_valid) < n_pad
    contig_x = x.stride(1) == 1
    groups = hidden // group_size
    # Decode n_pad is 4/8. A CTA per (token, 128-group) hides less work
    # behind launch than a CTA that walks 4 groups; graph replay ~1.6x.
    if n_pad <= 32:
        scale_u8 = torch.as_strided(
            payload.reshape(-1)[hidden:],
            (n_pad, groups),
            (payload.size(1), 1),
        )
        _quant_pack_group_kernel[(n_pad, groups)](
            x,
            x.stride(0),
            x.stride(1),
            weights,
            weights.stride(0),
            weights.stride(1),
            indices,
            indices.stride(0),
            indices.stride(1),
            x_out,
            x_out.stride(0),
            scale_u8,
            scale_u8.stride(0),
            w_out,
            w_out.stride(0),
            w_out.stride(1),
            i_out,
            i_out.stride(0),
            i_out.stride(1),
            int(n_valid),
            _FP8_INFO.min,
            _FP8_INFO.max,
            eps,
            TOPK=topk,
            TOPK_PAD=topk_pad,
            GROUP_SIZE=group_size,
            HAS_PAD=has_pad,
            CONTIG_X=contig_x,
            num_warps=2,
            num_stages=2,
        )
        return x_out, scale_out, w_out, i_out
    _quant_pack_dispatch_payload_kernel[(n_pad, packed)](
        x,
        x.stride(0),
        x.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        indices,
        indices.stride(0),
        indices.stride(1),
        x_out,
        x_out.stride(0),
        scale_out,
        scale_out.stride(0),
        scale_out.stride(1),
        w_out,
        w_out.stride(0),
        w_out.stride(1),
        i_out,
        i_out.stride(0),
        i_out.stride(1),
        int(n_valid),
        _FP8_INFO.min,
        _FP8_INFO.max,
        eps,
        TOPK=topk,
        TOPK_PAD=topk_pad,
        GROUP_SIZE=group_size,
        HAS_PAD=has_pad,
        CONTIG_X=contig_x,
        num_warps=4,
        num_stages=2,
    )
    return x_out, scale_out, w_out, i_out

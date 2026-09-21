"""Ordinary BF16 Qwen3.5 prefill mRoPE, output packing and paged KV store."""

import torch
import triton
import triton.language as tl

from .prefill_fusion import MROPE, enabled, in_prefill


# Keep the scalar FP32 rounding of legacy_mrope: Blackwell packed FP32
# arithmetic can move BF16 tie cases across a rounding boundary.
@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prefill_mrope_cache(
    QKV,
    OUT,
    POS,
    CU,
    PREFIX,
    OFFSETS,
    CACHE,
    SCALES,
    T: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    PAIRS: tl.constexpr,
    BASE: tl.constexpr,
    SCALE: tl.constexpr,
    HP: tl.constexpr,
    WP: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    NSEQ: tl.constexpr,
    HAS_POS: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
    STORE_CACHE: tl.constexpr,
    FP8_CACHE: tl.constexpr,
    QOUT: tl.constexpr,
    TPB: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    RP: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, BD)
    valid = (t[:, None] < T) & (d[None, :] < D)
    x = tl.load(QKV + t[:, None] * QS + head * D + d[None, :], valid, 0).to(tl.float32)
    # Binary search packed sequence offsets, independent of mRoPE coordinates.
    lo = tl.full((BM,), 0, tl.int32)
    hi = tl.full((BM,), NSEQ, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(CU + mid + 1, mid < NSEQ, 2147483647)
        before = t < end
        active = lo < hi
        hi = tl.where(active & before, mid, hi)
        lo = tl.where(active & ~before, mid + 1, lo)
    seq = tl.minimum(lo, NSEQ - 1)
    start = tl.load(CU + seq)
    physical = t - start
    if HAS_PREFIX:
        physical += tl.load(PREFIX + seq)
    if head < QH + KH:
        # Compute sin/cos once per rotary pair, then reuse for both halves.
        # Broadcasting also avoids evaluating trigonometry for the unrotated tail.
        rotary_pair = tl.arange(0, RP) % PAIRS
        pair = d % PAIRS
        if HAS_POS:
            axis = tl.where((rotary_pair % 3 == 1) & (rotary_pair < 3 * HP), 1, 0)
            axis = tl.where((rotary_pair % 3 == 2) & (rotary_pair < 3 * WP), 2, axis)
            position = tl.load(
                POS + t[:, None] * PS0 + axis[None, :] * PS1,
                (t[:, None] < T) & (rotary_pair[None, :] < PAIRS),
                0,
            ).to(tl.float32)
        else:
            position = physical[:, None].to(tl.float32)
        inv_freq = tl.exp(-tl.log(BASE) * rotary_pair.to(tl.float32) / PAIRS)
        angle = position * inv_freq[None, :] / SCALE
        cos = tl.broadcast_to(tl.cos(angle)[:, None, :], (BM, BD // RP, RP)).reshape(
            BM, BD
        )
        sin = tl.broadcast_to(tl.sin(angle)[:, None, :], (BM, BD // RP, RP)).reshape(
            BM, BD
        )
        other_d = tl.where(d < PAIRS, d + PAIRS, d - PAIRS)
        other = tl.load(
            QKV + t[:, None] * QS + head * D + other_d[None, :],
            (t[:, None] < T) & (d[None, :] < 2 * PAIRS),
            0,
        ).to(tl.float32)
        low = tl.where(d[None, :] < PAIRS, x, other)
        high = tl.where(d[None, :] < PAIRS, other, x)
        rotated = tl.where(
            d[None, :] < PAIRS,
            _sub_rn(_mul_rn(low, cos), _mul_rn(high, sin)),
            _add_rn(_mul_rn(high, cos), _mul_rn(low, sin)),
        )
        value = tl.where(d[None, :] < 2 * PAIRS, rotated, x).to(tl.bfloat16)
        # Preserve the old in-place Q/K update, including QOut callers.
        tl.store(QKV + t[:, None] * QS + head * D + d[None, :], value, valid)
    else:
        value = x.to(tl.bfloat16)
    if QOUT:
        if head < QH:
            tl.store(OUT + t[:, None] * (QH * D) + head * D + d[None, :], value, valid)
    if STORE_CACHE:
        if head >= QH:
            kind = (head - QH) // KH
            kh = (head - QH) % KH
            block = physical // TPB
            offset = tl.load(
                OFFSETS + (seq * 2 + kind) * MAX_BLOCKS + block, t < T, 0
            ).to(tl.int64)
            slot = physical % TPB
            address = (
                offset[:, None] * (KH * TPB * D)
                + kh * TPB * D
                + slot[:, None] * D
                + d[None, :]
            )
            if FP8_CACHE:
                # CUDA __nv_fp8_e4m3 uses finite saturation.
                cache_value = tl.minimum(
                    tl.maximum(value.to(tl.float32), -448.0), 448.0
                )
            else:
                cache_value = value
            tl.store(CACHE + address, cache_value, valid)
            if FP8_CACHE:
                tl.store(SCALES + offset * (KH * TPB) + kh * TPB + slot, 1.0, t < T)


def maybe_prefill_mrope_cache(qkv, kv_cache, params, config, *, qout, qkvout):
    if not in_prefill() or not enabled(MROPE) or params.decode_plan:
        return None
    rope = config.rope_config
    qh, kh, d = config.head_num, config.kv_head_num, config.size_per_head
    if not (
        qkv.is_cuda
        and qkv.dtype == torch.bfloat16
        and qkv.ndim == 2
        and qkv.stride(1) == 1
        and qkv.shape[1] == (qh + 2 * kh) * d
        and qh > 0
        and kh > 0
        and d in (128, 256)
        and rope.mrope_interleaved
        and 0 < rope.dim <= d
        and rope.dim % 2 == 0
        and not config.use_logn_attn
        and bool(qout) != bool(qkvout)
    ):
        return None
    pairs = rope.dim // 2
    if (
        rope.mrope_dim1 + rope.mrope_dim2 + rope.mrope_dim3 != pairs
        or min(rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3) < 0
        or rope.mrope_dim2 > (pairs + 1) // 3
        or rope.mrope_dim3 > pairs // 3
        or rope.base <= 0
        or rope.scale <= 0
    ):
        return None
    cu = params.cu_seqlens
    if not (
        cu.is_cuda
        and cu.device == qkv.device
        and cu.is_contiguous()
        and cu.ndim == 1
        and cu.numel() >= 2
        and cu.dtype in (torch.int32, torch.int64)
    ):
        return None
    pos = params.position_ids
    if pos is not None:
        if (
            pos.device != qkv.device
            or pos.numel() != qkv.shape[0] * 3
            or pos.dtype not in (torch.int32, torch.int64)
        ):
            return None
        pos = pos.reshape(-1, 3)
    # Serving keeps the host lengths for planning and already has a device copy.
    # Reuse it for cache hits instead of introducing a per-layer H2D transfer.
    prefix = getattr(params, "prefix_lengths_device", None)
    if prefix is None:
        prefix = params.prefix_lengths
    has_prefix = params.max_prefix_length > 0
    if has_prefix and not (
        prefix is not None
        and prefix.is_cuda
        and prefix.device == qkv.device
        and prefix.is_contiguous()
        and prefix.ndim == 1
        and prefix.dtype in (torch.int32, torch.int64)
        and prefix.numel() == cu.numel() - 1
    ):
        return None
    cache = offsets = scales = None
    tpb, max_blocks, fp8 = config.kernel_tokens_per_block, 0, False
    if tpb <= 0:
        return None
    if kv_cache is not None:
        cache, offsets, scales = (
            kv_cache.kv_cache_base,
            params.kv_cache_offset,
            kv_cache.kv_scale_base,
        )
        if (
            cache.device != qkv.device
            or not cache.is_contiguous()
            or cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            or offsets is None
            or offsets.device != qkv.device
            or offsets.dtype != torch.int32
            or not offsets.is_contiguous()
            or offsets.ndim != 4
            or tuple(offsets.shape[:3]) != (cu.numel() - 1, 1, 2)
            or cache.numel() % (2 * kh * tpb * d)
        ):
            return None
        max_blocks = offsets.shape[-1]
        fp8 = cache.dtype == torch.float8_e4m3fn
        if fp8 and (
            scales is None
            or scales.device != qkv.device
            or scales.dtype != torch.float32
            or not scales.is_contiguous()
            or scales.numel() < cache.numel() // d
        ):
            return None
    out = (
        torch.empty((qkv.shape[0], qh, d), device=qkv.device, dtype=qkv.dtype)
        if qout
        else qkv
    )
    if qkv.shape[0]:
        _prefill_mrope_cache[(triton.cdiv(qkv.shape[0], 8), qh + 2 * kh)](
            qkv,
            out,
            pos,
            cu,
            prefix,
            offsets,
            cache,
            scales,
            qkv.shape[0],
            qh,
            kh,
            d,
            qkv.stride(0),
            pairs,
            float(rope.base),
            float(rope.scale),
            rope.mrope_dim2,
            rope.mrope_dim3,
            pos.stride(0) if pos is not None else 0,
            pos.stride(1) if pos is not None else 0,
            cu.numel() - 1,
            pos is not None,
            has_prefix,
            cache is not None,
            fp8,
            qout,
            tpb,
            max_blocks,
            8,
            triton.next_power_of_2(d),
            RP=pairs if pairs & (pairs - 1) == 0 else triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out

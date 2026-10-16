# SPDX-License-Identifier: Apache-2.0
# Norm/RoPE arithmetic adapted from vLLM's fused_qk_norm_rope.py.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen Q/K RMSNorm, partial NeoX RoPE and HND paged KV append.

Keep vLLM's FP32 head reduction and explicit BF16 norm-to-RoPE boundary.
QKV remains read-only; cache positions are independent of multimodal positions.
"""

import math
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.prefill_input_log import trace_triton


@triton.jit
def _qk_norm_rope_cache(
    QKV,
    QW,
    KW,
    COS_SIN,
    POS,
    CU,
    PREFIX,
    LENGTHS,
    OFFSETS,
    CACHE,
    SCALES,
    OUT,
    QS: tl.constexpr,
    OS: tl.constexpr,
    CS: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    T: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    PAIRS: tl.constexpr,
    HP: tl.constexpr,
    WP: tl.constexpr,
    HAS_POS: tl.constexpr,
    MROPE: tl.constexpr,
    DECODE: tl.constexpr,
    LENGTHS_PLUS_ONE: tl.constexpr,
    NSEQ: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CACHE_BLOCKS: tl.constexpr,
    TPB: tl.constexpr,
    FP8_CACHE: tl.constexpr,
    HAS_SCALES: tl.constexpr,
    FP8_Q: tl.constexpr,
    EPS: tl.constexpr,
    BD: tl.constexpr,
    RP: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, BD)
    r = tl.arange(0, RP)
    total = tl.load(CU + NSEQ)
    if token < total:
        # Packed rows can include empty sequences and graph padding.
        lo = tl.full((), 0, tl.int32)
        hi = tl.full((), NSEQ, tl.int32)
        while lo < hi:
            mid = (lo + hi) // 2
            end = tl.load(CU + mid + 1)
            before = token < end
            hi = tl.where(before, mid, hi)
            lo = tl.where(before, lo, mid + 1)
        seq = lo
        begin = tl.load(CU + seq)
        end = tl.load(CU + seq + 1)
        if DECODE:
            physical = tl.load(LENGTHS + seq).to(tl.int64) + token - begin
            if LENGTHS_PLUS_ONE:
                physical -= 1
        else:
            physical = token - begin + tl.load(PREFIX + seq).to(tl.int64)
        row = QKV + token * QS + head.to(tl.int64) * D
        x = tl.load(row + d, d < D, 0).to(tl.float32)
        rot_low = tl.full((RP,), 0.0, tl.float32)
        rot_high = tl.full((RP,), 0.0, tl.float32)
        if head < QH + KH:
            wptr = tl.where(head < QH, QW, KW)
            variance = tl.sum(x * x, 0) / D
            inv = tl.rsqrt(variance + EPS)
            # Align with vLLM: raw gamma + 1 is computed in FP32, avoiding
            # an extra BF16 rounding. The norm output is then rounded to
            # BF16 before RoPE, preserving vLLM's norm-to-RoPE boundary.
            weight = tl.load(wptr + d, d < D, 0).to(tl.float32) + 1.0
            normalized = (x * inv * weight).to(tl.bfloat16).to(tl.float32)
            # Recompute rotary halves as in vLLM, with the same rounding.
            low = tl.load(row + r, r < PAIRS, 0).to(tl.float32)
            high = tl.load(row + PAIRS + r, r < PAIRS, 0).to(tl.float32)
            wl = tl.load(wptr + r, r < PAIRS, 0).to(tl.float32) + 1.0
            wh = tl.load(wptr + PAIRS + r, r < PAIRS, 0).to(tl.float32) + 1.0
            low = (low * inv * wl).to(tl.bfloat16).to(tl.float32)
            high = (high * inv * wh).to(tl.bfloat16).to(tl.float32)
            if HAS_POS:
                if MROPE:
                    axis = tl.where((r % 3 == 1) & (r < 3 * HP), 1, 0)
                    axis = tl.where((r % 3 == 2) & (r < 3 * WP), 2, axis)
                    pos = tl.load(POS + token * PS0 + axis * PS1, r < PAIRS, 0).to(
                        tl.int64
                    )
                else:
                    pos = tl.load(POS + token * PS0).to(tl.int64)
            else:
                pos = physical
            cos = tl.load(
                COS_SIN + pos * CS + r, (r < PAIRS) & (pos >= 0) & (pos < CACHE_ROWS), 0
            ).to(tl.float32)
            sin = tl.load(
                COS_SIN + pos * CS + PAIRS + r,
                (r < PAIRS) & (pos >= 0) & (pos < CACHE_ROWS),
                0,
            ).to(tl.float32)
            rot_low = (low * cos - high * sin).to(tl.bfloat16).to(tl.float32)
            rot_high = (high * cos + low * sin).to(tl.bfloat16).to(tl.float32)
        else:
            normalized = x
        if head < QH:
            out_row = OUT + token * OS + head.to(tl.int64) * D
            value = normalized
            if FP8_Q:
                value = tl.minimum(tl.maximum(value, -448.0), 448.0)
                rot_low = tl.minimum(tl.maximum(rot_low.to(tl.float32), -448.0), 448.0)
                rot_high = tl.minimum(
                    tl.maximum(rot_high.to(tl.float32), -448.0), 448.0
                )
            tl.store(out_row + d, value, (d < D) & (d >= 2 * PAIRS))
            tl.store(out_row + r, rot_low, r < PAIRS)
            tl.store(out_row + PAIRS + r, rot_high, r < PAIRS)
        else:
            kind = (head - QH) // KH
            kh = (head - QH) % KH
            block = physical // TPB
            block_valid = (physical >= 0) & (block < MAX_BLOCKS)
            offset = tl.load(
                OFFSETS + (seq * 2 + kind) * MAX_BLOCKS + block, block_valid, -1
            ).to(tl.int64)
            valid_page = block_valid & (offset >= 0) & (offset < CACHE_BLOCKS)
            slot = physical % TPB
            cache_row = CACHE + offset * (KH * TPB * D) + kh * TPB * D + slot * D
            value = normalized.to(tl.bfloat16).to(tl.float32)
            if FP8_CACHE:
                value = tl.minimum(tl.maximum(value.to(tl.float32), -448.0), 448.0)
            if kind == 0:
                if FP8_CACHE:
                    rot_low = tl.minimum(
                        tl.maximum(rot_low.to(tl.float32), -448.0), 448.0
                    )
                    rot_high = tl.minimum(
                        tl.maximum(rot_high.to(tl.float32), -448.0), 448.0
                    )
                tl.store(cache_row + d, value, valid_page & (d < D) & (d >= 2 * PAIRS))
                tl.store(cache_row + r, rot_low, valid_page & (r < PAIRS))
                tl.store(cache_row + PAIRS + r, rot_high, valid_page & (r < PAIRS))
            else:
                tl.store(cache_row + d, value, valid_page & (d < D))
                # TRTLLM's P@V reads invalid V slots: keep 0*NaN out of valid output.
                if FP8_CACHE:
                    tail = (physical + 1) % TPB
                    if valid_page & (token == end - 1) & (tail != 0):
                        flat = tl.arange(0, 1024)
                        for chunk in range(tl.cdiv(TPB * D, 1024)):
                            i = chunk * 1024 + flat
                            tail_row = i // D
                            address = CACHE + offset * (KH * TPB * D) + kh * TPB * D + i
                            tl.store(address, 0.0, (i < TPB * D) & (tail_row >= tail))
            if HAS_SCALES:
                tl.store(
                    SCALES + offset * (KH * TPB) + kh * TPB + slot, 1.0, valid_page
                )
    else:
        if head < QH:
            tl.store(OUT + token * OS + head.to(tl.int64) * D + d, 0.0, d < D)


def fused_qk_norm_rope_cache(
    qkv,
    q_weight,
    k_weight,
    cos_sin_cache,
    position_ids,
    cu_seqlens,
    prefix_lengths,
    kv_cache,
    kv_offsets,
    *,
    num_q_heads,
    num_kv_heads,
    rotary_dim,
    page_size,
    eps=1e-6,
    mrope_section=None,
    sequence_lengths=None,
    lengths_plus_one=False,
    kv_scales=None,
    fp8_query=False,
    out=None,
):
    """Append packed BF16 QKV; return Q with vLLM norm/RoPE rounding.

    Offsets have RTP's [sequence, 1, K/V, page] block-array layout. Inputs must
    own their writable pages exclusively, including the last partial V page.
    Position IDs must be nonnegative and within cos_sin_cache's first dimension.
    Metadata checks never read CUDA values; graph replay may update all metadata.
    """
    qh, kh = num_q_heads, num_kv_heads
    d = q_weight.numel()
    tensors = (qkv, q_weight, k_weight, cos_sin_cache, cu_seqlens, kv_cache, kv_offsets)
    if any(not t.is_cuda or t.device != qkv.device for t in tensors):
        raise ValueError("all fusion tensors must be on the same CUDA device")
    if (
        qkv.dtype != torch.bfloat16
        or qkv.ndim != 2
        or qkv.stride(1) != 1
        or d != 256
        or qh <= 0
        or kh <= 0
        or qkv.shape[1] != (qh + 2 * kh) * d
        or q_weight.shape != (d,)
        or k_weight.shape != (d,)
        or any(
            w.dtype != qkv.dtype or not w.is_contiguous() for w in (q_weight, k_weight)
        )
    ):
        raise ValueError("fusion requires packed BF16 QKV and raw width-256 gamma")
    if (
        rotary_dim <= 0
        or rotary_dim > d
        or rotary_dim % 2
        or page_size <= 0
        or not math.isfinite(eps)
        or eps <= 0
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or cos_sin_cache.dtype not in (torch.bfloat16, torch.float32)
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError("unsupported rotary cache or dimensions")
    nseq = cu_seqlens.numel() - 1
    if (
        nseq <= 0
        or cu_seqlens.ndim != 1
        or not cu_seqlens.is_contiguous()
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or kv_offsets.shape[:3] != (nseq, 1, 2)
        or kv_offsets.ndim != 4
        or kv_offsets.dtype != torch.int32
        or not kv_offsets.is_contiguous()
        or kv_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or not kv_cache.is_contiguous()
        or kv_cache.numel() % (2 * kh * page_size * d)
    ):
        raise ValueError("unsupported packed metadata or HND cache")
    lengths = sequence_lengths if sequence_lengths is not None else prefix_lengths
    if (
        lengths is None
        or not lengths.is_cuda
        or lengths.device != qkv.device
        or lengths.dtype not in (torch.int32, torch.int64)
        or not lengths.is_contiguous()
        or lengths.shape != (nseq,)
    ):
        raise ValueError("device prefix/sequence lengths are required")
    if mrope_section is not None:
        if (
            len(mrope_section) != 3
            or min(mrope_section) < 0
            or sum(mrope_section) != rotary_dim // 2
            or mrope_section[1] > (rotary_dim // 2 + 1) // 3
            or mrope_section[2] > rotary_dim // 6
        ):
            raise ValueError("unsupported interleaved mRoPE sections")
    if position_ids is not None:
        axes = 3 if mrope_section is not None else 1
        if (
            not position_ids.is_cuda
            or position_ids.device != qkv.device
            or position_ids.dtype not in (torch.int32, torch.int64)
            or position_ids.numel() != qkv.shape[0] * axes
        ):
            raise ValueError("position IDs must match QKV token rows")
        position_ids = position_ids.reshape(qkv.shape[0], axes)
    if kv_scales is not None and (
        not kv_scales.is_cuda
        or kv_scales.device != qkv.device
        or kv_scales.dtype != torch.float32
        or not kv_scales.is_contiguous()
        or kv_scales.numel() < kv_cache.numel() // d
    ):
        raise ValueError("invalid static unit-scale output buffer")
    dtype = torch.float8_e4m3fn if fp8_query else qkv.dtype
    if out is None:
        out = torch.empty((qkv.shape[0], qh, d), device=qkv.device, dtype=dtype)
    if (
        out.shape != (qkv.shape[0], qh, d)
        or out.device != qkv.device
        or out.dtype != dtype
        or not out.is_contiguous()
    ):
        raise ValueError("query output shape/dtype mismatch")
    if out.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr():
        raise ValueError("query output must not alias QKV input")
    if qkv.shape[0]:
        trace_triton(
            "qk_norm_rope_cache:_qk_norm_rope_cache",
            _qk_norm_rope_cache,
            (qkv.shape[0], qh + 2 * kh),
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            position_ids,
            cu_seqlens,
            prefix_lengths,
            lengths,
            kv_offsets,
            kv_cache,
            kv_scales,
            out,
            qkv.stride(0),
            out.stride(0),
            cos_sin_cache.stride(0),
            position_ids.stride(0) if position_ids is not None else 0,
            position_ids.stride(1) if position_ids is not None else 0,
            qkv.shape[0],
            qh,
            kh,
            d,
            cos_sin_cache.shape[0],
            rotary_dim // 2,
            mrope_section[1] if mrope_section is not None else 0,
            mrope_section[2] if mrope_section is not None else 0,
            position_ids is not None,
            mrope_section is not None,
            sequence_lengths is not None,
            lengths_plus_one,
            nseq,
            kv_offsets.shape[-1],
            kv_cache.numel() // (kh * page_size * d),
            page_size,
            kv_cache.dtype == torch.float8_e4m3fn,
            kv_scales is not None,
            fp8_query,
            eps,
            triton.next_power_of_2(d),
            triton.next_power_of_2(rotary_dim // 2),
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=True,
        )
    return out


@lru_cache(maxsize=8)
def _vllm_rope_cache(base, rotary_dim, max_position, device):
    # Match vLLM RotaryEmbedding._compute_cos_sin_cache and BF16 cache storage.
    with torch.cuda.device(device):
        inv = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                / rotary_dim
            )
        )
        t = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def maybe_fused_qk_norm_rope_cache(qkv, norm, impl, kv_cache):
    """Metadata-only dispatch for the Qwen TRTLLM prefill/decode paths."""
    if os.getenv("RTP_QWEN35_QK_NORM_ROPE_CACHE_FUSION", "1") != "1":
        return None
    if not getattr(norm, "zero_centered", False) or kv_cache is None:
        return None
    from rtp_llm.ops import RopeStyle

    config, params, inputs = impl.attn_configs, impl.rope_params, impl.attn_inputs
    if getattr(inputs, "context_parallel_info", None) is not None:
        return None
    rope = config.rope_config
    mrope = rope.style == RopeStyle.Mrope and rope.mrope_interleaved
    base = rope.style == RopeStyle.Base and rope.index_factor == 1
    if (
        not config.need_rope_kv_cache
        or not (mrope or base)
        or config.size_per_head != 256
        or qkv.dtype != torch.bfloat16
        or config.use_logn_attn
        or rope.scale != 1.0
        or rope.offset != 0
        or rope.base <= 0
        or rope.dim <= 0
        or rope.dim > 256
        or rope.dim % 2
    ):
        return None
    # A missing CUDA mirror must fall back before any normalization/cache writes.
    cu = inputs.cu_seqlens_device
    prefix = inputs.prefix_lengths_device
    decode = bool(params.decode_plan)
    lengths = (
        getattr(inputs, "sequence_lengths_plus_1_device", None) if decode else None
    )
    if cu is None or prefix is None or (decode and lengths is None):
        return None
    if (
        not cu.is_cuda
        or not prefix.is_cuda
        or (lengths is not None and not lengths.is_cuda)
    ):
        return None
    pos = params.position_ids
    if pos is not None and pos.numel() == 0:
        pos = None
    axes = 3 if mrope else 1
    if pos is not None and (not pos.is_cuda or pos.numel() != qkv.shape[0] * axes):
        return None
    sections = (rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3) if mrope else None
    if mrope and (
        sum(sections) != rope.dim // 2
        or min(sections) < 0
        or sections[1] > (rope.dim // 2 + 1) // 3
        or sections[2] > rope.dim // 6
    ):
        return None
    offsets = params.kv_cache_offset
    cache = kv_cache.kv_cache_base
    if (
        offsets is None
        or offsets.ndim != 4
        or not offsets.is_cuda
        or not offsets.is_contiguous()
        or offsets.dtype != torch.int32
        or offsets.shape[:3] != (cu.numel() - 1, 1, 2)
        or not cache.is_contiguous()
        or cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
    ):
        return None
    key = (
        float(rope.base),
        rope.dim,
        config.max_seq_len * (4 if mrope else 1),
        qkv.device,
    )
    prepared = getattr(impl, "_qk_norm_rope_cos_sin", None)
    if prepared is None or prepared[0] != key:
        table = _vllm_rope_cache(*key)
        # CUDA graphs hold raw pointers. Keep ownership on the backend instance
        # even if another model evicts this table from the shared LRU cache.
        impl._qk_norm_rope_cos_sin = (key, table)
    else:
        table = prepared[1]
    return fused_qk_norm_rope_cache(
        qkv,
        norm.q_weight,
        norm.k_weight,
        table,
        pos,
        cu,
        prefix,
        cache,
        offsets,
        num_q_heads=config.head_num,
        num_kv_heads=config.kv_head_num,
        rotary_dim=rope.dim,
        page_size=config.kernel_tokens_per_block,
        eps=norm.eps,
        mrope_section=sections,
        sequence_lengths=lengths,
        lengths_plus_one=decode,
        kv_scales=(
            kv_cache.kv_scale_base if cache.dtype == torch.float8_e4m3fn else None
        ),
        fp8_query=bool(getattr(impl, "native_fp8", False)),
    )

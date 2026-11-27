"""Zero-sync QSA (Indexer Sparse Attention) bridge for Qwen 4.5-Flash.

Bridges Qwen 4.5-Flash (MQA-1, output gate, KV short-conv, YoCo KV sharing,
partial NeoX RoPE, arbitrary power-of-2 page size in {16, 32, 64, 128}, and
both H_idx=8 [Qwen4.5 target] and H_idx=4 [Qwen3.8 compat]) onto the production
QSA side pools (``indexer_kv``, ``indexer_state``) and sparse FMHA kernels.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    _HAS_TRITON = True
except Exception:  # pragma: no cover - pure-Python / no-triton fallback
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    libdevice = None  # type: ignore[assignment]
    _HAS_TRITON = False

from rtp_llm.models_py.modules.qwen4_5.short_conv import _triton_int_spec, triton_fast_launch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    Qwen4ExpQSAIndexer,
    apply_partial_rope,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
    write_indexer_cache,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)

_MAX_SCORE_WORKSPACE_BYTES = 128 * 1024 * 1024
_PAGED_SCORE_BN = 64


# ---------------------------------------------------------------------------
# RoPE & Cache Pool Views
# ---------------------------------------------------------------------------


_QSA_INV_FREQ_CACHE: dict = {}
_QSA_TRITON_CACHE: dict = {}


def _get_qsa_inv_freq(rotary_dim: int, rope_theta: float, device: torch.device) -> torch.Tensor:
    key = (int(rotary_dim), float(rope_theta), device.type, int(device.index) if device.index is not None else -1)
    cached = _QSA_INV_FREQ_CACHE.get(key)
    if cached is not None:
        return cached
    half = int(rotary_dim) // 2
    inv_freq = float(rope_theta) ** (
        -2.0 * torch.arange(half, dtype=torch.float32, device=device) / int(rotary_dim)
    )
    _QSA_INV_FREQ_CACHE[key] = inv_freq
    return inv_freq


if _HAS_TRITON:

    @triton.jit
    def _build_qsa_neox_cos_sin_kernel(
        POS_ptr,
        INV_FREQ_ptr,
        COS_ptr,
        SIN_ptr,
        ROWS,
        STRIDE_P: tl.int64,
        HALF: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
        mask_r = row < ROWS
        pos = tl.load(POS_ptr + row * STRIDE_P, mask=mask_r, other=0).to(tl.float32)
        freq_idx = tl.arange(0, HALF)
        inv_f = tl.load(INV_FREQ_ptr + freq_idx)
        ang = pos[:, None] * inv_f[None, :]
        out_ty = COS_ptr.dtype.element_ty
        c_val = libdevice.cos(ang).to(out_ty)
        s_val = libdevice.sin(ang).to(out_ty)
        off = row[:, None] * (HALF * 2) + freq_idx[None, :]
        tl.store(COS_ptr + off, c_val, mask=mask_r[:, None])
        tl.store(COS_ptr + off + HALF, c_val, mask=mask_r[:, None])
        tl.store(SIN_ptr + off, s_val, mask=mask_r[:, None])
        tl.store(SIN_ptr + off + HALF, s_val, mask=mask_r[:, None])

    @triton.jit(do_not_specialize=["stride_qb", "stride_pb", "stride_ob", "S"])
    def _qsa_head_rmsnorm_partial_rope_d128_r64_kernel(
        Q_ptr,
        GAMMA_ptr,
        POS_ptr,
        INV_FREQ_ptr,
        OUT_ptr,
        stride_qb: tl.int64,
        stride_qs: tl.int64,
        stride_qh: tl.int64,
        stride_pb: tl.int64,
        stride_ps: tl.int64,
        stride_ob: tl.int64,
        stride_os: tl.int64,
        stride_oh: tl.int64,
        S,
        H: tl.constexpr,
        EPS: tl.constexpr,
        HAS_NORM: tl.constexpr,
    ):
        """Single-launch fused QSA head RMSNorm (optional) + inline NeoX partial RoPE (D=128, R=64)."""
        b = tl.program_id(0).to(tl.int64)
        s = tl.program_id(1).to(tl.int64)
        h = tl.program_id(2).to(tl.int64)
        offs = tl.arange(0, 128)

        q_row = Q_ptr + b * stride_qb + s * stride_qs + h * stride_qh
        if HAS_NORM:
            qf = tl.load(q_row + offs).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(qf * qf, axis=0) * (1.0 / 128.0) + EPS)
            gain = 1.0 + tl.load(GAMMA_ptr + offs).to(tl.float32)
            x_bf = (qf * inv_rms * gain).to(tl.bfloat16)
        else:
            x_bf = tl.load(q_row + offs).to(tl.bfloat16)

        partner = tl.where(offs < 32, offs + 32, offs - 32)
        rot_half = tl.gather(x_bf, partner, axis=0)
        pos = tl.load(POS_ptr + b * stride_pb + s * stride_ps).to(tl.float32)
        freq_idx = tl.where(offs < 32, offs, offs - 32)
        inv_f = tl.load(INV_FREQ_ptr + freq_idx, mask=offs < 64, other=0.0)
        ang = pos * inv_f
        cos_bf = tl.where(offs < 64, libdevice.cos(ang).to(tl.bfloat16), tl.cast(1.0, tl.bfloat16))
        sin_bf = tl.where(offs < 64, libdevice.sin(ang).to(tl.bfloat16), tl.cast(0.0, tl.bfloat16))
        sign = tl.where(offs < 32, -1.0, 1.0)
        left = (x_bf.to(tl.float32) * cos_bf.to(tl.float32)).to(tl.bfloat16)
        right = (rot_half.to(tl.float32) * sin_bf.to(tl.float32) * sign).to(tl.bfloat16)
        out_val = tl.where(offs < 64, (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16), x_bf)

        out_row = OUT_ptr + b * stride_ob + s * stride_os + h * stride_oh
        tl.store(out_row + offs, out_val.to(OUT_ptr.dtype.element_ty))

    @triton.jit(do_not_specialize=["KV_TABLE_WIDTH", "STATE_TABLE_WIDTH", "KV_POOL_BLOCKS", "STATE_POOL_BLOCKS", "KV_TOKENS_PER_BLOCK", "STATE_TOKENS_PER_BLOCK", "KV_ENTRIES_PER_BLOCK"])
    def _write_decode_qsa_cache_k4_kernel(
        raw_ptr,
        seq_before_ptr,
        valid_ptr,
        inv_freq_ptr,
        gamma_ptr,
        kv_pool_ptr,
        kv_table_ptr,
        state_pool_ptr,
        state_table_ptr,
        stride_rb: tl.int64,
        stride_kv_b: tl.int64,
        stride_kv_p: tl.int64,
        stride_st_b: tl.int64,
        stride_st_p: tl.int64,
        KV_TABLE_WIDTH,
        STATE_TABLE_WIDTH,
        KV_POOL_BLOCKS,
        STATE_POOL_BLOCKS,
        KV_TOKENS_PER_BLOCK,
        STATE_TOKENS_PER_BLOCK,
        KV_ENTRIES_PER_BLOCK,
        RING_SIZE: tl.constexpr,
        D: tl.constexpr,
        ROTARY: tl.constexpr,
        EPS: tl.constexpr,
        HAS_VALID: tl.constexpr,
    ):
        request = tl.program_id(0).to(tl.int64)
        d = tl.arange(0, D)
        active = (tl.load(valid_ptr + request) != 0) if HAS_VALID else True
        position = tl.load(seq_before_ptr + request).to(tl.int64)
        state_column = position // STATE_TOKENS_PER_BLOCK
        ok_st_col = active & (position >= 0) & (state_column < STATE_TABLE_WIDTH)
        state_id = tl.load(
            state_table_ptr + request * stride_st_b + tl.minimum(tl.maximum(state_column, 0), STATE_TABLE_WIDTH - 1) * stride_st_p,
            mask=ok_st_col,
            other=0,
        ).to(tl.int64)
        state_valid = ok_st_col & (state_id > 0) & (state_id < STATE_POOL_BLOCKS)
        raw = tl.load(raw_ptr + request * stride_rb + d, mask=active, other=0).to(tl.float32)
        tl.store(
            state_pool_ptr + (state_id * RING_SIZE + (position % RING_SIZE)) * D + d,
            raw,
            mask=state_valid,
        )

        if active and ((position + 1) % 4 == 0):
            group = tl.arange(0, 4)
            source_positions = position - 3 + group
            source_columns = source_positions // STATE_TOKENS_PER_BLOCK
            in_b = (source_positions >= 0) & (source_columns < STATE_TABLE_WIDTH)
            source_ids = tl.load(
                state_table_ptr + request * stride_st_b + tl.minimum(tl.maximum(source_columns, 0), STATE_TABLE_WIDTH - 1) * stride_st_p,
                mask=in_b,
                other=0,
            ).to(tl.int64)
            source_valid = in_b & (source_ids > 0) & (source_ids < STATE_POOL_BLOCKS)
            source_slots = source_ids * RING_SIZE + (source_positions % RING_SIZE)
            history = tl.load(
                state_pool_ptr + source_slots[:, None] * D + d[None, :],
                mask=source_valid[:, None],
                other=0.0,
            ).to(tl.float32)
            history = tl.where(group[:, None] == 3, raw[None, :], history)
            pooled = (tl.sum(history, axis=0) * 0.25).to(tl.bfloat16).to(tl.float32)
            gain = 1.0 + tl.load(gamma_ptr + d).to(tl.float32)
            rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / D + EPS)
            normalized = (pooled * rms * gain).to(tl.bfloat16)

            partner = tl.where(
                d < ROTARY,
                tl.where(d < ROTARY // 2, d + ROTARY // 2, d - ROTARY // 2),
                d,
            )
            rotated_half = tl.gather(normalized, partner, axis=0)
            freq_idx = tl.where(d < ROTARY // 2, d, d - ROTARY // 2)
            inv_f = tl.load(inv_freq_ptr + freq_idx, mask=d < ROTARY, other=0.0)
            ang = (position - 3).to(tl.float32) * inv_f
            cosine = tl.where(d < ROTARY, libdevice.cos(ang).to(tl.bfloat16), tl.cast(1.0, tl.bfloat16))
            sine = tl.where(d < ROTARY, libdevice.sin(ang).to(tl.bfloat16), tl.cast(0.0, tl.bfloat16))
            left = (normalized.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
            right = (
                rotated_half.to(tl.float32)
                * sine.to(tl.float32)
                * tl.where(d < ROTARY // 2, -1.0, 1.0)
            ).to(tl.bfloat16)
            value = tl.where(
                d < ROTARY,
                (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
                normalized,
            )

            kv_column = position // KV_TOKENS_PER_BLOCK
            ok_kv = (position >= 0) & (kv_column < KV_TABLE_WIDTH)
            kv_id = tl.load(
                kv_table_ptr + request * stride_kv_b + tl.minimum(tl.maximum(kv_column, 0), KV_TABLE_WIDTH - 1) * stride_kv_p,
                mask=ok_kv,
                other=0,
            ).to(tl.int64)
            kv_valid = ok_kv & (kv_id > 0) & (kv_id < KV_POOL_BLOCKS)
            entry = (position // 4) % KV_ENTRIES_PER_BLOCK
            tl.store(
                kv_pool_ptr + (kv_id * KV_ENTRIES_PER_BLOCK + entry) * D + d,
                value,
                mask=kv_valid,
            )

    @triton.jit(do_not_specialize=["N", "B", "KV_TABLE_WIDTH", "STATE_TABLE_WIDTH", "KV_POOL_BLOCKS", "STATE_POOL_BLOCKS", "KV_TOKENS_PER_BLOCK", "STATE_TOKENS_PER_BLOCK", "KV_ENTRIES_PER_BLOCK"])
    def _write_prefill_qsa_kv_k4_kernel(
        raw_ptr,
        cu_seqlens_ptr,
        start_pos_ptr,
        pos_ptr,
        tok_req_ptr,
        inv_freq_ptr,
        gamma_ptr,
        kv_pool_ptr,
        kv_table_ptr,
        state_pool_ptr,
        state_table_ptr,
        N,
        B,
        stride_rn: tl.int64,
        stride_kv_b: tl.int64,
        stride_kv_p: tl.int64,
        stride_st_b: tl.int64,
        stride_st_p: tl.int64,
        KV_TABLE_WIDTH,
        STATE_TABLE_WIDTH,
        KV_POOL_BLOCKS,
        STATE_POOL_BLOCKS,
        KV_TOKENS_PER_BLOCK,
        STATE_TOKENS_PER_BLOCK,
        KV_ENTRIES_PER_BLOCK,
        HAS_TOK_REQ: tl.constexpr,
        RING_SIZE: tl.constexpr,
        D: tl.constexpr,
        ROTARY: tl.constexpr,
        EPS: tl.constexpr,
    ):
        """Pool completed 4-token blocks from raw_keys (+ pre-prefill state_pool) and write kv_pool."""
        n = tl.program_id(0).to(tl.int64)
        if n >= N:
            return
        pos = tl.load(pos_ptr + n).to(tl.int64)
        if pos < 0 or ((pos + 1) % 4) != 0:
            return

        if HAS_TOK_REQ:
            b = tl.load(tok_req_ptr + n).to(tl.int64)
        else:
            lo = 0
            hi = B
            while lo + 1 < hi:
                mid = (lo + hi) >> 1
                if tl.load(cu_seqlens_ptr + mid).to(tl.int64) <= n:
                    lo = mid
                else:
                    hi = mid
            b = lo

        kv_col = pos // KV_TOKENS_PER_BLOCK
        if kv_col >= KV_TABLE_WIDTH:
            return
        kv_id = tl.load(kv_table_ptr + b * stride_kv_b + kv_col * stride_kv_p).to(tl.int64)
        if kv_id <= 0 or kv_id >= KV_POOL_BLOCKS:
            return

        req_start = tl.load(cu_seqlens_ptr + b).to(tl.int64)
        local_t = n - req_start
        d = tl.arange(0, D)

        acc = tl.zeros((D,), dtype=tl.float32)
        for j in tl.static_range(4):
            src_local = local_t - (3 - j)
            if src_local >= 0:
                val = tl.load(raw_ptr + (req_start + src_local) * stride_rn + d).to(tl.float32)
            else:
                src_pos = pos - (3 - j)
                st_col = src_pos // STATE_TOKENS_PER_BLOCK
                ok_st = (src_pos >= 0) & (st_col >= 0) & (st_col < STATE_TABLE_WIDTH)
                st_id = tl.load(
                    state_table_ptr + b * stride_st_b + tl.minimum(tl.maximum(st_col, 0), STATE_TABLE_WIDTH - 1) * stride_st_p,
                    mask=ok_st,
                    other=0,
                ).to(tl.int64)
                ok_row = ok_st & (st_id > 0) & (st_id < STATE_POOL_BLOCKS)
                slot = st_id * RING_SIZE + (src_pos % RING_SIZE)
                val = tl.load(state_pool_ptr + slot * D + d, mask=ok_row, other=0.0).to(tl.float32)
            acc += val

        pooled = (acc * 0.25).to(tl.bfloat16).to(tl.float32)
        gain = 1.0 + tl.load(gamma_ptr + d).to(tl.float32)
        rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / D + EPS)
        normalized = (pooled * rms * gain).to(tl.bfloat16)

        partner = tl.where(
            d < ROTARY,
            tl.where(d < ROTARY // 2, d + ROTARY // 2, d - ROTARY // 2),
            d,
        )
        rotated_half = tl.gather(normalized, partner, axis=0)
        freq_idx = tl.where(d < ROTARY // 2, d, d - ROTARY // 2)
        inv_f = tl.load(inv_freq_ptr + freq_idx, mask=d < ROTARY, other=0.0)
        ang = (pos - 3).to(tl.float32) * inv_f
        cosine = tl.where(d < ROTARY, libdevice.cos(ang).to(tl.bfloat16), tl.cast(1.0, tl.bfloat16))
        sine = tl.where(d < ROTARY, libdevice.sin(ang).to(tl.bfloat16), tl.cast(0.0, tl.bfloat16))
        left = (normalized.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
        right = (
            rotated_half.to(tl.float32)
            * sine.to(tl.float32)
            * tl.where(d < ROTARY // 2, -1.0, 1.0)
        ).to(tl.bfloat16)
        value = tl.where(
            d < ROTARY,
            (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
            normalized,
        )
        entry = (pos // 4) % KV_ENTRIES_PER_BLOCK
        tl.store(kv_pool_ptr + (kv_id * KV_ENTRIES_PER_BLOCK + entry) * D + d, value)

    @triton.jit(do_not_specialize=["N", "B", "STATE_TABLE_WIDTH", "STATE_POOL_BLOCKS", "STATE_TOKENS_PER_BLOCK"])
    def _write_prefill_qsa_state_k4_kernel(
        raw_ptr,
        cu_seqlens_ptr,
        start_pos_ptr,
        pos_ptr,
        tok_req_ptr,
        state_pool_ptr,
        state_table_ptr,
        N,
        B,
        stride_rn: tl.int64,
        stride_st_b: tl.int64,
        stride_st_p: tl.int64,
        STATE_TABLE_WIDTH,
        STATE_POOL_BLOCKS,
        STATE_TOKENS_PER_BLOCK,
        HAS_TOK_REQ: tl.constexpr,
        RING_SIZE: tl.constexpr,
        D: tl.constexpr,
    ):
        """Write final ring entries per state page directly without cross-CTA races."""
        n = tl.program_id(0).to(tl.int64)
        if n >= N:
            return
        if HAS_TOK_REQ:
            b = tl.load(tok_req_ptr + n).to(tl.int64)
        else:
            lo = 0
            hi = B
            while lo + 1 < hi:
                mid = (lo + hi) >> 1
                if tl.load(cu_seqlens_ptr + mid).to(tl.int64) <= n:
                    lo = mid
                else:
                    hi = mid
            b = lo

        pos = tl.load(pos_ptr + n).to(tl.int64)
        if pos < 0:
            return
        req_len = tl.load(cu_seqlens_ptr + b + 1).to(tl.int64) - tl.load(cu_seqlens_ptr + b).to(tl.int64)
        req_end_pos = tl.load(start_pos_ptr + b).to(tl.int64) + req_len
        blk_end_pos = tl.minimum(req_end_pos, ((pos // STATE_TOKENS_PER_BLOCK) + 1) * STATE_TOKENS_PER_BLOCK)
        if pos + RING_SIZE < blk_end_pos:
            return

        st_col = pos // STATE_TOKENS_PER_BLOCK
        if st_col >= STATE_TABLE_WIDTH:
            return
        st_id = tl.load(state_table_ptr + b * stride_st_b + st_col * stride_st_p).to(tl.int64)
        if st_id <= 0 or st_id >= STATE_POOL_BLOCKS:
            return

        d = tl.arange(0, D)
        raw = tl.load(raw_ptr + n * stride_rn + d).to(state_pool_ptr.dtype.element_ty)
        tl.store(state_pool_ptr + (st_id * RING_SIZE + (pos % RING_SIZE)) * D + d, raw)


def build_qsa_neox_cos_sin(
    positions: torch.Tensor,
    rotary_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build NeoX partial RoPE (cos, sin) matching ``apply_partial_rope``."""
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be positive even, got {rotary_dim}")
    device = positions.device
    half = rotary_dim // 2
    inv_freq = _get_qsa_inv_freq(rotary_dim, rope_theta, device)
    if (
        _HAS_TRITON
        and positions.is_cuda
        and rotary_dim <= 256
        and (rotary_dim & (rotary_dim - 1)) == 0
        and dtype in (torch.bfloat16, torch.float16, torch.float32)
        and positions.ndim >= 1
        and positions.stride(-1) > 0
        and positions.numel() > 0
    ):
        pos_c = positions if (positions.ndim <= 1 or positions.is_contiguous()) else positions.contiguous()
        rows = int(pos_c.numel())
        cos_out = torch.empty((*positions.shape, rotary_dim), dtype=dtype, device=device)
        sin_out = torch.empty((*positions.shape, rotary_dim), dtype=dtype, device=device)
        block_r = 16 if rows >= 32 else max(1, triton.next_power_of_2(rows))
        block_r = min(block_r, 32)
        sp = int(pos_c.stride(-1)) if pos_c.ndim == 1 else 1
        ckey = (device.index, 0, pos_c.dtype, dtype, _triton_int_spec(rows), _triton_int_spec(sp), half, block_r)
        triton_fast_launch(
            _QSA_TRITON_CACHE,
            ckey,
            _build_qsa_neox_cos_sin_kernel,
            triton.cdiv(rows, block_r),
            1,
            1,
            pos_c.data_ptr() | inv_freq.data_ptr() | cos_out.data_ptr() | sin_out.data_ptr(),
            (pos_c, inv_freq, cos_out, sin_out, rows, sp),
            {"HALF": half, "BLOCK_R": block_r},
            num_warps=4,
        )
        return cos_out, sin_out

    angles = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos_h = torch.cos(angles)
    sin_h = torch.sin(angles)
    return (
        torch.cat([cos_h, cos_h], dim=-1).to(dtype=dtype),
        torch.cat([sin_h, sin_h], dim=-1).to(dtype=dtype),
    )


def view_indexer_kv_pool(
    cache_or_base: Any,
    *,
    kv_tokens_per_block: int,
    compress_ratio: int,
    head_dim: int,
) -> torch.Tensor:
    """Zero-copy view of ``indexer_kv`` as ``[num_blocks, entries_per_block, head_dim]`` BF16."""
    base = (
        cache_or_base.kv_cache_base
        if hasattr(cache_or_base, "kv_cache_base")
        else cache_or_base
    )
    if base is None:
        raise RuntimeError("indexer_kv cache pool base tensor is None")
    if kv_tokens_per_block <= 0 or kv_tokens_per_block % compress_ratio != 0:
        raise ValueError(
            f"kv_tokens_per_block ({kv_tokens_per_block}) must be positive and "
            f"divisible by compress_ratio ({compress_ratio})"
        )
    entries = kv_tokens_per_block // compress_ratio
    nb = int(base.shape[0])
    if base.dtype == torch.bfloat16:
        if base.ndim == 3 and tuple(base.shape[1:]) == (entries, head_dim):
            return base
        if base.ndim == 2 and int(base.shape[1]) == entries * head_dim:
            return base.view(nb, entries, head_dim)
    elif base.dtype == torch.uint8:
        need_bytes = entries * head_dim * 2
        if base.ndim == 3 and tuple(base.shape[1:]) == (entries, head_dim * 2):
            if not base.is_contiguous():
                base = base.contiguous()
            return base.view(torch.bfloat16).view(nb, entries, head_dim)
        if base.ndim == 2 and int(base.shape[1]) >= need_bytes:
            if not base.is_contiguous():
                base = base.contiguous()
            if int(base.shape[1]) == need_bytes:
                return base.view(torch.bfloat16).view(nb, entries, head_dim)
            return base[:, :need_bytes].reshape(nb, entries, head_dim * 2).view(
                torch.bfloat16
            )
    raise RuntimeError(
        f"Unexpected indexer_kv pool shape={tuple(base.shape)} dtype={base.dtype} "
        f"(entries={entries}, head_dim={head_dim})"
    )


def view_indexer_state_pool(
    cache_or_base: Any,
    *,
    compress_ratio: int,
    head_dim: int,
) -> torch.Tensor:
    """Zero-copy view of ``indexer_state`` as ``[num_blocks, 2*ratio, head_dim]`` FP32."""
    base = (
        cache_or_base.kv_cache_base
        if hasattr(cache_or_base, "kv_cache_base")
        else cache_or_base
    )
    if base is None:
        raise RuntimeError("indexer_state cache pool base tensor is None")
    ring_entries = 2 * compress_ratio
    nb = int(base.shape[0])
    if base.dtype == torch.float32:
        if base.ndim == 3 and tuple(base.shape[1:]) == (ring_entries, head_dim):
            return base
        if base.ndim == 2 and int(base.shape[1]) == ring_entries * head_dim:
            return base.view(nb, ring_entries, head_dim)
        if base.ndim == 2 and int(base.shape[1]) > ring_entries * head_dim:
            return base[:, : ring_entries * head_dim].reshape(
                nb, ring_entries, head_dim
            )
    elif base.dtype == torch.uint8:
        need_bytes = ring_entries * head_dim * 4
        if base.ndim == 2 and int(base.shape[1]) >= need_bytes:
            if not base.is_contiguous():
                base = base.contiguous()
            if int(base.shape[1]) == need_bytes:
                return base.view(torch.float32).view(nb, ring_entries, head_dim)
            return base[:, :need_bytes].reshape(nb, ring_entries, head_dim * 4).view(
                torch.float32
            )
    raise RuntimeError(
        f"Unexpected indexer_state pool shape={tuple(base.shape)} dtype={base.dtype} "
        f"(ring_entries={ring_entries}, head_dim={head_dim})"
    )


def view_mha_kv_hnd(
    kv_cache_base: torch.Tensor,
    *,
    kv_heads: int,
    page_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Zero-copy HND view ``[pages, 2, H_kv, page_size, D]`` for ``sparse_paged_gqa_attn``.

    Fixes the ``H_kv == 1`` NHD-vs-HND stride subtlety: calling ``.transpose(2, 3)``
    on NHD ``[pages, 2, page_size, 1, D]`` leaves ``stride(2) == D`` instead of
    ``page_size * D``, whereas ``.view(pages, 2, 1, page_size, D)`` yields exact
    HND inner strides ``(2*page_size*D, page_size*D, page_size*D, D, 1)``.
    """
    expected_inner = (
        kv_heads * page_size * head_dim,
        page_size * head_dim,
        head_dim,
        1,
    )
    if kv_cache_base.ndim == 2:
        # HybridPool packed 2D view is natively accepted by sparse_paged_gqa_attn.
        return kv_cache_base
    if kv_cache_base.ndim != 5 or int(kv_cache_base.shape[1]) != 2:
        raise ValueError(
            f"Expected rank-5 [pages, 2, ...] or rank-2 MHA pool, got {tuple(kv_cache_base.shape)}"
        )
    pages = int(kv_cache_base.shape[0])
    if (
        int(kv_cache_base.shape[2]) == kv_heads
        and int(kv_cache_base.shape[3]) == page_size
        and int(kv_cache_base.shape[4]) == head_dim
        and tuple(kv_cache_base.stride()[1:]) == expected_inner
    ):
        return kv_cache_base
    if (
        int(kv_cache_base.shape[2]) == page_size
        and int(kv_cache_base.shape[3]) == kv_heads
        and int(kv_cache_base.shape[4]) == head_dim
    ):
        if kv_heads == 1 and kv_cache_base.is_contiguous():
            return kv_cache_base.view(pages, 2, 1, page_size, head_dim)
        transposed = kv_cache_base.transpose(2, 3)
        if tuple(transposed.stride()[1:]) == expected_inner:
            return transposed
        return transposed.contiguous()
    raise ValueError(
        f"Unexpected MHA kv_cache_base shape={tuple(kv_cache_base.shape)} "
        f"strides={tuple(kv_cache_base.stride())} for (kv_heads={kv_heads}, page_size={page_size}, D={head_dim})"
    )


# ---------------------------------------------------------------------------
# Parameterized H_IDX Triton Kernels (Supports H=8 Qwen4.5 & H=4 Qwen3.8)
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit(do_not_specialize=["S", "N", "QB", "QS", "QH", "KB", "KN", "SCALE"])
    def _qsa_prefill_score_kernel(
        q_ptr,
        k_ptr,
        o_ptr,
        S,
        N,
        QB,
        QS,
        QH,
        KB,
        KN,
        SCALE,
        H_IDX: tl.constexpr,
        D_IDX: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        b = tl.program_id(0).to(tl.int64)
        group = tl.program_id(1)
        col = tl.program_id(2)
        rows_per_group: tl.constexpr = BM // H_IDX
        row = group * rows_per_group + tl.arange(0, BM) // H_IDX
        head = tl.arange(0, BM) % H_IDX
        d = tl.arange(0, D_IDX)
        n = col * BN + tl.arange(0, BN)

        qv = tl.load(
            q_ptr + b * QB + row[:, None] * QS + head[:, None] * QH + d[None, :],
            mask=row[:, None] < S,
            other=0.0,
        )
        kv = tl.load(
            k_ptr + b * KB + n[None, :] * KN + d[:, None],
            mask=n[None, :] < N,
            other=0.0,
        )
        logits = tl.dot(qv, kv)
        logits = tl.maximum(logits, 0.0)
        scores = tl.sum(tl.reshape(logits, (rows_per_group, H_IDX, BN)), axis=1) * SCALE
        pos = group * rows_per_group + tl.arange(0, rows_per_group)
        tl.store(
            o_ptr + (b * S + pos[:, None]) * N + n[None, :],
            scores,
            mask=(pos[:, None] < S) & (n[None, :] < N),
        )

    @triton.jit(
        do_not_specialize=["max_ctx_len", "max_blocks", "pool_blocks", "next_n", "SCALE"]
    )
    def _qsa_paged_score_kernel(
        q_ptr,
        pool_ptr,
        bt_ptr,
        ctx_ptr,
        out_ptr,
        stride_q_row,
        max_ctx_len,
        max_blocks,
        pool_blocks,
        next_n,
        SCALE,
        H_IDX: tl.constexpr,
        D_IDX: tl.constexpr,
        block_size: tl.constexpr,
        BN: tl.constexpr,
    ):
        rd = tl.arange(0, D_IDX)
        row = tl.program_id(0).to(tl.int64)
        blk = tl.program_id(1)
        batch_idx = row // next_n

        rn = blk * BN + tl.arange(0, BN)
        n_mask = rn < max_ctx_len

        T = tl.load(ctx_ptr + row)
        in_context = n_mask & (rn < T)

        bi = rn // block_size
        ei = rn % block_size
        in_block_table = bi < max_blocks
        physical_bt = tl.load(
            bt_ptr + batch_idx * max_blocks + bi,
            mask=in_context & in_block_table,
            other=0,
        )
        has_block = (
            in_context & in_block_table & (physical_bt > 0) & (physical_bt < pool_blocks)
        )
        slot = physical_bt.to(tl.int64) * block_size + ei

        acc = tl.zeros((BN,), dtype=tl.float32)
        for h in tl.static_range(H_IDX):
            qh = (
                tl.load(q_ptr + row * stride_q_row + h * D_IDX + rd)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            k = (
                tl.load(
                    pool_ptr + slot[:, None] * D_IDX + rd[None, :],
                    mask=has_block[:, None],
                    other=0.0,
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            dot = tl.sum(qh[None, :] * k, axis=1)
            acc += tl.where(has_block, tl.maximum(dot, 0.0) * SCALE, 0.0)

        tl.store(
            out_ptr + row * max_ctx_len + rn,
            tl.where(has_block, acc, float("-inf")),
            mask=n_mask,
        )

    @triton.jit(
        do_not_specialize=["NEXT_N", "max_ctx_len", "max_blocks", "pool_blocks", "SCALE"]
    )
    def _qsa_paged_score_tc_kernel(
        q_ptr,
        pool_ptr,
        bt_ptr,
        ctx_ptr,
        out_ptr,
        stride_qb: tl.int64,
        stride_qs: tl.int64,
        NEXT_N,
        max_ctx_len,
        max_blocks,
        pool_blocks,
        SCALE,
        H_IDX: tl.constexpr,
        D_IDX: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        QS: tl.constexpr,
        BN: tl.constexpr,
        IS_FP8_K: tl.constexpr,
    ):
        """SM100 TensorCore QSA paged indexer scorer grouping QS queries per K load."""
        b = tl.program_id(0).to(tl.int64)
        q_grp = tl.program_id(1).to(tl.int64)
        blk = tl.program_id(2).to(tl.int64)

        rn = blk * BN + tl.arange(0, BN)
        n_mask = rn < max_ctx_len
        bi = rn // BLOCK_SIZE
        ei = rn % BLOCK_SIZE
        in_bt = n_mask & (bi < max_blocks)
        phys = tl.load(bt_ptr + b * max_blocks + bi, mask=in_bt, other=0)
        has_blk = in_bt & (phys > 0) & (phys < pool_blocks)
        slot = tl.where(has_blk, phys.to(tl.int64) * BLOCK_SIZE + ei, 0)

        rd = tl.arange(0, D_IDX)
        k_mat = tl.load(
            pool_ptr + slot[:, None] * D_IDX + rd[None, :],
            mask=has_blk[:, None],
            other=0.0,
        )

        SUB_QS: tl.constexpr = 8 if QS >= 8 else QS
        SUB_GRPS: tl.constexpr = QS // SUB_QS
        M_Q: tl.constexpr = SUB_QS * H_IDX
        rm = tl.arange(0, M_Q)
        q_local = rm // H_IDX
        q_head = rm % H_IDX

        for sg in tl.static_range(SUB_GRPS):
            s_offs = q_grp * QS + sg * SUB_QS + q_local
            s_live = s_offs < NEXT_N
            qv = tl.load(
                q_ptr
                + b * stride_qb
                + s_offs[:, None] * stride_qs
                + q_head[:, None] * D_IDX
                + rd[None, :],
                mask=s_live[:, None],
                other=0.0,
            )
            if IS_FP8_K:
                qv = qv.to(tl.float8e4nv)
            logits = tl.dot(qv, tl.trans(k_mat))
            logits = tl.maximum(logits, 0.0)
            scores = tl.sum(tl.reshape(logits, (SUB_QS, H_IDX, BN)), axis=1) * SCALE

            s_idx = q_grp * QS + sg * SUB_QS + tl.arange(0, SUB_QS)
            s_mask = s_idx < NEXT_N
            vis_s = tl.load(ctx_ptr + b * NEXT_N + s_idx, mask=s_mask, other=0)
            valid = s_mask[:, None] & has_blk[None, :] & (rn[None, :] < vis_s[:, None])
            out_vals = tl.where(valid, scores, float("-inf"))
            tl.store(
                out_ptr + (b * NEXT_N + s_idx[:, None]) * max_ctx_len + rn[None, :],
                out_vals,
                mask=s_mask[:, None] & n_mask[None, :],
            )

    @triton.jit(do_not_specialize=["stride_s", "stride_l", "stride_o", "N"])
    def _qsa_bsearch_topk_expand_kernel(
        scores_ptr,
        len_ptr,
        out_ptr,
        stride_s: tl.int64,
        stride_l: tl.int64,
        stride_o: tl.int64,
        N,
        BN: tl.constexpr,
        K_BLKS: tl.constexpr,
        RATIO: tl.constexpr,
        BUDGET: tl.constexpr,
    ):
        """Single-launch exact binary-search block Top-K + token expansion (CG-safe)."""
        row = tl.program_id(0).to(tl.int64)
        vis = tl.load(len_ptr + row * stride_l)
        complete = tl.maximum(vis // RATIO, 0)
        cols = tl.arange(0, BN)
        in_c = (cols < complete) & (cols < N)
        vals = tl.load(scores_ptr + row * stride_s + cols, mask=in_c, other=float("-inf")).to(
            tl.float32
        )
        is_fin = in_c & (vals > float("-inf")) & (vals < float("inf"))
        bits = vals.to(tl.uint32, bitcast=True)
        ord_u = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        ord_u = tl.where(is_fin, ord_u, 0)

        valid_cnt = tl.sum(is_fin.to(tl.int32), 0)
        k_rem = tl.minimum(K_BLKS, valid_cnt)
        lo = tl.zeros((), dtype=tl.uint32)
        hi = tl.full((), 0xFFFFFFFF, tl.uint32)
        for _bit in tl.static_range(32):
            mid = lo + ((hi - lo) >> 1) + 1
            cnt = tl.sum((is_fin & (ord_u >= mid)).to(tl.int32), 0)
            lo = tl.where(cnt >= k_rem, mid, lo)
            hi = tl.where(cnt < k_rem, mid - 1, hi)

        gt_m = is_fin & (ord_u > lo)
        gt_cnt = tl.sum(gt_m.to(tl.int32), 0)
        eq_m = is_fin & (ord_u == lo)
        eq_rank = tl.cumsum(eq_m.to(tl.int32), 0) - 1
        sel_m = gt_m | (eq_m & (gt_cnt + eq_rank < k_rem))
        sel_pos = tl.cumsum(sel_m.to(tl.int32), 0) - 1

        out_row = out_ptr + row * stride_o
        for r in tl.static_range(RATIO):
            tl.store(
                out_row + sel_pos * RATIO + r,
                cols * RATIO + r,
                mask=sel_m & (sel_pos < K_BLKS),
            )
        if k_rem < K_BLKS:
            tail_s = tl.arange(0, K_BLKS)
            for r in tl.static_range(RATIO):
                tl.store(out_row + tail_s * RATIO + r, -1, mask=(tail_s >= k_rem))

        tail_cnt = tl.maximum(vis, 0) % RATIO
        for r in tl.static_range(RATIO - 1):
            tl.store(out_row + BUDGET + r, complete * RATIO + r if r < tail_cnt else -1)

    @triton.jit(do_not_specialize=["stride_s", "stride_l", "stride_o", "N"])
    def _qsa_radix2_chunked_topk_expand_kernel(
        scores_ptr,
        len_ptr,
        out_ptr,
        stride_s: tl.int64,
        stride_l: tl.int64,
        stride_o: tl.int64,
        N,
        CHUNK: tl.constexpr,
        N_CHUNKS: tl.constexpr,
        K_BLKS: tl.constexpr,
        RATIO: tl.constexpr,
        BUDGET: tl.constexpr,
    ):
        """CG-safe 2-bit radix-select chunked Top-K + token expansion for long contexts (N <= 32768)."""
        row = tl.program_id(0).to(tl.int64)
        vis = tl.load(len_ptr + row * stride_l)
        complete = tl.maximum(vis // RATIO, 0)

        valid_cnt = tl.zeros((), dtype=tl.int32)
        for c in tl.static_range(N_CHUNKS):
            offs = c * CHUNK + tl.arange(0, CHUNK)
            in_c = (offs < complete) & (offs < N)
            vals = (
                tl.load(scores_ptr + row * stride_s + offs, mask=in_c, other=float("-inf")).to(
                    tl.float32
                )
                + 0.0
            )
            is_fin = in_c & (vals > float("-inf")) & (vals < float("inf"))
            valid_cnt += tl.sum(is_fin.to(tl.int32), 0)

        k_tot = tl.minimum(K_BLKS, valid_cnt)
        lo = tl.zeros((), dtype=tl.uint32)
        hi = tl.full((), 0xFFFFFFFF, tl.uint32)
        k_rem = k_tot
        if k_tot > 0:
            for p in tl.static_range(16):
                shift = 30 - 2 * p
                packed_cnt = tl.zeros((), dtype=tl.int64)
                for c in tl.static_range(N_CHUNKS):
                    offs = c * CHUNK + tl.arange(0, CHUNK)
                    in_c = (offs < complete) & (offs < N)
                    vals = (
                        tl.load(scores_ptr + row * stride_s + offs, mask=in_c, other=float("-inf")).to(
                            tl.float32
                        )
                        + 0.0
                    )
                    is_fin = in_c & (vals > float("-inf")) & (vals < float("inf"))
                    bits = vals.to(tl.uint32, bitcast=True)
                    ord_u = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
                    act = is_fin & (ord_u >= lo) & (ord_u <= hi)
                    q2 = (ord_u >> shift) & 3
                    w123 = (
                        tl.where(act & (q2 == 1), 1, 0).to(tl.int64)
                        + (tl.where(act & (q2 == 2), 1, 0).to(tl.int64) << 20)
                        + (tl.where(act & (q2 == 3), 1, 0).to(tl.int64) << 40)
                    )
                    packed_cnt += tl.sum(w123, 0)
                c3 = (packed_cnt >> 40).to(tl.int32)
                c2 = ((packed_cnt >> 20) & 0xFFFFF).to(tl.int32)
                c1 = (packed_cnt & 0xFFFFF).to(tl.int32)
                b = tl.where(
                    k_rem <= c3,
                    3,
                    tl.where(k_rem <= c3 + c2, 2, tl.where(k_rem <= c3 + c2 + c1, 1, 0)),
                ).to(tl.uint32)
                sub = tl.where(
                    b == 3, 0, tl.where(b == 2, c3, tl.where(b == 1, c3 + c2, c3 + c2 + c1))
                )
                k_rem = k_rem - sub
                step_span = (1 << shift) - 1
                lo = lo + (b << tl.full((), shift, tl.uint32))
                hi = lo + tl.full((), step_span, tl.uint32)

        out_row = out_ptr + row * stride_o
        if k_tot > 0:
            eq_take = k_rem
            total_gt = k_tot - eq_take
            gt_ptr = tl.zeros((), dtype=tl.int32)
            eq_ptr = total_gt
            eq_rem = eq_take
            for c in tl.static_range(N_CHUNKS):
                offs = c * CHUNK + tl.arange(0, CHUNK)
                in_c = (offs < complete) & (offs < N)
                vals = (
                    tl.load(scores_ptr + row * stride_s + offs, mask=in_c, other=float("-inf")).to(
                        tl.float32
                    )
                    + 0.0
                )
                is_fin = in_c & (vals > float("-inf")) & (vals < float("inf"))
                bits = vals.to(tl.uint32, bitcast=True)
                ord_u = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
                gt_m = is_fin & (ord_u > lo)
                eq_m = is_fin & (ord_u == lo)
                gt_rank = tl.cumsum(gt_m.to(tl.int32), 0) - 1
                gt_pos = gt_ptr + gt_rank
                eq_rank = tl.cumsum(eq_m.to(tl.int32), 0) - 1
                take_eq = eq_m & (eq_rank < eq_rem)
                eq_pos = eq_ptr + eq_rank
                sel_m = gt_m | take_eq
                sel_pos = tl.where(gt_m, gt_pos, eq_pos)
                for r in tl.static_range(RATIO):
                    tl.store(
                        out_row + sel_pos * RATIO + r,
                        offs * RATIO + r,
                        mask=sel_m & (sel_pos < K_BLKS),
                    )
                gt_ptr += tl.sum(gt_m.to(tl.int32), 0)
                eq_c = tl.sum(take_eq.to(tl.int32), 0)
                eq_ptr += eq_c
                eq_rem -= eq_c

        if k_tot < K_BLKS:
            tail_s = tl.arange(0, K_BLKS)
            for r in tl.static_range(RATIO):
                tl.store(out_row + tail_s * RATIO + r, -1, mask=(tail_s >= k_tot))

        tail_cnt = tl.maximum(vis, 0) % RATIO
        for r in tl.static_range(RATIO - 1):
            tl.store(out_row + BUDGET + r, complete * RATIO + r if r < tail_cnt else -1)

    @triton.jit(do_not_specialize=["stride_s", "stride_l", "stride_cv", "N"])
    def _qsa_tiled_topk_stage1_kernel(
        scores_ptr,
        len_ptr,
        cand_val_ptr,
        cand_col_ptr,
        stride_s: tl.int64,
        stride_l: tl.int64,
        stride_cv: tl.int64,
        N,
        TILE_N: tl.constexpr,
        K_BLKS: tl.constexpr,
        RATIO: tl.constexpr,
    ):
        """Stage 1 of CG-safe 2-stage tiled exact Top-K: select <= K_BLKS per tile in one HBM read."""
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        vis = tl.load(len_ptr + row * stride_l)
        complete = tl.maximum(vis // RATIO, 0)
        cols = tile * TILE_N + tl.arange(0, TILE_N)
        in_c = (cols < complete) & (cols < N)
        vals = tl.load(
            scores_ptr + row * stride_s + cols, mask=in_c, other=float("-inf")
        ).to(tl.float32)
        is_fin = in_c & (vals > float("-inf")) & (vals < float("inf"))
        bits = vals.to(tl.uint32, bitcast=True)
        ord_u = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        ord_u = tl.where(is_fin, ord_u, 0)

        valid_cnt = tl.sum(is_fin.to(tl.int32), 0)
        out_base = row * stride_cv + tile * K_BLKS
        if valid_cnt == 0:
            sk = tl.arange(0, K_BLKS)
            tl.store(cand_val_ptr + out_base + sk, float("-inf"))
            tl.store(cand_col_ptr + out_base + sk, -1)
            return

        if valid_cnt <= K_BLKS:
            sel_pos = tl.cumsum(is_fin.to(tl.int32), 0) - 1
            sk = tl.arange(0, K_BLKS)
            tl.store(cand_val_ptr + out_base + sk, float("-inf"))
            tl.store(cand_col_ptr + out_base + sk, -1)
            tl.store(
                cand_val_ptr + out_base + sel_pos,
                vals,
                mask=is_fin & (sel_pos < K_BLKS),
            )
            tl.store(
                cand_col_ptr + out_base + sel_pos,
                cols.to(tl.int32),
                mask=is_fin & (sel_pos < K_BLKS),
            )
            return

        lo = tl.zeros((), dtype=tl.uint32)
        hi = tl.full((), 0xFFFFFFFF, tl.uint32)
        for _bit in tl.static_range(32):
            mid = lo + ((hi - lo) >> 1) + 1
            cnt = tl.sum((is_fin & (ord_u >= mid)).to(tl.int32), 0)
            lo = tl.where(cnt >= K_BLKS, mid, lo)
            hi = tl.where(cnt < K_BLKS, mid - 1, hi)

        gt_m = is_fin & (ord_u > lo)
        gt_cnt = tl.sum(gt_m.to(tl.int32), 0)
        eq_m = is_fin & (ord_u == lo)
        eq_rank = tl.cumsum(eq_m.to(tl.int32), 0) - 1
        sel_m = gt_m | (eq_m & (gt_cnt + eq_rank < K_BLKS))
        sel_pos = tl.cumsum(sel_m.to(tl.int32), 0) - 1
        tl.store(
            cand_val_ptr + out_base + sel_pos,
            vals,
            mask=sel_m & (sel_pos < K_BLKS),
        )
        tl.store(
            cand_col_ptr + out_base + sel_pos,
            cols.to(tl.int32),
            mask=sel_m & (sel_pos < K_BLKS),
        )

    @triton.jit(do_not_specialize=["stride_cv", "stride_l", "stride_o", "VALID_CANDS"])
    def _qsa_tiled_topk_stage2_kernel(
        cand_val_ptr,
        cand_col_ptr,
        len_ptr,
        out_ptr,
        stride_cv: tl.int64,
        stride_l: tl.int64,
        stride_o: tl.int64,
        VALID_CANDS,
        N_CANDS: tl.constexpr,
        K_BLKS: tl.constexpr,
        RATIO: tl.constexpr,
        BUDGET: tl.constexpr,
    ):
        """Stage 2 of CG-safe 2-stage tiled exact Top-K: merge candidates and expand tokens."""
        row = tl.program_id(0).to(tl.int64)
        vis = tl.load(len_ptr + row * stride_l)
        complete = tl.maximum(vis // RATIO, 0)
        offs = tl.arange(0, N_CANDS)
        in_v = offs < VALID_CANDS
        vals = tl.load(
            cand_val_ptr + row * stride_cv + offs, mask=in_v, other=float("-inf")
        ).to(tl.float32)
        orig_cols = tl.load(
            cand_col_ptr + row * stride_cv + offs, mask=in_v, other=-1
        )
        is_fin = (
            in_v
            & (orig_cols >= 0)
            & (vals > float("-inf"))
            & (vals < float("inf"))
        )
        bits = vals.to(tl.uint32, bitcast=True)
        ord_u = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        ord_u = tl.where(is_fin, ord_u, 0)

        valid_cnt = tl.sum(is_fin.to(tl.int32), 0)
        k_rem = tl.minimum(K_BLKS, valid_cnt)
        lo = tl.zeros((), dtype=tl.uint32)
        hi = tl.full((), 0xFFFFFFFF, tl.uint32)
        for _bit in tl.static_range(32):
            mid = lo + ((hi - lo) >> 1) + 1
            cnt = tl.sum((is_fin & (ord_u >= mid)).to(tl.int32), 0)
            lo = tl.where(cnt >= k_rem, mid, lo)
            hi = tl.where(cnt < k_rem, mid - 1, hi)

        gt_m = is_fin & (ord_u > lo)
        gt_cnt = tl.sum(gt_m.to(tl.int32), 0)
        eq_m = is_fin & (ord_u == lo)
        eq_rank = tl.cumsum(eq_m.to(tl.int32), 0) - 1
        sel_m = gt_m | (eq_m & (gt_cnt + eq_rank < k_rem))
        sel_pos = tl.cumsum(sel_m.to(tl.int32), 0) - 1

        out_row = out_ptr + row * stride_o
        for r in tl.static_range(RATIO):
            tl.store(
                out_row + sel_pos * RATIO + r,
                orig_cols * RATIO + r,
                mask=sel_m & (sel_pos < K_BLKS),
            )
        if k_rem < K_BLKS:
            tail_s = tl.arange(0, K_BLKS)
            for r in tl.static_range(RATIO):
                tl.store(out_row + tail_s * RATIO + r, -1, mask=(tail_s >= k_rem))

        tail_cnt = tl.maximum(vis, 0) % RATIO
        for r in tl.static_range(RATIO - 1):
            tl.store(out_row + BUDGET + r, complete * RATIO + r if r < tail_cnt else -1)


def qsa_prefill_score(
    q_rot: torch.Tensor,
    block_keys: torch.Tensor,
) -> torch.Tensor:
    """Compute ReLU-gated weighted block scores ``[B, S, N]`` FP32.

    Uses ``_qsa_prefill_score_kernel`` parameterized by ``H_IDX: tl.constexpr``
    on CUDA (supporting ``H=8`` and ``H=4`` across arbitrary ``S, N >= 1``), with
    an exact FP32 PyTorch fallback.
    """
    bsz, seq_len, n_heads, head_dim = q_rot.shape
    num_blocks = int(block_keys.shape[1])
    if seq_len == 0 or num_blocks == 0:
        return torch.empty((bsz, seq_len, num_blocks), device=q_rot.device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(head_dim)
    if (
        _HAS_TRITON
        and q_rot.is_cuda
        and q_rot.dtype == torch.bfloat16
        and block_keys.dtype == torch.bfloat16
        and q_rot.stride(-1) == 1
        and block_keys.stride(-1) == 1
        and head_dim > 0
        and (head_dim & (head_dim - 1)) == 0
        and n_heads > 0
        and (n_heads & (n_heads - 1)) == 0
    ):
        if seq_len >= 128 and num_blocks >= 128 and n_heads <= 8:
            bm, bn, nw, stg = 128, 128, 8, 2
        else:
            bm, bn, nw, stg = max(64, 4 * n_heads), 64, 4, 1
        rows_per_group = bm // n_heads
        out = torch.empty(
            (bsz, seq_len, num_blocks), device=q_rot.device, dtype=torch.float32
        )
        grid = (
            bsz,
            triton.cdiv(seq_len, rows_per_group),
            triton.cdiv(num_blocks, bn),
        )
        _qsa_prefill_score_kernel[grid](
            q_rot,
            block_keys,
            out,
            S=seq_len,
            N=num_blocks,
            QB=q_rot.stride(0),
            QS=q_rot.stride(1),
            QH=q_rot.stride(2),
            KB=block_keys.stride(0),
            KN=block_keys.stride(1),
            SCALE=scale,
            H_IDX=n_heads,
            D_IDX=head_dim,
            BM=bm,
            BN=bn,
            num_warps=nw,
            num_stages=stg,
        )
        return out

    logits = torch.matmul(
        q_rot.float(), block_keys.float().transpose(-1, -2).unsqueeze(1)
    )
    return torch.relu(logits.transpose(-1, -2)).sum(dim=-1) * scale


def qsa_paged_score(
    q_rot_bf16: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    max_ctx_len: int,
) -> torch.Tensor:
    """Zero-sync paged indexer scoring ``[B*next_n, max_ctx_len]`` FP32.

    Runs ``_qsa_paged_score_kernel`` parameterized by ``H_IDX: tl.constexpr`` on
    CUDA (supports ``H=8`` & ``H=4`` and any positive ``block_size``), with CPU
    reference fallback.
    """
    bsz, next_n, n_heads, head_dim = q_rot_bf16.shape
    m_rows = bsz * next_n
    if m_rows == 0 or max_ctx_len <= 0:
        return torch.empty(
            (m_rows, max(0, max_ctx_len)), device=q_rot_bf16.device, dtype=torch.float32
        )

    kv_entries_per_block = int(kv_pool.shape[1])
    pool_blocks = int(kv_pool.shape[0])
    max_blocks = int(kv_table.shape[1])
    scale = 1.0 / math.sqrt(head_dim)

    if _HAS_TRITON and q_rot_bf16.is_cuda:
        q_c = q_rot_bf16.contiguous()
        pool_flat = kv_pool.contiguous().flatten(0, 1)
        bt_c = kv_table.contiguous()
        ctx_c = context_lens.to(dtype=torch.int32).contiguous().view(-1)
        logits = torch.empty(
            (m_rows, max_ctx_len), device=q_rot_bf16.device, dtype=torch.float32
        )
        is_fp8_k = kv_pool.dtype == torch.float8_e4m3fn
        cap_major = (
            torch.cuda.get_device_capability(q_rot_bf16.device)[0]
            if torch.version.hip is None
            else 0
        )
        can_tc = (
            cap_major >= 8
            and (kv_pool.dtype == torch.bfloat16 or (is_fp8_k and cap_major >= 9))
            and q_c.dtype == torch.bfloat16
            and head_dim >= 16
            and (head_dim & (head_dim - 1)) == 0
            and 1 <= n_heads <= 16
            and (n_heads & (n_heads - 1)) == 0
            and max_ctx_len >= 16
        )
        if can_tc:
            min_qs = max(1, 16 // n_heads)
            if next_n >= 128 and bsz * next_n >= 1024 and max_ctx_len >= 64:
                qs = 32 if n_heads <= 8 else 16
                bn = 64
            else:
                qs = max(min_qs, min(8, triton.next_power_of_2(max(1, next_n))))
                bn = 32 if (bsz * triton.cdiv(max_ctx_len, 64) < 32 and max_ctx_len >= 32) else 64
            if max_ctx_len < 32:
                bn = 16
            if qs == 8 and next_n == 8 and bsz <= 2 and min_qs <= 4:
                qs = 4
            ns_tc = 2 if bn >= 32 else 1
            rt_tc = (
                q_c, pool_flat, bt_c, ctx_c, logits,
                q_c.stride(0), q_c.stride(1), next_n, max_ctx_len, max_blocks, pool_blocks, scale,
            )
            ptr_tc = q_c.data_ptr() | pool_flat.data_ptr() | bt_c.data_ptr() | ctx_c.data_ptr() | logits.data_ptr()
            ck_tc = (
                q_c.device.index, 200, q_c.dtype, pool_flat.dtype, bt_c.dtype, ctx_c.dtype, logits.dtype,
                tuple(_triton_int_spec(v_s) for v_s in rt_tc[5:7]),
                float(scale), n_heads, head_dim, kv_entries_per_block, qs, bn, is_fp8_k, ns_tc,
            )
            triton_fast_launch(
                _QSA_TRITON_CACHE, ck_tc, _qsa_paged_score_tc_kernel,
                bsz, triton.cdiv(next_n, qs), triton.cdiv(max_ctx_len, bn),
                ptr_tc, rt_tc,
                {
                    "H_IDX": n_heads, "D_IDX": head_dim, "BLOCK_SIZE": kv_entries_per_block,
                    "QS": qs, "BN": bn, "IS_FP8_K": is_fp8_k,
                },
                num_warps=4, num_stages=ns_tc,
            )
            return logits
        grid = (m_rows, triton.cdiv(max_ctx_len, _PAGED_SCORE_BN))
        _qsa_paged_score_kernel[grid](
            q_c,
            pool_flat if not is_fp8_k else pool_flat.to(torch.bfloat16),
            bt_c,
            ctx_c,
            logits,
            stride_q_row=n_heads * head_dim,
            max_ctx_len=max_ctx_len,
            max_blocks=max_blocks,
            pool_blocks=pool_blocks,
            next_n=next_n,
            SCALE=scale,
            H_IDX=n_heads,
            D_IDX=head_dim,
            block_size=kv_entries_per_block,
            BN=_PAGED_SCORE_BN,
            num_warps=4,
        )
        return logits

    # CPU / pure-PyTorch reference fallback
    logits = torch.full(
        (m_rows, max_ctx_len), float("-inf"), device=q_rot_bf16.device, dtype=torch.float32
    )
    q_flat = q_rot_bf16.reshape(m_rows, n_heads, head_dim).float()
    ctx_flat = context_lens.reshape(-1)
    for row in range(m_rows):
        b = row // next_n
        t_len = min(int(ctx_flat[row].item()), max_ctx_len)
        if t_len <= 0:
            continue
        for c in range(t_len):
            bi = c // kv_entries_per_block
            ei = c % kv_entries_per_block
            if bi >= max_blocks:
                continue
            phys = int(kv_table[b, bi].item())
            if phys <= 0 or phys >= pool_blocks:
                continue
            k_vec = kv_pool[phys, ei].float()
            dot = torch.matmul(q_flat[row], k_vec)
            logits[row, c] = torch.relu(dot).sum() * scale
    return logits


# ---------------------------------------------------------------------------
# Prefill & Decode QSA Selection / Cache Write / Sparse Attention
# ---------------------------------------------------------------------------


def _select_top_tokens(
    block_scores: torch.Tensor,
    visible_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    token_budget: int,
    scores_are_masked: bool,
) -> torch.Tensor:
    """Zero-sync token index expansion with CG-safe Triton TopK fast path."""
    vis_1d = visible_lengths.to(dtype=torch.int32).reshape(-1)
    rows = int(block_scores.shape[0])
    cols = int(block_scores.shape[1]) if block_scores.ndim == 2 else 0
    if (
        _HAS_TRITON
        and block_scores.is_cuda
        and torch.version.hip is None
        and block_scores.dtype == torch.float32
        and block_scores.ndim == 2
        and block_scores.stride(-1) == 1
        and compress_ratio == 4
        and token_budget > 0
        and token_budget % compress_ratio == 0
        and 1 <= cols <= 32768
        and (rows <= 1024 or (cols <= 2048 and rows <= 16384))
        and torch.cuda.get_device_capability(block_scores.device)[0] >= 8
    ):
        k_blks = token_budget // compress_ratio
        if k_blks <= 1024 and (k_blks & (k_blks - 1)) == 0:
            if cols <= 8192:
                bn = max(64, triton.next_power_of_2(cols))
                out = torch.empty(
                    (rows, token_budget + compress_ratio - 1),
                    device=block_scores.device,
                    dtype=torch.int32,
                )
                nw = 4 if (rows > 1024 and bn <= 2048) else (8 if bn >= 4096 else 4)
                rt_bs = (block_scores, vis_1d, out, int(block_scores.stride(0)), int(vis_1d.stride(0)), int(out.stride(0)), cols)
                ptr_bs = block_scores.data_ptr() | vis_1d.data_ptr() | out.data_ptr()
                ck_bs = (
                    block_scores.device.index, 201, block_scores.dtype, vis_1d.dtype, out.dtype,
                    bn, k_blks, compress_ratio, token_budget, nw,
                )
                triton_fast_launch(
                    _QSA_TRITON_CACHE, ck_bs, _qsa_bsearch_topk_expand_kernel,
                    rows, 1, 1, ptr_bs, rt_bs,
                    {"BN": bn, "K_BLKS": k_blks, "RATIO": compress_ratio, "BUDGET": token_budget},
                    num_warps=nw,
                )
                return out
            elif rows <= 128 or (cols > 16384 and rows <= 1024):
                out = torch.empty(
                    (rows, token_budget + compress_ratio - 1),
                    device=block_scores.device,
                    dtype=torch.int32,
                )
                tile_n = 4096 if (cols <= 16384 and rows <= 64 and k_blks <= 512) else (4096 if k_blks == 1024 else 2048)
                n_tiles = triton.cdiv(cols, tile_n)
                valid_cands = n_tiles * k_blks
                n_cands = max(64, triton.next_power_of_2(valid_cands))
                if n_cands <= 8192:
                    cand_val = torch.empty(
                        (rows, valid_cands),
                        device=block_scores.device,
                        dtype=torch.float32,
                    )
                    cand_col = torch.empty(
                        (rows, valid_cands),
                        device=block_scores.device,
                        dtype=torch.int32,
                    )
                    nw1 = 8 if (tile_n == 4096 or rows <= 16) else 4
                    nw2 = 8 if n_cands >= 4096 else 4
                    rt_t1 = (
                        block_scores, vis_1d, cand_val, cand_col,
                        int(block_scores.stride(0)), int(vis_1d.stride(0)), int(cand_val.stride(0)), cols,
                    )
                    ptr_t1 = block_scores.data_ptr() | vis_1d.data_ptr() | cand_val.data_ptr() | cand_col.data_ptr()
                    ck_t1 = (
                        block_scores.device.index, 204, block_scores.dtype, vis_1d.dtype, cand_val.dtype, cand_col.dtype,
                        tile_n, k_blks, compress_ratio, nw1,
                    )
                    triton_fast_launch(
                        _QSA_TRITON_CACHE, ck_t1, _qsa_tiled_topk_stage1_kernel,
                        rows, n_tiles, 1, ptr_t1, rt_t1,
                        {"TILE_N": tile_n, "K_BLKS": k_blks, "RATIO": compress_ratio},
                        num_warps=nw1,
                    )
                    rt_t2 = (
                        cand_val, cand_col, vis_1d, out,
                        int(cand_val.stride(0)), int(vis_1d.stride(0)), int(out.stride(0)), valid_cands,
                    )
                    ptr_t2 = cand_val.data_ptr() | cand_col.data_ptr() | vis_1d.data_ptr() | out.data_ptr()
                    ck_t2 = (
                        block_scores.device.index, 205, cand_val.dtype, cand_col.dtype, vis_1d.dtype, out.dtype,
                        n_cands, k_blks, compress_ratio, token_budget, nw2,
                    )
                    triton_fast_launch(
                        _QSA_TRITON_CACHE, ck_t2, _qsa_tiled_topk_stage2_kernel,
                        rows, 1, 1, ptr_t2, rt_t2,
                        {"N_CANDS": n_cands, "K_BLKS": k_blks, "RATIO": compress_ratio, "BUDGET": token_budget},
                        num_warps=nw2,
                    )
                    return out
                if rows <= 128:
                    n_chunks = triton.cdiv(cols, 8192)
                    rt_r2 = (
                        block_scores, vis_1d, out,
                        int(block_scores.stride(0)), int(vis_1d.stride(0)), int(out.stride(0)), cols,
                    )
                    ptr_r2 = block_scores.data_ptr() | vis_1d.data_ptr() | out.data_ptr()
                    ck_r2 = (
                        block_scores.device.index, 203, block_scores.dtype, vis_1d.dtype, out.dtype,
                        n_chunks, k_blks, compress_ratio, token_budget,
                    )
                    triton_fast_launch(
                        _QSA_TRITON_CACHE, ck_r2, _qsa_radix2_chunked_topk_expand_kernel,
                        rows, 1, 1, ptr_r2, rt_r2,
                        {"CHUNK": 8192, "N_CHUNKS": n_chunks, "K_BLKS": k_blks, "RATIO": compress_ratio, "BUDGET": token_budget},
                        num_warps=8,
                    )
                    return out

    return select_qsa_paged_tokens(
        block_scores,
        vis_1d,
        compress_ratio=compress_ratio,
        token_budget=token_budget,
        validate_lengths=False,
        scores_are_masked=scores_are_masked,
    )


def _cp_all_gather_dim0(
    local_buf: torch.Tensor,
    *,
    cp_size: int,
    cp_group: Any,
    fast_all_reduce: Any = None,
) -> torch.Tensor:
    """Gather `[M_loc, *tail]` -> `[cp_size * M_loc, *tail]` across `cp_group`."""
    local_c = local_buf if local_buf.is_contiguous() else local_buf.contiguous()
    if (
        fast_all_reduce is not None
        and getattr(fast_all_reduce, "group", None) is cp_group
        and hasattr(fast_all_reduce, "all_gather")
    ):
        return fast_all_reduce.all_gather(local_c, dim=0)
    if not local_c.is_cuda:
        chunks = [torch.empty_like(local_c) for _ in range(cp_size)]
        torch.distributed.all_gather(chunks, local_c, group=cp_group)
        return torch.cat(chunks, dim=0)
    try:
        backend = str(torch.distributed.get_backend(cp_group)).lower()
    except Exception:
        backend = ""
    if "gloo" in backend:
        cpu_in = local_c.detach().cpu()
        chunks_cpu = [torch.empty_like(cpu_in) for _ in range(cp_size)]
        torch.distributed.all_gather(chunks_cpu, cpu_in, group=cp_group)
        return torch.cat(chunks_cpu, dim=0).to(device=local_c.device, dtype=local_c.dtype)
    gathered = torch.empty(
        (cp_size * int(local_c.shape[0]), *local_c.shape[1:]),
        dtype=local_c.dtype,
        device=local_c.device,
    )
    torch.distributed.all_gather_into_tensor(
        gathered.view(cp_size, *local_c.shape),
        local_c,
        group=cp_group,
    )
    return gathered


def _cp_use_batch_shard(bsz: int, seq_len: int, cp_size: int) -> bool:
    return bsz >= cp_size and (seq_len <= 32 or (bsz % cp_size) == 0)


def qsa_select_causal_from_projected(
    indexer: Qwen4ExpQSAIndexer,
    q_unrot: torch.Tensor,
    raw_keys: torch.Tensor,
    cos_all: Optional[torch.Tensor] = None,
    sin_all: Optional[torch.Tensor] = None,
    *,
    past_raw_keys: Optional[torch.Tensor] = None,
    k_norm_gamma: Optional[torch.Tensor] = None,
    q_norm_gamma: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    rotary_dim: int = 64,
    rope_theta: float = 10000.0,
    sp_size: int = 1,
    sp_rank: int = 0,
    sp_group: Any = None,
    fast_all_reduce: Any = None,
) -> torch.Tensor:
    """Select causal QSA tokens ``[B, S, max_selected]`` for standalone/cold prefill.

    Supports both ``indexer.k_norm_gamma`` (source layer) and an overridden
    ``k_norm_gamma`` (YoCo consumer scoring against source raw keys), uses the
    parameterized ``H_IDX: tl.constexpr`` Triton prefill scorer on CUDA, and
    shards query scoring + Top-K across ``sp_group`` (CP-Indexer) when enabled.
    """
    bsz, seq_len, n_heads, head_dim = q_unrot.shape
    ratio = indexer.compress_ratio
    device = q_unrot.device

    cp_s = max(1, int(sp_size))
    cp_r = int(sp_rank)
    can_cp = (
        cp_s > 1
        and sp_group is not None
        and 0 <= cp_r < cp_s
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and bsz > 0
        and seq_len > 0
        and ((bsz >= cp_s and (seq_len <= 32 or (bsz % cp_s) == 0)) or seq_len >= cp_s)
    )
    if can_cp and _cp_use_batch_shard(bsz, seq_len, cp_s):
        max_sel = indexer.max_selected
        b_chunk = (bsz + cp_s - 1) // cp_s
        b_start = cp_r * b_chunk
        b_end = min(bsz, b_start + b_chunk)
        b_loc = max(0, b_end - b_start)
        if b_loc > 0:
            pos_sub = (
                positions[b_start:b_end]
                if (positions is not None and positions.ndim >= 2 and int(positions.shape[0]) >= b_end)
                else positions
            )
            sel_loc = qsa_select_causal_from_projected(
                indexer,
                q_unrot[b_start:b_end],
                raw_keys[b_start:b_end],
                cos_all[b_start:b_end] if cos_all is not None else None,
                sin_all[b_start:b_end] if sin_all is not None else None,
                past_raw_keys=past_raw_keys[b_start:b_end] if past_raw_keys is not None else None,
                k_norm_gamma=k_norm_gamma,
                q_norm_gamma=q_norm_gamma,
                positions=pos_sub,
                rotary_dim=rotary_dim,
                rope_theta=rope_theta,
                sp_size=1,
            )
            if b_loc == b_chunk:
                loc_flat = sel_loc.contiguous().reshape(b_chunk * seq_len, max_sel)
            else:
                loc_pad = torch.empty((b_chunk, seq_len, max_sel), dtype=torch.int32, device=device)
                loc_pad[:b_loc].copy_(sel_loc)
                loc_flat = loc_pad.reshape(b_chunk * seq_len, max_sel)
        else:
            loc_flat = torch.empty((b_chunk * seq_len, max_sel), dtype=torch.int32, device=device)
        gathered = _cp_all_gather_dim0(
            loc_flat,
            cp_size=cp_s,
            cp_group=sp_group,
            fast_all_reduce=fast_all_reduce,
        )
        return gathered.view(cp_s * b_chunk, seq_len, max_sel)[:bsz]

    all_raw = (
        torch.cat([past_raw_keys, raw_keys], dim=1)
        if past_raw_keys is not None
        else raw_keys
    )
    past_len = int(all_raw.shape[1]) - seq_len
    use_cp_s = can_cp and not _cp_use_batch_shard(bsz, seq_len, cp_s)
    use_fast_rope = positions is not None
    if use_cp_s:
        s_pad = ((seq_len + cp_s - 1) // cp_s) * cp_s
        s_chunk = s_pad // cp_s
        q_slice = q_unrot[:, cp_r:seq_len:cp_s]
        q_rows = int(q_slice.shape[1])
        if use_fast_rope:
            pos_q_all = positions.unsqueeze(0).expand(bsz, seq_len) if positions.ndim == 1 else positions
            pos_q = pos_q_all[:, cp_r:seq_len:cp_s]
            q_rot = _qsa_norm_and_rope_fast(
                q_slice,
                pos_q,
                rotary_dim,
                rope_theta,
                q_norm_gamma=q_norm_gamma,
                norm_eps=indexer.norm_eps,
            )
        else:
            assert cos_all is not None and sin_all is not None
            q_in = (
                exact_head_rms_norm(q_slice.contiguous(), q_norm_gamma, indexer.norm_eps)
                if q_norm_gamma is not None
                else q_slice.contiguous()
            )
            q_rot = apply_partial_rope(
                q_in,
                cos_all[:, past_len + cp_r : past_len + seq_len : cp_s].unsqueeze(2),
                sin_all[:, past_len + cp_r : past_len + seq_len : cp_s].unsqueeze(2),
            )
        q_vis_pos = past_len + cp_r + torch.arange(q_rows, dtype=torch.int32, device=device) * cp_s
    else:
        q_rows = seq_len
        if use_fast_rope:
            pos_q = positions.unsqueeze(0).expand(bsz, seq_len) if positions.ndim == 1 else positions
            q_rot = _qsa_norm_and_rope_fast(
                q_unrot,
                pos_q,
                rotary_dim,
                rope_theta,
                q_norm_gamma=q_norm_gamma,
                norm_eps=indexer.norm_eps,
            )
        else:
            assert cos_all is not None and sin_all is not None
            q_in = (
                exact_head_rms_norm(q_unrot, q_norm_gamma, indexer.norm_eps)
                if q_norm_gamma is not None
                else q_unrot
            )
            q_rot = apply_partial_rope(
                q_in, cos_all[:, past_len:].unsqueeze(2), sin_all[:, past_len:].unsqueeze(2)
            )
        q_vis_pos = past_len + torch.arange(seq_len, dtype=torch.int32, device=device)

    num_blocks = int(all_raw.shape[1]) // ratio
    visible_1d = (q_vis_pos + 1).unsqueeze(0).expand(bsz, q_rows).reshape(-1)
    if num_blocks == 0:
        empty_scores = torch.empty((bsz * q_rows, 0), dtype=torch.float32, device=device)
        sel_local = _select_top_tokens(
            empty_scores,
            visible_1d,
            compress_ratio=ratio,
            token_budget=indexer.token_budget,
            scores_are_masked=False,
        ).view(bsz, q_rows, indexer.max_selected)
    else:
        gamma = k_norm_gamma if k_norm_gamma is not None else indexer.k_norm_gamma
        groups = all_raw[:, : num_blocks * ratio].unflatten(1, (num_blocks, ratio))
        pooled = groups.float().mean(dim=2).to(all_raw.dtype)
        if use_fast_rope:
            starts_pos = (
                (torch.arange(num_blocks, dtype=torch.int32, device=device) * ratio)
                .unsqueeze(0)
                .expand(bsz, num_blocks)
            )
            block_keys = _qsa_norm_and_rope_fast(
                pooled.unsqueeze(2),
                starts_pos,
                rotary_dim,
                rope_theta,
                q_norm_gamma=gamma,
                norm_eps=indexer.norm_eps,
            ).squeeze(2)
        else:
            assert cos_all is not None and sin_all is not None
            pooled = exact_head_rms_norm(pooled, gamma, indexer.norm_eps)
            starts = torch.arange(num_blocks, device=device) * ratio
            block_keys = apply_partial_rope(
                pooled, cos_all.index_select(1, starts), sin_all.index_select(1, starts)
            )

        bytes_per_query = max(1, bsz * num_blocks * (n_heads + 1) * 4)
        chunk_size = max(1, _MAX_SCORE_WORKSPACE_BYTES // bytes_per_query)
        if q_rows <= chunk_size:
            scores = qsa_prefill_score(q_rot, block_keys)
            sel_local = _select_top_tokens(
                scores.reshape(bsz * q_rows, num_blocks),
                visible_1d,
                compress_ratio=ratio,
                token_budget=indexer.token_budget,
                scores_are_masked=False,
            ).view(bsz, q_rows, indexer.max_selected)
        else:
            sel_local = torch.empty(
                (bsz, q_rows, indexer.max_selected), dtype=torch.int32, device=device
            )
            for start in range(0, q_rows, chunk_size):
                end = min(start + chunk_size, q_rows)
                sub_scores = qsa_prefill_score(q_rot[:, start:end], block_keys)
                sub_vis = (
                    (q_vis_pos[start:end] + 1)
                    .unsqueeze(0)
                    .expand(bsz, end - start)
                    .reshape(-1)
                )
                sub_sel = _select_top_tokens(
                    sub_scores.reshape(bsz * (end - start), num_blocks),
                    sub_vis,
                    compress_ratio=ratio,
                    token_budget=indexer.token_budget,
                    scores_are_masked=False,
                ).view(bsz, end - start, indexer.max_selected)
                sel_local[:, start:end].copy_(sub_sel)

    if not use_cp_s:
        return sel_local
    max_sel = indexer.max_selected
    if q_rows == s_chunk:
        loc_buf = sel_local.contiguous()
    else:
        loc_buf = torch.empty((bsz, s_chunk, max_sel), dtype=torch.int32, device=device)
        loc_buf[:, :q_rows].copy_(sel_local)
    gathered = _cp_all_gather_dim0(
        loc_buf.reshape(bsz * s_chunk, max_sel),
        cp_size=cp_s,
        cp_group=sp_group,
        fast_all_reduce=fast_all_reduce,
    ).view(cp_s, bsz, s_chunk, max_sel)
    return gathered.permute(1, 2, 0, 3).reshape(bsz, s_pad, max_sel)[:, :seq_len, :].contiguous()


def _qsa_norm_and_rope_fast(
    q_bf16: torch.Tensor,
    pos_2d: torch.Tensor,
    rotary_dim: int,
    rope_theta: float,
    *,
    q_norm_gamma: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    bsz, seq_len, n_heads, head_dim = q_bf16.shape
    if (
        _HAS_TRITON
        and q_bf16.is_cuda
        and pos_2d.is_cuda
        and head_dim == 128
        and rotary_dim == 64
        and q_bf16.stride(-1) == 1
        and q_bf16.dtype == torch.bfloat16
        and bsz > 0
        and seq_len > 0
    ):
        inv_freq = _get_qsa_inv_freq(rotary_dim, rope_theta, q_bf16.device)
        out = torch.empty((bsz, seq_len, n_heads, head_dim), dtype=q_bf16.dtype, device=q_bf16.device)
        has_norm = q_norm_gamma is not None
        g_ptr = q_norm_gamma if has_norm else inv_freq
        if has_norm and not q_norm_gamma.is_contiguous():
            g_ptr = q_norm_gamma.contiguous()
        sqb = int(q_bf16.stride(0))
        sqs = int(q_bf16.stride(1))
        sqh = int(q_bf16.stride(2))
        spb = int(pos_2d.stride(0))
        sps = int(pos_2d.stride(1))
        sob = int(out.stride(0))
        sos = int(out.stride(1))
        soh = int(out.stride(2))
        ckey = (
            q_bf16.device.index,
            1,
            q_bf16.dtype,
            g_ptr.dtype,
            pos_2d.dtype,
            inv_freq.dtype,
            tuple(_triton_int_spec(v) for v in (sqs, sqh, sps, sos, soh)),
            n_heads,
            float(norm_eps),
            has_norm,
        )
        triton_fast_launch(
            _QSA_TRITON_CACHE,
            ckey,
            _qsa_head_rmsnorm_partial_rope_d128_r64_kernel,
            bsz,
            seq_len,
            n_heads,
            q_bf16.data_ptr() | g_ptr.data_ptr() | pos_2d.data_ptr() | inv_freq.data_ptr() | out.data_ptr(),
            (q_bf16, g_ptr, pos_2d, inv_freq, out, sqb, sqs, sqh, spb, sps, sob, sos, soh, seq_len),
            {"H": n_heads, "EPS": float(norm_eps), "HAS_NORM": has_norm},
            num_warps=4,
        )
        return out
    q_normed = exact_head_rms_norm(q_bf16, q_norm_gamma, norm_eps) if q_norm_gamma is not None else q_bf16
    cos, sin = build_qsa_neox_cos_sin(pos_2d, rotary_dim, rope_theta, q_bf16.dtype)
    return apply_partial_rope(q_normed, cos.unsqueeze(2), sin.unsqueeze(2))


def score_and_select_qsa_paged(
    q_unrot_bf16: torch.Tensor,
    positions: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_table: torch.Tensor,
    visible_lens_dev: torch.Tensor,
    *,
    compress_ratio: int,
    token_budget: int,
    rotary_dim: int,
    rope_theta: float,
    max_ctx_len: Optional[int] = None,
    is_cuda_graph: bool = False,
    q_norm_gamma: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    sp_size: int = 1,
    sp_rank: int = 0,
    sp_group: Any = None,
    fast_all_reduce: Any = None,
) -> torch.Tensor:
    """Zero-sync paged QSA scoring + token selection for prefill reuse & decode.

    Args:
        q_unrot_bf16: Unrotated indexer query ``[B, S, H_idx, D_idx]`` (raw if ``q_norm_gamma`` passed).
        positions: Token positions ``[B, S]`` or ``[B]``.
        kv_pool: ``indexer_kv`` view ``[pool_blocks, entries_per_block, D_idx]``.
        kv_table: ``indexer_kv`` block table ``[B, max_blocks]`` int32.
        visible_lens_dev: Visible token lengths ``[B, S]`` or ``[B]`` int32.
    Returns:
        Selected token indices ``[B, S, token_budget + compress_ratio - 1]`` int32.
    """
    bsz, seq_len, n_heads, head_dim = q_unrot_bf16.shape
    if positions.ndim == 1:
        if bsz == 1 and seq_len > 1 and int(positions.numel()) == seq_len:
            pos_2d = positions.unsqueeze(0)
        else:
            pos_2d = positions.unsqueeze(1)
    else:
        pos_2d = positions
    vis_2d = (
        visible_lens_dev.unsqueeze(1)
        if visible_lens_dev.ndim == 1
        else visible_lens_dev
    )
    if vis_2d.dtype != torch.int32:
        vis_2d = vis_2d.to(dtype=torch.int32)
    if max_ctx_len is None:
        kv_entries_per_block = int(kv_pool.shape[1])
        max_ctx_len = int(kv_table.shape[1]) * kv_entries_per_block

    cp_s = max(1, int(sp_size))
    cp_r = int(sp_rank)
    can_cp = (
        cp_s > 1
        and sp_group is not None
        and 0 <= cp_r < cp_s
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and bsz > 0
        and seq_len > 0
        and ((bsz >= cp_s and (seq_len <= 32 or (bsz % cp_s) == 0)) or seq_len >= cp_s)
    )
    if can_cp:
        max_sel = token_budget + compress_ratio - 1
        dev = q_unrot_bf16.device
        if _cp_use_batch_shard(bsz, seq_len, cp_s):
            b_chunk = (bsz + cp_s - 1) // cp_s
            b_start = cp_r * b_chunk
            b_end = min(bsz, b_start + b_chunk)
            b_loc = max(0, b_end - b_start)
            if b_loc > 0:
                pos_sub = pos_2d[b_start:b_end] if int(pos_2d.shape[0]) >= b_end else pos_2d
                vis_sub = vis_2d[b_start:b_end] if int(vis_2d.shape[0]) >= b_end else vis_2d
                sel_loc = score_and_select_qsa_paged(
                    q_unrot_bf16[b_start:b_end],
                    pos_sub,
                    kv_pool,
                    kv_table[b_start:b_end],
                    vis_sub,
                    compress_ratio=compress_ratio,
                    token_budget=token_budget,
                    rotary_dim=rotary_dim,
                    rope_theta=rope_theta,
                    max_ctx_len=max_ctx_len,
                    is_cuda_graph=is_cuda_graph,
                    q_norm_gamma=q_norm_gamma,
                    norm_eps=norm_eps,
                    sp_size=1,
                )
                if b_loc == b_chunk:
                    loc_flat = sel_loc.contiguous().reshape(b_chunk * seq_len, max_sel)
                else:
                    loc_pad = torch.empty((b_chunk, seq_len, max_sel), dtype=torch.int32, device=dev)
                    loc_pad[:b_loc].copy_(sel_loc)
                    loc_flat = loc_pad.reshape(b_chunk * seq_len, max_sel)
            else:
                loc_flat = torch.empty((b_chunk * seq_len, max_sel), dtype=torch.int32, device=dev)
            gathered = _cp_all_gather_dim0(
                loc_flat,
                cp_size=cp_s,
                cp_group=sp_group,
                fast_all_reduce=fast_all_reduce,
            )
            return gathered.view(cp_s * b_chunk, seq_len, max_sel)[:bsz]

        s_pad = ((seq_len + cp_s - 1) // cp_s) * cp_s
        s_chunk = s_pad // cp_s
        q_loc = q_unrot_bf16[:, cp_r:seq_len:cp_s]
        if q_loc.stride(-1) != 1:
            q_loc = q_loc.contiguous()
        q_rows = int(q_loc.shape[1])
        pos_loc = pos_2d[:, cp_r:seq_len:cp_s] if int(pos_2d.shape[1]) >= seq_len else pos_2d
        vis_loc = vis_2d[:, cp_r:seq_len:cp_s] if int(vis_2d.shape[1]) >= seq_len else vis_2d
        sel_loc = score_and_select_qsa_paged(
            q_loc,
            pos_loc,
            kv_pool,
            kv_table,
            vis_loc,
            compress_ratio=compress_ratio,
            token_budget=token_budget,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            max_ctx_len=max_ctx_len,
            is_cuda_graph=is_cuda_graph,
            q_norm_gamma=q_norm_gamma,
            norm_eps=norm_eps,
            sp_size=1,
        )
        if q_rows == s_chunk:
            loc_buf = sel_loc.contiguous()
        else:
            loc_buf = torch.empty((bsz, s_chunk, max_sel), dtype=torch.int32, device=dev)
            loc_buf[:, :q_rows].copy_(sel_loc)
        gathered = _cp_all_gather_dim0(
            loc_buf.reshape(bsz * s_chunk, max_sel),
            cp_size=cp_s,
            cp_group=sp_group,
            fast_all_reduce=fast_all_reduce,
        ).view(cp_s, bsz, s_chunk, max_sel)
        return gathered.permute(1, 2, 0, 3).reshape(bsz, s_pad, max_sel)[:, :seq_len, :].contiguous()

    q_rot = _qsa_norm_and_rope_fast(
        q_unrot_bf16,
        pos_2d,
        rotary_dim,
        rope_theta,
        q_norm_gamma=q_norm_gamma,
        norm_eps=norm_eps,
    )

    ctx_lens = torch.div(vis_2d, compress_ratio, rounding_mode="floor")

    block_logits = qsa_paged_score(
        q_rot,
        kv_pool,
        kv_table,
        ctx_lens,
        max_ctx_len=max_ctx_len,
    )
    selected = _select_top_tokens(
        block_logits,
        vis_2d.reshape(-1),
        compress_ratio=compress_ratio,
        token_budget=token_budget,
        scores_are_masked=True,
    )
    return selected.view(bsz, seq_len, token_budget + compress_ratio - 1)


def write_qsa_prefill_cache(
    raw_keys: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    pos_1d: torch.Tensor,
    k_norm_gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_block_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_block_table: torch.Tensor,
    *,
    ratio: int,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    rotary_dim: int,
    rope_theta: float,
    norm_eps: float,
    is_s_padded: bool = False,
    tok_req_i32: Optional[torch.Tensor] = None,
) -> None:
    """Write raw indexer keys (state ring) and pooled keys (kv pool) for prefill/suffix."""
    token_count = int(raw_keys.shape[0])
    if token_count == 0:
        return
    head_dim = int(raw_keys.shape[1])
    bsz = int(cu_seqlens.numel()) - 1
    if (
        _HAS_TRITON
        and raw_keys.is_cuda
        and ratio == 4
        and head_dim == 128
        and rotary_dim == 64
        and kv_tokens_per_block >= 4
        and kv_tokens_per_block % 4 == 0
        and state_tokens_per_block >= 4
        and raw_keys.dtype == torch.bfloat16
        and kv_pool.dtype == torch.bfloat16
        and raw_keys.stride(1) == 1
        and k_norm_gamma.is_contiguous()
        and bsz > 0
    ):
        inv_freq = _get_qsa_inv_freq(rotary_dim, rope_theta, raw_keys.device)
        cu_c = cu_seqlens if cu_seqlens.is_contiguous() else cu_seqlens.contiguous()
        sp_c = start_positions if start_positions.is_contiguous() else start_positions.contiguous()
        pos_c = pos_1d if pos_1d.is_contiguous() else pos_1d.contiguous()
        has_tr = tok_req_i32 is not None and int(tok_req_i32.numel()) == token_count
        tr_arg = (tok_req_i32 if tok_req_i32.is_contiguous() else tok_req_i32.contiguous()) if has_tr else cu_c
        ring_sz = int(state_pool.shape[1])
        kv_entries = kv_tokens_per_block // ratio
        srn = int(raw_keys.stride(0))
        skvb = int(kv_block_table.stride(0))
        skvp = int(kv_block_table.stride(1))
        sstb = int(state_block_table.stride(0))
        sstp = int(state_block_table.stride(1))
        kv_tw = int(kv_block_table.shape[1])
        st_tw = int(state_block_table.shape[1])
        kv_pb = int(kv_pool.shape[0])
        st_pb = int(state_pool.shape[0])

        # Kernel 1: read raw_keys + pre-prefill state_pool -> write kv_pool (race-free)
        rt1 = (
            raw_keys, cu_c, sp_c, pos_c, tr_arg, inv_freq, k_norm_gamma,
            kv_pool, kv_block_table, state_pool, state_block_table,
            token_count, bsz, srn, skvb, skvp, sstb, sstp,
            kv_tw, st_tw, kv_pb, st_pb,
            kv_tokens_per_block, state_tokens_per_block, kv_entries,
        )
        k1 = (
            raw_keys.device.index, 10, raw_keys.dtype, cu_c.dtype, sp_c.dtype, pos_c.dtype, tr_arg.dtype,
            inv_freq.dtype, k_norm_gamma.dtype, kv_pool.dtype, kv_block_table.dtype, state_pool.dtype, state_block_table.dtype,
            _triton_int_spec(srn), _triton_int_spec(skvb), _triton_int_spec(skvp),
            _triton_int_spec(sstb), _triton_int_spec(sstp),
            has_tr, ring_sz, float(norm_eps),
        )
        ptr1 = (
            raw_keys.data_ptr() | cu_c.data_ptr() | sp_c.data_ptr() | pos_c.data_ptr()
            | tr_arg.data_ptr() | inv_freq.data_ptr() | k_norm_gamma.data_ptr()
            | kv_pool.data_ptr() | kv_block_table.data_ptr() | state_pool.data_ptr() | state_block_table.data_ptr()
        )
        triton_fast_launch(
            _QSA_TRITON_CACHE, k1, _write_prefill_qsa_kv_k4_kernel,
            token_count, 1, 1, ptr1, rt1,
            {"HAS_TOK_REQ": has_tr, "RING_SIZE": ring_sz, "D": 128, "ROTARY": 64, "EPS": float(norm_eps)},
            num_warps=4,
        )

        # Kernel 2: write final ring entries per state block (race-free)
        rt2 = (
            raw_keys, cu_c, sp_c, pos_c, tr_arg, state_pool, state_block_table,
            token_count, bsz, srn, sstb, sstp, st_tw, st_pb, state_tokens_per_block,
        )
        k2 = (
            raw_keys.device.index, 11, raw_keys.dtype, cu_c.dtype, sp_c.dtype, pos_c.dtype, tr_arg.dtype,
            state_pool.dtype, state_block_table.dtype,
            _triton_int_spec(srn), _triton_int_spec(sstb), _triton_int_spec(sstp),
            has_tr, ring_sz,
        )
        ptr2 = (
            raw_keys.data_ptr() | cu_c.data_ptr() | sp_c.data_ptr() | pos_c.data_ptr()
            | tr_arg.data_ptr() | state_pool.data_ptr() | state_block_table.data_ptr()
        )
        triton_fast_launch(
            _QSA_TRITON_CACHE, k2, _write_prefill_qsa_state_k4_kernel,
            token_count, 1, 1, ptr2, rt2,
            {"HAS_TOK_REQ": has_tr, "RING_SIZE": ring_sz, "D": 128},
            num_warps=4,
        )
        return

    pos_i64 = pos_1d.to(torch.int64)
    block_starts = pos_i64 - torch.remainder(pos_i64, ratio)
    block_cos, block_sin = build_qsa_neox_cos_sin(
        block_starts, rotary_dim, rope_theta, raw_keys.dtype
    )
    write_indexer_cache(
        raw_keys,
        cu_seqlens,
        start_positions,
        block_cos,
        block_sin,
        k_norm_gamma,
        kv_pool,
        kv_block_table,
        state_pool,
        state_block_table,
        ratio=ratio,
        kv_tokens_per_block=kv_tokens_per_block,
        state_tokens_per_block=state_tokens_per_block,
        norm_eps=norm_eps,
        rope_is_token_aligned=True,
        is_s_padded=is_s_padded,
    )


if _HAS_TRITON:

    @triton.jit(do_not_specialize=["KV_TABLE_WIDTH", "KV_POOL_BLOCKS", "KV_TOKENS_PER_BLOCK", "KV_ENTRIES_PER_BLOCK"])
    def _write_verify_qsa_cache_k4_kernel(
        raw_ptr,
        seq_before_ptr,
        valid_ptr,
        inv_freq_ptr,
        gamma_ptr,
        init_state_ptr,
        stage_state_ptr,
        kv_pool_ptr,
        kv_table_ptr,
        stride_rb: tl.int64,
        stride_rv: tl.int64,
        stride_ib: tl.int64,
        stride_is: tl.int64,
        stride_stb: tl.int64,
        stride_stv: tl.int64,
        stride_sts: tl.int64,
        stride_kv_b: tl.int64,
        stride_kv_p: tl.int64,
        KV_TABLE_WIDTH,
        KV_POOL_BLOCKS,
        KV_TOKENS_PER_BLOCK,
        KV_ENTRIES_PER_BLOCK,
        V_STEPS: tl.constexpr,
        D: tl.constexpr,
        ROTARY: tl.constexpr,
        EPS: tl.constexpr,
        HAS_VALID: tl.constexpr,
    ):
        """Batched V-step QSA indexer cache write + per-step ring state staging with inline RoPE."""
        request = tl.program_id(0).to(tl.int64)
        d = tl.arange(0, D)
        ring_offs = tl.arange(0, 8)
        active = (tl.load(valid_ptr + request) != 0) if HAS_VALID else True
        pos0 = tl.load(seq_before_ptr + request).to(tl.int64)

        ring = tl.load(
            init_state_ptr + request * stride_ib + ring_offs[:, None] * stride_is + d[None, :]
        ).to(tl.float32)
        gain = 1.0 + tl.load(gamma_ptr + d).to(tl.float32)

        partner = tl.where(
            d < ROTARY,
            tl.where(d < ROTARY // 2, d + ROTARY // 2, d - ROTARY // 2),
            d,
        )
        half_sign = tl.where(d < ROTARY // 2, -1.0, 1.0)
        freq_idx = tl.where(d < ROTARY // 2, d, d - ROTARY // 2)
        inv_f = tl.load(inv_freq_ptr + freq_idx, mask=d < ROTARY, other=0.0)

        for v in tl.static_range(V_STEPS):
            pos = pos0 + v
            is_step_active = active & (pos >= 0)
            raw = tl.load(
                raw_ptr + request * stride_rb + v * stride_rv + d,
                mask=is_step_active,
                other=0.0,
            ).to(tl.float32)
            slot = pos % 8
            ring = tl.where(is_step_active & (ring_offs[:, None] == slot), raw[None, :], ring)

            tl.store(
                stage_state_ptr
                + request * stride_stb
                + v * stride_stv
                + ring_offs[:, None] * stride_sts
                + d[None, :],
                ring,
            )

            if is_step_active and (pos + 1) % 4 == 0:
                use_hi = (pos % 8) >= 4
                grp_mask = (ring_offs >= 4) if use_hi else (ring_offs < 4)
                grp_sum = tl.sum(tl.where(grp_mask[:, None], ring, 0.0), axis=0)
                pooled = (grp_sum * 0.25).to(tl.bfloat16).to(tl.float32)
                rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / D + EPS)
                normalized = (pooled * rms * gain).to(tl.bfloat16)

                rotated_half = tl.gather(normalized, partner, axis=0)
                ang = (pos - 3).to(tl.float32) * inv_f
                cosine = tl.where(d < ROTARY, libdevice.cos(ang).to(tl.bfloat16), tl.cast(1.0, tl.bfloat16))
                sine = tl.where(d < ROTARY, libdevice.sin(ang).to(tl.bfloat16), tl.cast(0.0, tl.bfloat16))
                left = (normalized.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
                right = (rotated_half.to(tl.float32) * sine.to(tl.float32) * half_sign).to(
                    tl.bfloat16
                )
                value = tl.where(
                    d < ROTARY,
                    (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
                    normalized,
                )

                kv_column = pos // KV_TOKENS_PER_BLOCK
                ok_kv = kv_column < KV_TABLE_WIDTH
                kv_id = tl.load(
                    kv_table_ptr + request * stride_kv_b + tl.minimum(kv_column, KV_TABLE_WIDTH - 1) * stride_kv_p,
                    mask=ok_kv,
                    other=0,
                ).to(tl.int64)
                kv_valid = ok_kv & (kv_id > 0) & (kv_id < KV_POOL_BLOCKS)
                entry = (pos // 4) % KV_ENTRIES_PER_BLOCK
                tl.store(
                    kv_pool_ptr + (kv_id * KV_ENTRIES_PER_BLOCK + entry) * D + d,
                    value,
                    mask=kv_valid,
                )


def write_qsa_verify_cache_gpu_(
    raw_keys: torch.Tensor,
    seq_lens_before_dev: torch.Tensor,
    k_norm_gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_table: torch.Tensor,
    init_state: torch.Tensor,
    stage_state: torch.Tensor,
    *,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    ratio: int,
    rotary_dim: int,
    rope_theta: float,
    norm_eps: float,
    valid: Optional[torch.Tensor] = None,
) -> None:
    """Zero-sync batched V-step QSA indexer cache write + rollback ring staging."""
    bsz, verify_w, head_dim = int(raw_keys.shape[0]), int(raw_keys.shape[1]), int(raw_keys.shape[2])
    if bsz == 0 or verify_w == 0:
        return
    if (
        _HAS_TRITON
        and raw_keys.is_cuda
        and ratio == 4
        and head_dim == 128
        and rotary_dim == 64
        and verify_w <= 16
        and kv_tokens_per_block >= 4
        and kv_tokens_per_block % 4 == 0
        and state_tokens_per_block >= 4
        and raw_keys.stride(-1) == 1
    ):
        inv_freq = _get_qsa_inv_freq(rotary_dim, rope_theta, raw_keys.device)
        has_valid = valid is not None
        valid_arg = valid if has_valid else seq_lens_before_dev
        srb = int(raw_keys.stride(0))
        srv = int(raw_keys.stride(1))
        sib = int(init_state.stride(0))
        sis = int(init_state.stride(1))
        sstb = int(stage_state.stride(0))
        sstv = int(stage_state.stride(1))
        ssts = int(stage_state.stride(2))
        skvb = int(kv_table.stride(0))
        skvp = int(kv_table.stride(1))
        rt_args = (
            raw_keys,
            seq_lens_before_dev,
            valid_arg,
            inv_freq,
            k_norm_gamma,
            init_state,
            stage_state,
            kv_pool,
            kv_table,
            srb,
            srv,
            sib,
            sis,
            sstb,
            sstv,
            ssts,
            skvb,
            skvp,
            int(kv_table.shape[1]),
            int(kv_pool.shape[0]),
            kv_tokens_per_block,
            kv_tokens_per_block // ratio,
        )
        ckey = (
            raw_keys.device.index,
            2,
            raw_keys.dtype,
            seq_lens_before_dev.dtype,
            valid_arg.dtype,
            k_norm_gamma.dtype,
            init_state.dtype,
            stage_state.dtype,
            kv_pool.dtype,
            kv_table.dtype,
            tuple(_triton_int_spec(v) for v in (srb, srv, sib, sis, sstb, sstv, ssts, skvb, skvp)),
            verify_w,
            head_dim,
            rotary_dim,
            float(norm_eps),
            has_valid,
        )
        ptr_or = (
            raw_keys.data_ptr()
            | seq_lens_before_dev.data_ptr()
            | valid_arg.data_ptr()
            | inv_freq.data_ptr()
            | k_norm_gamma.data_ptr()
            | init_state.data_ptr()
            | stage_state.data_ptr()
            | kv_pool.data_ptr()
            | kv_table.data_ptr()
        )
        triton_fast_launch(
            _QSA_TRITON_CACHE,
            ckey,
            _write_verify_qsa_cache_k4_kernel,
            bsz,
            1,
            1,
            ptr_or,
            rt_args,
            {"V_STEPS": verify_w, "D": head_dim, "ROTARY": rotary_dim, "EPS": float(norm_eps), "HAS_VALID": has_valid},
            num_warps=4,
        )
        return

    # Fallback path for non-4 ratio / CPU
    scratch_pool = torch.zeros(
        (bsz + 1, *init_state.shape[1:]), dtype=torch.float32, device=raw_keys.device
    )
    scratch_pool[1 : bsz + 1].copy_(init_state)
    row_ids = (
        torch.where(valid, torch.arange(1, bsz + 1, dtype=torch.int32, device=raw_keys.device), 0)
        if valid is not None
        else torch.arange(1, bsz + 1, dtype=torch.int32, device=raw_keys.device)
    )
    scratch_bt = row_ids.unsqueeze(1).expand(bsz, int(kv_table.shape[1])).contiguous()
    for step in range(verify_w):
        write_qsa_decode_cache_gpu_(
            raw_keys[:, step],
            seq_lens_before_dev + step,
            k_norm_gamma,
            kv_pool,
            kv_table,
            scratch_pool,
            scratch_bt,
            kv_tokens_per_block=kv_tokens_per_block,
            state_tokens_per_block=state_tokens_per_block,
            ratio=ratio,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            valid=valid,
        )
        stage_state[:, step].copy_(scratch_pool[1 : bsz + 1])


def write_qsa_decode_cache_gpu_(
    raw_keys: torch.Tensor,
    seq_lens_before_dev: torch.Tensor,
    k_norm_gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_table: torch.Tensor,
    *,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    ratio: int,
    rotary_dim: int,
    rope_theta: float,
    norm_eps: float,
    valid: Optional[torch.Tensor] = None,
) -> None:
    """Zero-sync single-token decode write supporting any power-of-2 page size."""
    bsz, head_dim = int(raw_keys.shape[0]), int(raw_keys.shape[1])
    if bsz == 0:
        return
    if (
        _HAS_TRITON
        and raw_keys.is_cuda
        and ratio == 4
        and head_dim == 128
        and rotary_dim == 64
        and kv_tokens_per_block >= 4
        and kv_tokens_per_block % 4 == 0
        and state_tokens_per_block >= 4
        and raw_keys.stride(-1) == 1
        and k_norm_gamma.is_contiguous()
    ):
        inv_freq = _get_qsa_inv_freq(rotary_dim, rope_theta, raw_keys.device)
        has_valid = valid is not None
        valid_arg = valid if has_valid else seq_lens_before_dev
        ring_sz = int(state_pool.shape[1])
        srb = int(raw_keys.stride(0))
        skvb = int(kv_table.stride(0))
        skvp = int(kv_table.stride(1))
        sstb = int(state_table.stride(0))
        sstp = int(state_table.stride(1))
        rt_args = (
            raw_keys,
            seq_lens_before_dev,
            valid_arg,
            inv_freq,
            k_norm_gamma,
            kv_pool,
            kv_table,
            state_pool,
            state_table,
            srb,
            skvb,
            skvp,
            sstb,
            sstp,
            int(kv_table.shape[1]),
            int(state_table.shape[1]),
            int(kv_pool.shape[0]),
            int(state_pool.shape[0]),
            kv_tokens_per_block,
            state_tokens_per_block,
            kv_tokens_per_block // ratio,
        )
        ckey = (
            raw_keys.device.index,
            3,
            raw_keys.dtype,
            seq_lens_before_dev.dtype,
            valid_arg.dtype,
            k_norm_gamma.dtype,
            kv_pool.dtype,
            kv_table.dtype,
            state_pool.dtype,
            state_table.dtype,
            _triton_int_spec(srb),
            _triton_int_spec(skvb),
            _triton_int_spec(skvp),
            _triton_int_spec(sstb),
            _triton_int_spec(sstp),
            ring_sz,
            head_dim,
            rotary_dim,
            float(norm_eps),
            has_valid,
        )
        ptr_or = (
            raw_keys.data_ptr()
            | seq_lens_before_dev.data_ptr()
            | valid_arg.data_ptr()
            | inv_freq.data_ptr()
            | k_norm_gamma.data_ptr()
            | kv_pool.data_ptr()
            | kv_table.data_ptr()
            | state_pool.data_ptr()
            | state_table.data_ptr()
        )
        triton_fast_launch(
            _QSA_TRITON_CACHE,
            ckey,
            _write_decode_qsa_cache_k4_kernel,
            bsz,
            1,
            1,
            ptr_or,
            rt_args,
            {"RING_SIZE": ring_sz, "D": head_dim, "ROTARY": rotary_dim, "EPS": float(norm_eps), "HAS_VALID": has_valid},
            num_warps=4,
        )
        return

    pos = seq_lens_before_dev.to(dtype=torch.int64)
    block_starts = pos - torch.remainder(pos, ratio)
    block_cos, block_sin = build_qsa_neox_cos_sin(
        block_starts, rotary_dim, rope_theta, raw_keys.dtype
    )

    cu_seqlens = torch.arange(bsz + 1, dtype=torch.int32, device=raw_keys.device)
    write_indexer_cache(
        raw_keys.contiguous(),
        cu_seqlens,
        pos,
        block_cos,
        block_sin,
        k_norm_gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
        ratio=ratio,
        kv_tokens_per_block=kv_tokens_per_block,
        state_tokens_per_block=state_tokens_per_block,
        norm_eps=norm_eps,
        rope_is_token_aligned=True,
        is_s_padded=valid is not None,
    )


if _HAS_TRITON:

    @triton.jit(do_not_specialize=["T", "K", "WORDS"])
    def _qsa_pack_bitmap_words_kernel(
        ids_ptr,
        packed_ptr,
        ends_ptr,
        dup_ptr,
        T,
        K,
        WORDS,
        BK: tl.constexpr,
        CHECK_DUP: tl.constexpr = True,
    ):
        """Pack per-query selected indices into 1-bit uint32 words and detect duplicates."""
        row = tl.program_id(0).to(tl.int64)
        row_end = tl.full((), -1, tl.int32)
        has_bad = tl.zeros((), dtype=tl.int32)
        base_ptr = ids_ptr + row * K
        pack_base = packed_ptr + row * WORDS
        for start in range(0, K, BK):
            slots = start + tl.arange(0, BK)
            in_k = slots < K
            idx = tl.load(base_ptr + slots, mask=in_k, other=-1)
            valid = in_k & (idx >= 0) & (idx < T)
            safe_idx = tl.where(valid, idx, 0)
            bit = tl.full((BK,), 1, tl.uint32) << (safe_idx & 31)
            if CHECK_DUP:
                old = tl.atomic_or(
                    pack_base + (safe_idx >> 5), bit, mask=valid, sem="relaxed"
                ).to(tl.uint32)
                repeated = valid & ((old & bit) != 0)
                invalid = in_k & (idx >= T)
                if tl.max((repeated | invalid).to(tl.int32), 0) != 0:
                    has_bad = 1
            else:
                tl.atomic_or(
                    pack_base + (safe_idx >> 5), bit, mask=valid, sem="relaxed"
                )
            row_end = tl.maximum(row_end, tl.max(tl.where(valid, idx, -1), 0))
        if CHECK_DUP:
            if has_bad != 0:
                tl.store(dup_ptr + row, 1)
        tl.store(ends_ptr + row, row_end + 1)

    @triton.jit(do_not_specialize=["S", "T", "WORDS"])
    def _qsa_bitmap_packed_gqa_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        packed_ptr,
        ends_ptr,
        bad_ptr,
        out_ptr,
        S,
        T,
        WORDS,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        G: tl.constexpr,
        QS: tl.constexpr,
        M_ROWS: tl.constexpr,
        KT: tl.constexpr,
        CHECK_EMPTY: tl.constexpr,
        CHECK_BAD: tl.constexpr = False,
    ):
        """Multi-query GQA sparse prefill reading 1-bit packed bitmap + split-BF16 P@V."""
        b = tl.program_id(0).to(tl.int64)
        hk = tl.program_id(1).to(tl.int64)
        n_groups = tl.num_programs(2).to(tl.int64)
        group = n_groups - 1 - tl.program_id(2).to(tl.int64)

        r = tl.arange(0, M_ROWS)
        pos = group * QS + r // G
        live = (r < QS * G) & (pos < S)
        if CHECK_BAD:
            row_bad = tl.load(bad_ptr + b * S + pos, mask=live, other=0) != 0
            live = live & ~row_bad
            if tl.max(live.to(tl.int32), 0) == 0:
                return
        h = hk * G + (r % G)
        d = tl.arange(0, D)

        qv = tl.load(
            q_ptr + ((b * HQ + h[:, None]) * S + pos[:, None]) * D + d[None, :],
            mask=live[:, None],
            other=0.0,
        )
        last = tl.max(tl.load(ends_ptr + b * S + pos, mask=live, other=0), 0)
        mx = tl.full((M_ROWS,), float("-inf"), tl.float32)
        z = tl.zeros((M_ROWS,), tl.float32)
        acc = tl.zeros((M_ROWS, D), tl.float32)

        t_rel = tl.arange(0, KT)
        word_idx = t_rel >> 5
        bit_mask = tl.full((KT,), 1, tl.uint32) << (t_rel & 31)
        WORDS_PER_TILE: tl.constexpr = KT // 32

        for tile in range(tl.cdiv(last, KT)):
            w_base = tile * WORDS_PER_TILE
            if KT == 32:
                words = tl.load(
                    packed_ptr + (b * S + pos) * WORDS + w_base,
                    mask=live & (w_base < WORDS),
                    other=0,
                ).to(tl.uint32)
                valid = live[:, None] & ((words[:, None] & bit_mask[None, :]) != 0)
            else:
                w_off = w_base + word_idx[None, :]
                words = tl.load(
                    packed_ptr + (b * S + pos[:, None]) * WORDS + w_off,
                    mask=live[:, None] & (w_off < WORDS),
                    other=0,
                ).to(tl.uint32)
                valid = live[:, None] & ((words & bit_mask[None, :]) != 0)
            has_any = True
            if CHECK_EMPTY:
                has_any = tl.max((words != 0).to(tl.int32)) != 0
            if has_any:
                t = tile * KT + t_rel
                t_live = t < T
                kval = tl.load(
                    k_ptr + ((b * HK + hk) * T + t[:, None]) * D + d[None, :],
                    mask=t_live[:, None],
                    other=0.0,
                )
                logits = tl.dot(qv, tl.trans(kval)) * SCALE
                logits = tl.where(valid, logits, float("-inf"))
                nxt = tl.maximum(mx, tl.max(logits, 1))
                rescale = tl.where(nxt == float("-inf"), 0.0, tl.exp(mx - nxt))
                prob = tl.where(valid, tl.exp(logits - nxt[:, None]), 0.0)
                vv = tl.load(
                    v_ptr + ((b * HK + hk) * T + t[:, None]) * D + d[None, :],
                    mask=t_live[:, None],
                    other=0.0,
                )
                high = prob.to(tl.bfloat16)
                low = (prob - high.to(tl.float32)).to(tl.bfloat16)
                acc = acc * rescale[:, None]
                acc = tl.dot(high, vv, acc)
                acc = tl.dot(low, vv, acc)
                z = z * rescale + tl.sum(prob, 1)
                mx = nxt

        res = tl.where(z[:, None] > 0, acc / z[:, None], 0.0)
        tl.store(
            out_ptr + ((b * HQ + h[:, None]) * S + pos[:, None]) * D + d[None, :],
            res.to(out_ptr.dtype.element_ty),
            mask=live[:, None],
        )

    @triton.jit(do_not_specialize=["stride_sel_b", "stride_sel_s", "S", "T", "K", "N_WORDS", "N_GRP_S"])
    def _qsa_q4_pack_quads_kernel(
        sel_ptr,
        bmp_ptr,
        dup_ptr,
        stride_sel_b: tl.int64,
        stride_sel_s: tl.int64,
        S,
        T,
        K,
        N_WORDS,
        N_GRP_S,
        BK: tl.constexpr,
        CHECK_DUP: tl.constexpr = True,
    ):
        """Pack 4 consecutive queries' selected indices into per-quad 16-bit bitmasks."""
        grp = tl.program_id(0).to(tl.int64)
        b = grp // N_GRP_S
        g_s = grp % N_GRP_S
        bmp_base = bmp_ptr + grp * N_WORDS
        has_bad = tl.zeros((), dtype=tl.int32)

        for j in tl.static_range(4):
            s_idx = g_s * 4 + j
            if s_idx < S:
                row_ptr = sel_ptr + b * stride_sel_b + s_idx * stride_sel_s
                for start in range(0, K, BK):
                    offs = start + tl.arange(0, BK)
                    in_k = offs < K
                    tok = tl.load(row_ptr + offs, mask=in_k, other=-1)
                    valid = in_k & (tok >= 0) & (tok < T)
                    safe_tok = tl.where(valid, tok, 0)
                    quad = safe_tok >> 2
                    w_idx = quad >> 1
                    bit_pos = ((quad & 1) << 4) + (j << 2) + (safe_tok & 3)
                    bit = tl.full((BK,), 1, tl.uint32) << bit_pos.to(tl.uint32)
                    if CHECK_DUP:
                        old = tl.atomic_or(bmp_base + w_idx, bit, mask=valid, sem="relaxed").to(
                            tl.uint32
                        )
                        bad = (valid & ((old & bit) != 0)) | (in_k & (tok >= T))
                        if tl.max(bad.to(tl.int32), 0) != 0:
                            has_bad = 1
                    else:
                        tl.atomic_or(bmp_base + w_idx, bit, mask=valid, sem="relaxed")
        if CHECK_DUP:
            if has_bad != 0:
                tl.store(dup_ptr + grp, 1)

    @triton.jit(do_not_specialize=["N_WORDS", "MAX_U"])
    def _qsa_q4_compact_quads_kernel(
        bmp_ptr,
        u_packed_ptr,
        u_cnt_ptr,
        bad_ptr,
        N_WORDS,
        MAX_U,
        BW: tl.constexpr,
        CHECK_OVERFLOW: tl.constexpr = False,
    ):
        """Stream-compact non-zero 4-token quads into packed ``(quad_id << 16) | qmask16``."""
        grp = tl.program_id(0).to(tl.int64)
        bmp_base = bmp_ptr + grp * N_WORDS
        u_base = u_packed_ptr + grp * MAX_U
        cnt = tl.zeros((), dtype=tl.int32)
        for start in range(0, N_WORDS, BW):
            offs = start + tl.arange(0, BW)
            in_w = offs < N_WORDS
            w = tl.load(bmp_base + offs, mask=in_w, other=0).to(tl.uint32)
            m0 = w & 0xFFFF
            m1 = (w >> 16) & 0xFFFF
            nz0 = in_w & (m0 != 0)
            nz1 = in_w & (m1 != 0)
            c_both = nz0.to(tl.int32) + nz1.to(tl.int32)
            pos_end = tl.cumsum(c_both, 0)
            pos_start = cnt + (pos_end - c_both)
            q0 = offs.to(tl.uint32) * 2
            q1 = q0 + 1
            tl.store(
                u_base + pos_start,
                ((q0 << 16) | m0).to(tl.int32),
                mask=nz0 & (pos_start < MAX_U),
            )
            tl.store(
                u_base + pos_start + nz0.to(tl.int32),
                ((q1 << 16) | m1).to(tl.int32),
                mask=nz1 & ((pos_start + nz0.to(tl.int32)) < MAX_U),
            )
            cnt += tl.sum(c_both, 0)
        tl.store(u_cnt_ptr + grp, cnt)
        if CHECK_OVERFLOW:
            if cnt > MAX_U:
                tl.store(bad_ptr + grp, 1)

    @triton.jit(
        do_not_specialize=[
            "stride_scp",
            "stride_sckv",
            "stride_sch",
            "stride_sct",
            "stride_btb",
            "S",
            "T",
            "MAX_PAGES",
            "NUM_PAGES",
            "MAX_U",
            "N_GRP_S",
        ]
    )
    def _qsa_q4_gqa_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        sc_ptr,
        bt_ptr,
        u_packed_ptr,
        u_cnt_ptr,
        bad_ptr,
        out_ptr,
        stride_qb: tl.int64,
        stride_qh: tl.int64,
        stride_qs: tl.int64,
        stride_kp: tl.int64,
        stride_kkv: tl.int64,
        stride_kh: tl.int64,
        stride_kt: tl.int64,
        stride_vp: tl.int64,
        stride_vh: tl.int64,
        stride_vt: tl.int64,
        stride_scp: tl.int64,
        stride_sckv: tl.int64,
        stride_sch: tl.int64,
        stride_sct: tl.int64,
        stride_btb: tl.int64,
        stride_btp: tl.int64,
        stride_ob: tl.int64,
        stride_oh: tl.int64,
        stride_os: tl.int64,
        S,
        T,
        MAX_PAGES,
        NUM_PAGES,
        MAX_U,
        N_GRP_S,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        HK: tl.constexpr,
        G: tl.constexpr,
        M_ROWS: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BK_Q: tl.constexpr,
        IS_PAGED: tl.constexpr,
        IS_FP8: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        G_BLK: tl.constexpr = 8,
        SUB_G: tl.constexpr = 1,
        CHECK_BAD: tl.constexpr = False,
    ):
        """Q4 deduplicated 4-query GQA TensorCore prefill kernel (dense or paged, BF16 or FP8)."""
        g_s = N_GRP_S - 1 - tl.program_id(0).to(tl.int64)
        b = tl.program_id(1).to(tl.int64)
        pid_hs = tl.program_id(2).to(tl.int64)
        hk = pid_hs // SUB_G
        sub_g = pid_hs % SUB_G
        grp = b * N_GRP_S + g_s
        if CHECK_BAD:
            if tl.load(bad_ptr + grp) != 0:
                return

        r = tl.arange(0, M_ROWS)
        q_j = r // G_BLK
        pos = g_s * 4 + q_j
        q_local = sub_g * G_BLK + (r % G_BLK)
        live = (r < 4 * G_BLK) & (q_local < G) & (pos < S)
        h = hk * G + q_local
        d = tl.arange(0, D)

        qv = tl.load(
            q_ptr + b * stride_qb + h[:, None] * stride_qh + pos[:, None] * stride_qs + d[None, :],
            mask=live[:, None],
            other=0.0,
        )
        u_cnt = tl.minimum(tl.load(u_cnt_ptr + grp), MAX_U)
        mx = tl.full((M_ROWS,), float("-inf"), tl.float32)
        z = tl.zeros((M_ROWS,), tl.float32)
        acc = tl.zeros((M_ROWS, D), tl.float32)

        q_u_base = u_packed_ptr + grp * MAX_U
        QUADS_PER_PAGE: tl.constexpr = PAGE_SIZE // 4
        sub_offs = tl.arange(0, 4)

        for start in range(0, u_cnt, BK_Q):
            qo = start + tl.arange(0, BK_Q)
            qin = qo < u_cnt
            packed = tl.load(q_u_base + qo, mask=qin, other=0).to(tl.uint32)
            quad_id = (packed >> 16).to(tl.int32)
            qmask16 = (packed & 0xFFFF).to(tl.int32)

            masks_tok = tl.broadcast_to(qmask16[:, None], (BK_Q, 4)).reshape(BK_Q * 4)
            sub_tok = tl.broadcast_to(sub_offs[None, :], (BK_Q, 4)).reshape(BK_Q * 4)
            shift = q_j[:, None] * 4 + sub_tok[None, :]

            if IS_PAGED:
                log_p = tl.where(qin, quad_id // QUADS_PER_PAGE, 0)
                p_in = qin & (log_p < MAX_PAGES)
                phys_q = tl.load(
                    bt_ptr + b * stride_btb + tl.minimum(log_p, MAX_PAGES - 1) * stride_btp,
                    mask=p_in,
                    other=0,
                ).to(tl.int64)
                slot_q = (quad_id % QUADS_PER_PAGE)[:, None] * 4 + sub_offs[None, :]
                tok_glob = (quad_id[:, None] * 4 + sub_offs[None, :]).reshape(BK_Q * 4)
                q_ok = p_in & (phys_q > 0) & (phys_q < NUM_PAGES)
                phys_tok = tl.broadcast_to(phys_q[:, None], (BK_Q, 4)).reshape(BK_Q * 4)
                slot_tok = slot_q.reshape(BK_Q * 4)
                t_ok = (
                    tl.broadcast_to(q_ok[:, None], (BK_Q, 4)).reshape(BK_Q * 4)
                    & (tok_glob < T)
                )
                valid = live[:, None] & t_ok[None, :] & (((masks_tok[None, :] >> shift) & 1) != 0)
                k_ptrs = (
                    k_ptr
                    + phys_tok[:, None] * stride_kp
                    + hk * stride_kh
                    + slot_tok[:, None] * stride_kt
                    + d[None, :]
                )
                v_ptrs = k_ptrs + stride_kkv
                if HAS_SCALE:
                    sc_base = (
                        sc_ptr
                        + phys_tok * stride_scp
                        + hk * stride_sch
                        + slot_tok * stride_sct
                    )
            else:
                tok_q = quad_id[:, None] * 4 + sub_offs[None, :]
                tok_glob = tok_q.reshape(BK_Q * 4)
                t_ok = tl.broadcast_to(qin[:, None], (BK_Q, 4)).reshape(BK_Q * 4) & (tok_glob < T)
                safe_tok = tl.where(t_ok, tok_glob, 0).to(tl.int64)
                valid = live[:, None] & t_ok[None, :] & (((masks_tok[None, :] >> shift) & 1) != 0)
                k_ptrs = k_ptr + b * stride_kp + hk * stride_kh + safe_tok[:, None] * stride_kt + d[None, :]
                v_ptrs = v_ptr + b * stride_vp + hk * stride_vh + safe_tok[:, None] * stride_vt + d[None, :]

            k_val = tl.load(k_ptrs, mask=t_ok[:, None], other=0.0)
            if IS_FP8:
                k_val = k_val.to(tl.bfloat16)
            logits = tl.dot(qv, tl.trans(k_val)) * SCALE
            if IS_FP8 and HAS_SCALE:
                sc_k = tl.load(sc_base, mask=t_ok, other=1.0).to(tl.float32)
                logits = logits * sc_k[None, :]
            logits = tl.where(valid, logits, float("-inf"))
            nxt = tl.maximum(mx, tl.max(logits, 1))
            rescale = tl.where(nxt == float("-inf"), 0.0, tl.exp(mx - nxt))
            prob = tl.where(valid, tl.exp(logits - nxt[:, None]), 0.0)

            v_val = tl.load(v_ptrs, mask=t_ok[:, None], other=0.0)
            if IS_FP8:
                if HAS_SCALE:
                    sc_v = tl.load(sc_base + stride_sckv, mask=t_ok, other=1.0).to(tl.float32)
                    prob_v = (prob * sc_v[None, :]).to(tl.bfloat16)
                else:
                    prob_v = prob.to(tl.bfloat16)
                v_val = v_val.to(tl.bfloat16)
            else:
                prob_v = prob.to(tl.bfloat16)

            acc = acc * rescale[:, None]
            acc = tl.dot(prob_v, v_val, acc)
            z = z * rescale + tl.sum(prob, 1)
            mx = nxt

        res = tl.where(z[:, None] > 0, acc / z[:, None], 0.0)
        tl.store(
            out_ptr + b * stride_ob + h[:, None] * stride_oh + pos[:, None] * stride_os + d[None, :],
            res.to(out_ptr.dtype.element_ty),
            mask=live[:, None],
        )

    @triton.jit(
        do_not_specialize=[
            "stride_sc_p",
            "stride_sc_kv",
            "stride_sc_h",
            "stride_sc_t",
            "stride_bt_b",
            "stride_bt_p",
            "stride_idx_b",
            "stride_idx_s",
            "K",
            "T",
            "MAX_PAGES",
            "NUM_PAGES",
            "MAX_BLKS",
            "N_BAD_S",
        ]
    )
    def _qsa_sparse_prefill_grouped_fast_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        sc_ptr,
        bt_ptr,
        idx_ptr,
        bad_ptr,
        out_ptr,
        stride_q_b: tl.int64,
        stride_q_h: tl.int64,
        stride_q_s: tl.int64,
        stride_out_b: tl.int64,
        stride_out_h: tl.int64,
        stride_out_s: tl.int64,
        stride_k_b: tl.int64,
        stride_k_kv: tl.int64,
        stride_k_h: tl.int64,
        stride_k_t: tl.int64,
        stride_v_b: tl.int64,
        stride_v_h: tl.int64,
        stride_v_t: tl.int64,
        stride_sc_p: tl.int64,
        stride_sc_kv: tl.int64,
        stride_sc_h: tl.int64,
        stride_sc_t: tl.int64,
        stride_bt_b: tl.int64,
        stride_bt_p: tl.int64,
        stride_idx_b: tl.int64,
        stride_idx_s: tl.int64,
        K,
        T,
        MAX_PAGES,
        NUM_PAGES,
        MAX_BLKS,
        N_BAD_S,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        q_per_kv: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLK_K: tl.constexpr,
        BLK_H: tl.constexpr,
        IS_PAGED: tl.constexpr,
        IS_FP8: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        SUB_G: tl.constexpr = 1,
        ONLY_BAD: tl.constexpr = False,
        BAD_SHIFT: tl.constexpr = 2,
    ):
        """Grouped sparse FMHA with empty-block skip and split-BF16 P@V for SM80+/SM103 (dense or paged)."""
        pid_s = tl.num_programs(0).to(tl.int64) - 1 - tl.program_id(0).to(tl.int64)
        pid_b = tl.program_id(1).to(tl.int64)
        if ONLY_BAD:
            if tl.load(bad_ptr + pid_b * N_BAD_S + (pid_s >> BAD_SHIFT)) == 0:
                return
        pid_g = tl.program_id(2).to(tl.int64)
        pid_kv = pid_g // SUB_G
        sub_g = pid_g % SUB_G

        rh = tl.arange(0, BLK_H)
        rd = tl.arange(0, D)
        q_off = sub_g * BLK_H + rh
        live_h = q_off < q_per_kv
        q = tl.load(
            q_ptr
            + pid_b * stride_q_b
            + (pid_kv * q_per_kv + q_off[:, None]) * stride_q_h
            + pid_s * stride_q_s
            + rd[None, :],
            mask=live_h[:, None],
            other=0.0,
        )
        idx_base = idx_ptr + pid_b * stride_idx_b + pid_s * stride_idx_s
        max_score = tl.full((BLK_H,), float("-inf"), tl.float32)
        sum_exp = tl.zeros((BLK_H,), tl.float32)
        acc = tl.zeros((BLK_H, D), tl.float32)

        for blk in range(MAX_BLKS):
            tr = blk * BLK_K + tl.arange(0, BLK_K)
            idx = tl.load(idx_base + tr, mask=tr < K, other=-1)
            valid = (tr < K) & (idx >= 0) & (idx < T)
            if tl.sum(valid.to(tl.int32), 0) > 0:
                if IS_PAGED:
                    safe_idx = tl.where(valid, idx, 0)
                    log_p = safe_idx // PAGE_SIZE
                    p_ok = valid & (log_p < MAX_PAGES)
                    phys = tl.load(
                        bt_ptr + pid_b * stride_bt_b + tl.minimum(log_p, MAX_PAGES - 1) * stride_bt_p,
                        mask=p_ok,
                        other=0,
                    ).to(tl.int64)
                    valid = p_ok & (phys > 0) & (phys < NUM_PAGES)
                    slot = (safe_idx % PAGE_SIZE).to(tl.int64)
                    k_ptrs = k_ptr + phys[:, None] * stride_k_b + pid_kv * stride_k_h + slot[:, None] * stride_k_t + rd[None, :]
                    v_ptrs = k_ptrs + stride_k_kv
                    if HAS_SCALE:
                        sc_base = sc_ptr + phys * stride_sc_p + pid_kv * stride_sc_h + slot * stride_sc_t
                else:
                    safe_idx = tl.where(valid, idx, 0).to(tl.int64)
                    k_ptrs = k_ptr + pid_b * stride_k_b + pid_kv * stride_k_h + safe_idx[:, None] * stride_k_t + rd[None, :]
                    v_ptrs = v_ptr + pid_b * stride_v_b + pid_kv * stride_v_h + safe_idx[:, None] * stride_v_t + rd[None, :]

                k = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
                if IS_FP8:
                    k = k.to(tl.bfloat16)
                score = tl.dot(q, tl.trans(k)) * SCALE
                if IS_FP8 and HAS_SCALE:
                    sc_k = tl.load(sc_base, mask=valid, other=1.0).to(tl.float32)
                    score = score * sc_k[None, :]
                score = tl.where(valid[None, :], score, float("-inf"))
                next_max = tl.maximum(max_score, tl.max(score, axis=1))
                rescale = tl.where(
                    next_max == float("-inf"), 0.0, tl.exp(max_score - next_max)
                )
                weights = tl.where(valid[None, :], tl.exp(score - next_max[:, None]), 0.0)
                v = tl.load(v_ptrs, mask=valid[:, None], other=0.0)
                if IS_FP8:
                    if HAS_SCALE:
                        sc_v = tl.load(sc_base + stride_sc_kv, mask=valid, other=1.0).to(tl.float32)
                        w_f = weights * sc_v[None, :]
                    else:
                        w_f = weights
                    v = v.to(tl.bfloat16)
                else:
                    w_f = weights
                w_pv = w_f.to(tl.bfloat16)
                acc = acc * rescale[:, None]
                acc = tl.dot(w_pv, v, acc)
                sum_exp = sum_exp * rescale + tl.sum(weights, axis=1)
                max_score = next_max

        out = tl.where(sum_exp[:, None] > 0, acc / sum_exp[:, None], 0.0)
        tl.store(
            out_ptr
            + pid_b * stride_out_b
            + (pid_kv * q_per_kv + q_off[:, None]) * stride_out_h
            + pid_s * stride_out_s
            + rd[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=live_h[:, None],
        )

    @triton.jit(
        do_not_specialize=[
            "stride_sc_p",
            "stride_sc_kv",
            "stride_sc_h",
            "stride_sc_t",
            "stride_bt_b",
            "stride_bt_p",
            "stride_idx_b",
            "stride_idx_s",
            "K",
            "T",
            "MAX_PAGES",
            "NUM_PAGES",
            "MAX_BLKS",
            "N_BAD_S",
        ]
    )
    def _qsa_sparse_prefill_dsplit_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        sc_ptr,
        bt_ptr,
        idx_ptr,
        bad_ptr,
        out_ptr,
        stride_q_b: tl.int64,
        stride_q_h: tl.int64,
        stride_q_s: tl.int64,
        stride_out_b: tl.int64,
        stride_out_h: tl.int64,
        stride_out_s: tl.int64,
        stride_k_b: tl.int64,
        stride_k_kv: tl.int64,
        stride_k_h: tl.int64,
        stride_k_t: tl.int64,
        stride_v_b: tl.int64,
        stride_v_h: tl.int64,
        stride_v_t: tl.int64,
        stride_sc_p: tl.int64,
        stride_sc_kv: tl.int64,
        stride_sc_h: tl.int64,
        stride_sc_t: tl.int64,
        stride_bt_b: tl.int64,
        stride_bt_p: tl.int64,
        stride_idx_b: tl.int64,
        stride_idx_s: tl.int64,
        K,
        T,
        MAX_PAGES,
        NUM_PAGES,
        MAX_BLKS,
        N_BAD_S,
        SCALE: tl.constexpr,
        q_per_kv: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLK_K: tl.constexpr,
        BLK_H: tl.constexpr,
        IS_PAGED: tl.constexpr,
        IS_FP8: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        FP8_QK: tl.constexpr,
        SUB_G: tl.constexpr = 1,
        ONLY_BAD: tl.constexpr = False,
        BAD_SHIFT: tl.constexpr = 2,
    ):
        """Sequential D=256 (2x128) grouped QSA prefill avoiding register spills + native FP8 QK TC."""
        pid_s = tl.num_programs(0).to(tl.int64) - 1 - tl.program_id(0).to(tl.int64)
        pid_b = tl.program_id(1).to(tl.int64)
        if ONLY_BAD:
            if tl.load(bad_ptr + pid_b * N_BAD_S + (pid_s >> BAD_SHIFT)) == 0:
                return
        pid_g = tl.program_id(2).to(tl.int64)
        pid_kv = pid_g // SUB_G
        sub_g = pid_g % SUB_G

        rh = tl.arange(0, BLK_H)
        q_off = sub_g * BLK_H + rh
        live_h = q_off < q_per_kv
        r128 = tl.arange(0, 128)
        q_base = (
            q_ptr
            + pid_b * stride_q_b
            + (pid_kv * q_per_kv + q_off[:, None]) * stride_q_h
            + pid_s * stride_q_s
        )
        q0 = tl.load(q_base + r128[None, :], mask=live_h[:, None], other=0.0)
        q1 = tl.load(q_base + 128 + r128[None, :], mask=live_h[:, None], other=0.0)
        if IS_FP8 and FP8_QK:
            q_abs = tl.maximum(
                tl.max(tl.abs(q0.to(tl.float32)), axis=1),
                tl.max(tl.abs(q1.to(tl.float32)), axis=1),
            )
            q_sc = tl.maximum(q_abs, 1e-6) / 448.0
            q0_f8 = (q0.to(tl.float32) / q_sc[:, None]).to(tl.float8e4nv)
            q1_f8 = (q1.to(tl.float32) / q_sc[:, None]).to(tl.float8e4nv)
            q_scale_vec = q_sc * SCALE

        idx_base = idx_ptr + pid_b * stride_idx_b + pid_s * stride_idx_s
        max_score = tl.full((BLK_H,), float("-inf"), tl.float32)
        sum_exp = tl.zeros((BLK_H,), tl.float32)
        acc0 = tl.zeros((BLK_H, 128), tl.float32)
        acc1 = tl.zeros((BLK_H, 128), tl.float32)

        for blk in range(MAX_BLKS):
            tr = blk * BLK_K + tl.arange(0, BLK_K)
            idx = tl.load(idx_base + tr, mask=tr < K, other=-1)
            valid = (tr < K) & (idx >= 0) & (idx < T)
            if tl.sum(valid.to(tl.int32), 0) > 0:
                if IS_PAGED:
                    safe_idx = tl.where(valid, idx, 0)
                    log_p = safe_idx // PAGE_SIZE
                    p_ok = valid & (log_p < MAX_PAGES)
                    phys = tl.load(
                        bt_ptr
                        + pid_b * stride_bt_b
                        + tl.minimum(log_p, MAX_PAGES - 1) * stride_bt_p,
                        mask=p_ok,
                        other=0,
                    ).to(tl.int64)
                    valid = p_ok & (phys > 0) & (phys < NUM_PAGES)
                    slot = (safe_idx % PAGE_SIZE).to(tl.int64)
                    k_base = (
                        k_ptr
                        + phys[:, None] * stride_k_b
                        + pid_kv * stride_k_h
                        + slot[:, None] * stride_k_t
                    )
                    v_base = k_base + stride_k_kv
                    if HAS_SCALE:
                        sc_base = (
                            sc_ptr
                            + phys * stride_sc_p
                            + pid_kv * stride_sc_h
                            + slot * stride_sc_t
                        )
                else:
                    safe_idx = tl.where(valid, idx, 0).to(tl.int64)
                    k_base = (
                        k_ptr
                        + pid_b * stride_k_b
                        + pid_kv * stride_k_h
                        + safe_idx[:, None] * stride_k_t
                    )
                    v_base = (
                        v_ptr
                        + pid_b * stride_v_b
                        + pid_kv * stride_v_h
                        + safe_idx[:, None] * stride_v_t
                    )

                k0 = tl.load(k_base + r128[None, :], mask=valid[:, None], other=0.0)
                if IS_FP8 and FP8_QK:
                    score = tl.dot(q0_f8, tl.trans(k0))
                else:
                    if IS_FP8:
                        k0 = k0.to(tl.bfloat16)
                    score = tl.dot(q0, tl.trans(k0))
                k1 = tl.load(k_base + 128 + r128[None, :], mask=valid[:, None], other=0.0)
                if IS_FP8 and FP8_QK:
                    score = tl.dot(q1_f8, tl.trans(k1), score) * q_scale_vec[:, None]
                else:
                    if IS_FP8:
                        k1 = k1.to(tl.bfloat16)
                    score = tl.dot(q1, tl.trans(k1), score) * SCALE
                if IS_FP8 and HAS_SCALE:
                    sc_k = tl.load(sc_base, mask=valid, other=1.0).to(tl.float32)
                    score = score * sc_k[None, :]
                score = tl.where(valid[None, :], score, float("-inf"))
                next_max = tl.maximum(max_score, tl.max(score, axis=1))
                rescale = tl.where(
                    next_max == float("-inf"), 0.0, tl.exp(max_score - next_max)
                )
                weights = tl.where(valid[None, :], tl.exp(score - next_max[:, None]), 0.0)
                if IS_FP8 and HAS_SCALE:
                    sc_v = tl.load(sc_base + stride_sc_kv, mask=valid, other=1.0).to(tl.float32)
                    w_pv = (weights * sc_v[None, :]).to(tl.bfloat16)
                else:
                    w_pv = weights.to(tl.bfloat16)
                v0 = tl.load(v_base + r128[None, :], mask=valid[:, None], other=0.0)
                if IS_FP8:
                    v0 = v0.to(tl.bfloat16)
                acc0 = acc0 * rescale[:, None]
                acc0 = tl.dot(w_pv, v0, acc0)
                v1 = tl.load(v_base + 128 + r128[None, :], mask=valid[:, None], other=0.0)
                if IS_FP8:
                    v1 = v1.to(tl.bfloat16)
                acc1 = acc1 * rescale[:, None]
                acc1 = tl.dot(w_pv, v1, acc1)
                sum_exp = sum_exp * rescale + tl.sum(weights, axis=1)
                max_score = next_max

        inv_sum = tl.where(sum_exp > 0, 1.0 / sum_exp, 0.0)[:, None]
        out_base = (
            out_ptr
            + pid_b * stride_out_b
            + (pid_kv * q_per_kv + q_off[:, None]) * stride_out_h
            + pid_s * stride_out_s
        )
        tl.store(
            out_base + r128[None, :],
            (acc0 * inv_sum).to(out_ptr.dtype.element_ty),
            mask=live_h[:, None],
        )
        tl.store(
            out_base + 128 + r128[None, :],
            (acc1 * inv_sum).to(out_ptr.dtype.element_ty),
            mask=live_h[:, None],
        )

    @triton.jit
    def _qsa_gqa_paged_split_tc_kernel(
        q_ptr,
        kv_ptr,
        sc_ptr,
        bt_ptr,
        kv_lens_ptr,
        sel_ptr,
        p_out_ptr,
        p_m_ptr,
        p_z_ptr,
        gate_ptr,
        stride_qb: tl.int64,
        stride_qh: tl.int64,
        stride_qs: tl.int64,
        stride_kp: tl.int64,
        stride_kkv: tl.int64,
        stride_kh: tl.int64,
        stride_kt: tl.int64,
        stride_scp: tl.int64,
        stride_sckv: tl.int64,
        stride_sch: tl.int64,
        stride_sct: tl.int64,
        stride_btb: tl.int64,
        stride_btp: tl.int64,
        stride_lb: tl.int64,
        stride_ls: tl.int64,
        stride_sb: tl.int64,
        stride_ss: tl.int64,
        stride_ob: tl.int64,
        stride_oh: tl.int64,
        stride_os: tl.int64,
        S,
        K,
        MAX_PAGES,
        NUM_PAGES,
        HK: tl.constexpr,
        G: tl.constexpr,
        M_ROWS: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        KT: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        WITH_GATE: tl.constexpr,
        IS_FP8: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        SUB_G: tl.constexpr = 1,
    ):
        """GQA TensorCore paged split-KV kernel for QSA decode/verify (BF16 & FP8)."""
        pid_m = tl.program_id(0).to(tl.int64)
        split = tl.program_id(1).to(tl.int64)
        hk_sub = pid_m % (HK * SUB_G)
        bs = pid_m // (HK * SUB_G)
        hk = hk_sub // SUB_G
        sub_g = hk_sub % SUB_G
        b = bs // S
        s = bs % S

        r = tl.arange(0, M_ROWS)
        q_off = sub_g * M_ROWS + r
        live_h = q_off < G
        h = hk * G + q_off
        d = tl.arange(0, D)

        qv = tl.load(
            q_ptr + b * stride_qb + h[:, None] * stride_qh + s * stride_qs + d[None, :],
            mask=live_h[:, None],
            other=0.0,
        )
        kv_len = tl.load(kv_lens_ptr + b * stride_lb + s * stride_ls).to(tl.int32)
        sel_row = sel_ptr + b * stride_sb + s * stride_ss

        mx = tl.full((M_ROWS,), float("-inf"), tl.float32)
        z = tl.zeros((M_ROWS,), tl.float32)
        acc = tl.zeros((M_ROWS, D), tl.float32)

        step = NUM_SPLITS * KT
        n_blks = tl.maximum(0, tl.cdiv(K - split * KT, step))
        for blk in range(n_blks):
            off = (blk * NUM_SPLITS + split) * KT + tl.arange(0, KT)
            in_k = off < K
            tok = tl.load(sel_row + off, mask=in_k, other=-1)
            cand = in_k & (tok >= 0) & (tok < kv_len)
            safe_tok = tl.where(cand, tok, 0)
            log_p = safe_tok // PAGE_SIZE
            p_in = cand & (log_p < MAX_PAGES)
            phys = tl.load(
                bt_ptr + b * stride_btb + tl.minimum(log_p, MAX_PAGES - 1) * stride_btp,
                mask=p_in,
                other=0,
            ).to(tl.int64)
            valid = p_in & (phys > 0) & (phys < NUM_PAGES)
            if tl.sum(valid.to(tl.int32), 0) > 0:
                slot = (safe_tok % PAGE_SIZE).to(tl.int64)
                k_ptrs = (
                    kv_ptr
                    + phys[:, None] * stride_kp
                    + hk * stride_kh
                    + slot[:, None] * stride_kt
                    + d[None, :]
                )
                k_val = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
                if IS_FP8:
                    k_val = k_val.to(tl.bfloat16)
                logits = tl.dot(qv, tl.trans(k_val)) * SCALE
                if IS_FP8 and HAS_SCALE:
                    sc_base = sc_ptr + phys * stride_scp + hk * stride_sch + slot * stride_sct
                    sc_k = tl.load(sc_base, mask=valid, other=1.0).to(tl.float32)
                    logits = logits * sc_k[None, :]
                logits = tl.where(valid[None, :], logits, float("-inf"))
                nxt = tl.maximum(mx, tl.max(logits, 1))
                rescale = tl.where(nxt == float("-inf"), 0.0, tl.exp(mx - nxt))
                prob = tl.where(valid[None, :], tl.exp(logits - nxt[:, None]), 0.0)
                v_val = tl.load(k_ptrs + stride_kkv, mask=valid[:, None], other=0.0)
                if IS_FP8:
                    if HAS_SCALE:
                        sc_v = tl.load(sc_base + stride_sckv, mask=valid, other=1.0).to(tl.float32)
                        prob_v = (prob * sc_v[None, :]).to(tl.bfloat16)
                    else:
                        prob_v = prob.to(tl.bfloat16)
                    v_val = v_val.to(tl.bfloat16)
                else:
                    prob_v = prob.to(tl.bfloat16)
                acc = acc * rescale[:, None]
                acc = tl.dot(prob_v, v_val, acc)
                z = z * rescale + tl.sum(prob, 1)
                mx = nxt

        if NUM_SPLITS == 1:
            res = tl.where(z[:, None] > 0, acc / z[:, None], 0.0)
            if WITH_GATE:
                hq_tot = HK * G
                g_offs = ((b * S + s) * hq_tot + h[:, None]) * D + d[None, :]
                gate = tl.load(gate_ptr + g_offs, mask=live_h[:, None], other=0.0).to(tl.float32)
                sig = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
                res = res.to(tl.bfloat16).to(tl.float32) * sig
            tl.store(
                p_out_ptr + b * stride_ob + h[:, None] * stride_oh + s * stride_os + d[None, :],
                res.to(p_out_ptr.dtype.element_ty),
                mask=live_h[:, None],
            )
        else:
            row_base = (bs * (HK * G) + h) * NUM_SPLITS + split
            tl.store(
                p_out_ptr + row_base[:, None] * D + d[None, :],
                acc,
                mask=live_h[:, None],
            )
            tl.store(p_m_ptr + row_base, mx, mask=live_h)
            tl.store(p_z_ptr + row_base, z, mask=live_h)

    @triton.jit
    def _qsa_gqa_paged_split_tc_dsplit_kernel(
        q_ptr,
        kv_ptr,
        sc_ptr,
        bt_ptr,
        kv_lens_ptr,
        sel_ptr,
        p_out_ptr,
        p_m_ptr,
        p_z_ptr,
        gate_ptr,
        stride_qb: tl.int64,
        stride_qh: tl.int64,
        stride_qs: tl.int64,
        stride_kp: tl.int64,
        stride_kkv: tl.int64,
        stride_kh: tl.int64,
        stride_kt: tl.int64,
        stride_scp: tl.int64,
        stride_sckv: tl.int64,
        stride_sch: tl.int64,
        stride_sct: tl.int64,
        stride_btb: tl.int64,
        stride_btp: tl.int64,
        stride_lb: tl.int64,
        stride_ls: tl.int64,
        stride_sb: tl.int64,
        stride_ss: tl.int64,
        stride_ob: tl.int64,
        stride_oh: tl.int64,
        stride_os: tl.int64,
        S,
        K,
        MAX_PAGES,
        NUM_PAGES,
        HK: tl.constexpr,
        G: tl.constexpr,
        M_ROWS: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        SCALE: tl.constexpr,
        KT: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        WITH_GATE: tl.constexpr,
        IS_FP8: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        FP8_QK: tl.constexpr,
        SUB_G: tl.constexpr = 1,
    ):
        """Sequential D=256 (2x128) split-KV decode/verify kernel eliminating register spills."""
        pid_m = tl.program_id(0).to(tl.int64)
        split = tl.program_id(1).to(tl.int64)
        hk_sub = pid_m % (HK * SUB_G)
        bs = pid_m // (HK * SUB_G)
        hk = hk_sub // SUB_G
        sub_g = hk_sub % SUB_G
        b = bs // S
        s = bs % S

        r = tl.arange(0, M_ROWS)
        q_off = sub_g * M_ROWS + r
        live_h = q_off < G
        h = hk * G + q_off
        r128 = tl.arange(0, 128)

        q_base = q_ptr + b * stride_qb + h[:, None] * stride_qh + s * stride_qs
        q0 = tl.load(q_base + r128[None, :], mask=live_h[:, None], other=0.0)
        q1 = tl.load(q_base + 128 + r128[None, :], mask=live_h[:, None], other=0.0)
        if IS_FP8 and FP8_QK:
            q_abs = tl.maximum(
                tl.max(tl.abs(q0.to(tl.float32)), axis=1),
                tl.max(tl.abs(q1.to(tl.float32)), axis=1),
            )
            q_sc = tl.maximum(q_abs, 1e-6) / 448.0
            q0_f8 = (q0.to(tl.float32) / q_sc[:, None]).to(tl.float8e4nv)
            q1_f8 = (q1.to(tl.float32) / q_sc[:, None]).to(tl.float8e4nv)
            q_scale_vec = q_sc * SCALE

        kv_len = tl.load(kv_lens_ptr + b * stride_lb + s * stride_ls).to(tl.int32)
        sel_row = sel_ptr + b * stride_sb + s * stride_ss

        mx = tl.full((M_ROWS,), float("-inf"), tl.float32)
        z = tl.zeros((M_ROWS,), tl.float32)
        acc0 = tl.zeros((M_ROWS, 128), tl.float32)
        acc1 = tl.zeros((M_ROWS, 128), tl.float32)

        step = NUM_SPLITS * KT
        n_blks = tl.maximum(0, tl.cdiv(K - split * KT, step))
        for blk in range(n_blks):
            off = (blk * NUM_SPLITS + split) * KT + tl.arange(0, KT)
            in_k = off < K
            tok = tl.load(sel_row + off, mask=in_k, other=-1)
            cand = in_k & (tok >= 0) & (tok < kv_len)
            safe_tok = tl.where(cand, tok, 0)
            log_p = safe_tok // PAGE_SIZE
            p_in = cand & (log_p < MAX_PAGES)
            phys = tl.load(
                bt_ptr + b * stride_btb + tl.minimum(log_p, MAX_PAGES - 1) * stride_btp,
                mask=p_in,
                other=0,
            ).to(tl.int64)
            valid = p_in & (phys > 0) & (phys < NUM_PAGES)
            if tl.sum(valid.to(tl.int32), 0) > 0:
                slot = (safe_tok % PAGE_SIZE).to(tl.int64)
                k_base = (
                    kv_ptr
                    + phys[:, None] * stride_kp
                    + hk * stride_kh
                    + slot[:, None] * stride_kt
                )
                k0 = tl.load(k_base + r128[None, :], mask=valid[:, None], other=0.0)
                if IS_FP8 and FP8_QK:
                    logits = tl.dot(q0_f8, tl.trans(k0))
                else:
                    if IS_FP8:
                        k0 = k0.to(tl.bfloat16)
                    logits = tl.dot(q0, tl.trans(k0))
                k1 = tl.load(k_base + 128 + r128[None, :], mask=valid[:, None], other=0.0)
                if IS_FP8 and FP8_QK:
                    logits = tl.dot(q1_f8, tl.trans(k1), logits) * q_scale_vec[:, None]
                else:
                    if IS_FP8:
                        k1 = k1.to(tl.bfloat16)
                    logits = tl.dot(q1, tl.trans(k1), logits) * SCALE
                if IS_FP8 and HAS_SCALE:
                    sc_base = sc_ptr + phys * stride_scp + hk * stride_sch + slot * stride_sct
                    sc_k = tl.load(sc_base, mask=valid, other=1.0).to(tl.float32)
                    logits = logits * sc_k[None, :]
                logits = tl.where(valid[None, :], logits, float("-inf"))
                nxt = tl.maximum(mx, tl.max(logits, 1))
                rescale = tl.where(nxt == float("-inf"), 0.0, tl.exp(mx - nxt))
                prob = tl.where(valid[None, :], tl.exp(logits - nxt[:, None]), 0.0)
                v_base = k_base + stride_kkv
                if IS_FP8 and HAS_SCALE:
                    sc_v = tl.load(sc_base + stride_sckv, mask=valid, other=1.0).to(tl.float32)
                    prob_v = (prob * sc_v[None, :]).to(tl.bfloat16)
                else:
                    prob_v = prob.to(tl.bfloat16)
                v0 = tl.load(v_base + r128[None, :], mask=valid[:, None], other=0.0)
                if IS_FP8:
                    v0 = v0.to(tl.bfloat16)
                acc0 = acc0 * rescale[:, None]
                acc0 = tl.dot(prob_v, v0, acc0)
                v1 = tl.load(v_base + 128 + r128[None, :], mask=valid[:, None], other=0.0)
                if IS_FP8:
                    v1 = v1.to(tl.bfloat16)
                acc1 = acc1 * rescale[:, None]
                acc1 = tl.dot(prob_v, v1, acc1)
                z = z * rescale + tl.sum(prob, 1)
                mx = nxt

        if NUM_SPLITS == 1:
            inv_z = tl.where(z > 0, 1.0 / z, 0.0)[:, None]
            o0 = acc0 * inv_z
            o1 = acc1 * inv_z
            if WITH_GATE:
                hq_tot = HK * G
                g_base = gate_ptr + ((b * S + s) * hq_tot + h[:, None]) * 256
                g0 = tl.load(g_base + r128[None, :], mask=live_h[:, None], other=0.0).to(tl.float32)
                g1 = tl.load(g_base + 128 + r128[None, :], mask=live_h[:, None], other=0.0).to(tl.float32)
                o0 = o0.to(tl.bfloat16).to(tl.float32) * tl.sigmoid(g0).to(tl.bfloat16).to(tl.float32)
                o1 = o1.to(tl.bfloat16).to(tl.float32) * tl.sigmoid(g1).to(tl.bfloat16).to(tl.float32)
            out_base = p_out_ptr + b * stride_ob + h[:, None] * stride_oh + s * stride_os
            tl.store(
                out_base + r128[None, :],
                o0.to(p_out_ptr.dtype.element_ty),
                mask=live_h[:, None],
            )
            tl.store(
                out_base + 128 + r128[None, :],
                o1.to(p_out_ptr.dtype.element_ty),
                mask=live_h[:, None],
            )
        else:
            row_base = (bs * (HK * G) + h) * NUM_SPLITS + split
            po_base = p_out_ptr + row_base[:, None] * 256
            tl.store(po_base + r128[None, :], acc0, mask=live_h[:, None])
            tl.store(po_base + 128 + r128[None, :], acc1, mask=live_h[:, None])
            tl.store(p_m_ptr + row_base, mx, mask=live_h)
            tl.store(p_z_ptr + row_base, z, mask=live_h)

    @triton.jit
    def _qsa_gqa_paged_merge_kernel(
        p_out_ptr,
        p_m_ptr,
        p_z_ptr,
        out_ptr,
        gate_ptr,
        stride_ob: tl.int64,
        stride_oh: tl.int64,
        stride_os: tl.int64,
        S,
        HQ: tl.constexpr,
        D: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SP_PAD: tl.constexpr,
        WITH_GATE: tl.constexpr,
    ):
        """Merge split-KV partial accumulators with optional fused BF16 sigmoid output gate."""
        row = tl.program_id(0).to(tl.int64)
        h = row % HQ
        bs = row // HQ
        s = bs % S
        b = bs // S

        sp = tl.arange(0, SP_PAD)
        sp_m = sp < NUM_SPLITS
        rd = tl.arange(0, D)

        base_ms = row * NUM_SPLITS + sp
        m = tl.load(p_m_ptr + base_ms, mask=sp_m, other=float("-inf"))
        z = tl.load(p_z_ptr + base_ms, mask=sp_m, other=0.0)
        gmax = tl.max(m, axis=0)
        scale = tl.where(sp_m & (z > 0), tl.exp(m - gmax), 0.0)
        denom = tl.sum(z * scale, axis=0)

        p = tl.load(
            p_out_ptr + (row * NUM_SPLITS + sp[:, None]) * D + rd[None, :],
            mask=sp_m[:, None],
            other=0.0,
        )
        num = tl.sum(p * scale[:, None], axis=0)
        out = tl.where(denom > 0, num / denom, 0.0)
        if WITH_GATE:
            gate = tl.load(gate_ptr + (bs * HQ + h) * D + rd).to(tl.float32)
            sig = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
            out = out.to(tl.bfloat16).to(tl.float32) * sig
        tl.store(
            out_ptr + b * stride_ob + h * stride_oh + s * stride_os + rd,
            out.to(out_ptr.dtype.element_ty),
        )


def _view_mha_kv_and_scale_hnd(
    kv_cache_base: torch.Tensor,
    kv_scale: Optional[torch.Tensor],
    *,
    kv_heads: int,
    page_size: int,
    head_dim: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return zero-copy rank-5 HND KV cache and optional rank-4 HND/broadcast FP32 scale."""
    if kv_cache_base.ndim == 2:
        pages = int(kv_cache_base.shape[0])
        required = 2 * kv_heads * page_size * head_dim
        if int(kv_cache_base.shape[1]) < required or kv_cache_base.stride(1) != 1:
            raise ValueError(
                f"Invalid packed HybridPool MHA shape={tuple(kv_cache_base.shape)} "
                f"stride={tuple(kv_cache_base.stride())} for required={required}"
            )
        kv_hnd = torch.as_strided(
            kv_cache_base,
            (pages, 2, kv_heads, page_size, head_dim),
            (
                int(kv_cache_base.stride(0)),
                kv_heads * page_size * head_dim,
                page_size * head_dim,
                head_dim,
                1,
            ),
        )
    else:
        kv_hnd = view_mha_kv_hnd(
            kv_cache_base, kv_heads=kv_heads, page_size=page_size, head_dim=head_dim
        )
    if kv_scale is None or kv_hnd.dtype != torch.float8_e4m3fn:
        return kv_hnd, None
    pages = int(kv_hnd.shape[0])
    if kv_scale.ndim == 1 and int(kv_scale.numel()) == 2:
        sc_hnd = torch.as_strided(
            kv_scale,
            (pages, 2, kv_heads, page_size),
            (0, int(kv_scale.stride(0)), 0, 0),
        )
        return kv_hnd, sc_hnd
    if kv_scale.ndim == 2:
        width = 2 * kv_heads * page_size
        if int(kv_scale.shape[0]) == pages and int(kv_scale.shape[1]) >= width and kv_scale.stride(1) == 1:
            sc_hnd = torch.as_strided(
                kv_scale,
                (pages, 2, kv_heads, page_size),
                (int(kv_scale.stride(0)), kv_heads * page_size, page_size, 1),
            )
            return kv_hnd, sc_hnd
    if tuple(kv_scale.shape) == (pages, 2, kv_heads, page_size):
        return kv_hnd, kv_scale
    if tuple(kv_scale.shape) == (pages, 2, page_size, kv_heads):
        return kv_hnd, kv_scale.transpose(2, 3)
    raise ValueError(f"Unsupported FP8 KV scale shape {tuple(kv_scale.shape)}")


def _try_run_q4_prefill(
    q_bhtd: torch.Tensor,
    k_ptr_obj: torch.Tensor,
    v_ptr_obj: torch.Tensor,
    sc_hnd: Optional[torch.Tensor],
    bt_kv: Optional[torch.Tensor],
    sel_c: torch.Tensor,
    *,
    T: int,
    page_size: int,
    is_paged: bool,
    max_sub_g: int = 1,
    assume_unique_selected: bool = False,
) -> Optional[torch.Tensor]:
    """Attempt Q4 block-dedup sparse prefill; returns None if fallback to Direct is faster/needed."""
    B, H_q, S, D = q_bhtd.shape
    H_kv = int(k_ptr_obj.shape[2] if is_paged else k_ptr_obj.shape[1])
    K = int(sel_c.shape[-1])
    q_per_kv = H_q // max(H_kv, 1)
    g_blk = min(8, max(1, q_per_kv))
    sub_g = triton.cdiv(max(1, q_per_kv), g_blk)
    # When sub_g > 1 (e.g. q_per_kv = 16 or 32), Direct BLK_H=16/32 loads K/V once per query for all Q heads,
    # whereas Q4 reloads deduplicated quads sub_g times across separate CTAs. Callers may override max_sub_g.
    if (
        not _HAS_TRITON
        or torch.cuda.is_current_stream_capturing()
        or S < 8
        or K < 32
        or T <= 0
        or sub_g > max_sub_g
        or (is_paged and (page_size % 4 != 0 or bt_kv is None))
    ):
        return None
    n_grp_s = triton.cdiv(S, 4)
    n_grp = B * n_grp_s
    n_quads = triton.cdiv(T, 4)
    n_words = triton.cdiv(n_quads, 2)
    cap_u = min(n_quads, max(triton.cdiv(K, 4) * 5, 1024))
    # Cap Q4 workspace at ~200 MB to keep peak memory bounded for very wide batches/contexts
    if (n_grp * (n_words + cap_u) * 4) > 200 * 1024 * 1024:
        return None

    dev = q_bhtd.device
    check_bad = not assume_unique_selected
    bmp = torch.zeros((n_grp, n_words), dtype=torch.int32, device=dev)
    u_packed = torch.empty((n_grp, cap_u), dtype=torch.int32, device=dev)
    u_cnt = torch.empty((n_grp,), dtype=torch.int32, device=dev)
    bad_grp = torch.zeros((n_grp,), dtype=torch.int32, device=dev) if check_bad else u_cnt
    bk = 512 if K >= 256 else max(16, triton.next_power_of_2(K))
    rt_pack = (sel_c, bmp, bad_grp, int(sel_c.stride(0)), int(sel_c.stride(1)), S, T, K, n_words, n_grp_s)
    ptr_pack = sel_c.data_ptr() | bmp.data_ptr() | bad_grp.data_ptr()
    ck_pack = (dev.index, 206, sel_c.dtype, bmp.dtype, bad_grp.dtype, bk, check_bad)
    triton_fast_launch(
        _QSA_TRITON_CACHE, ck_pack, _qsa_q4_pack_quads_kernel,
        n_grp, 1, 1, ptr_pack, rt_pack,
        {"BK": bk, "CHECK_DUP": check_bad},
        num_warps=4,
    )
    rt_cmp = (bmp, u_packed, u_cnt, bad_grp, n_words, cap_u)
    ptr_cmp = bmp.data_ptr() | u_packed.data_ptr() | u_cnt.data_ptr() | bad_grp.data_ptr()
    ck_cmp = (dev.index, 207, bmp.dtype, u_packed.dtype, u_cnt.dtype, bad_grp.dtype, 1024, check_bad)
    triton_fast_launch(
        _QSA_TRITON_CACHE, ck_cmp, _qsa_q4_compact_quads_kernel,
        n_grp, 1, 1, ptr_cmp, rt_cmp,
        {"BW": 1024, "CHECK_OVERFLOW": check_bad},
        num_warps=8,
    )

    out = torch.empty_like(q_bhtd)
    m_rows = 16 if g_blk <= 4 else 32
    is_fp8 = k_ptr_obj.dtype == torch.float8_e4m3fn
    has_scale = is_fp8 and sc_hnd is not None
    dummy_bt = bt_kv if is_paged and bt_kv is not None else sel_c
    dummy_sc = sc_hnd if has_scale and sc_hnd is not None else k_ptr_obj
    max_pages = int(bt_kv.shape[1]) if is_paged and bt_kv is not None else 1
    num_pages = int(k_ptr_obj.shape[0]) if is_paged else 1
    page_sz_eff = max(4, page_size) if is_paged else 4
    ns_q4 = 1 if is_fp8 else 2
    rt_q4 = (
        q_bhtd, k_ptr_obj, v_ptr_obj, dummy_sc, dummy_bt, u_packed, u_cnt, bad_grp, out,
        int(q_bhtd.stride(0)), int(q_bhtd.stride(1)), int(q_bhtd.stride(2)),
        int(k_ptr_obj.stride(0)),
        int(k_ptr_obj.stride(1)) if is_paged else 0,
        int(k_ptr_obj.stride(2)) if is_paged else int(k_ptr_obj.stride(1)),
        int(k_ptr_obj.stride(3)) if is_paged else int(k_ptr_obj.stride(2)),
        int(v_ptr_obj.stride(0)), int(v_ptr_obj.stride(1)), int(v_ptr_obj.stride(2)),
        int(sc_hnd.stride(0)) if has_scale and sc_hnd is not None else 0,
        int(sc_hnd.stride(1)) if has_scale and sc_hnd is not None else 0,
        int(sc_hnd.stride(2)) if has_scale and sc_hnd is not None else 0,
        int(sc_hnd.stride(3)) if has_scale and sc_hnd is not None else 0,
        int(bt_kv.stride(0)) if is_paged and bt_kv is not None else 0,
        int(bt_kv.stride(1)) if is_paged and bt_kv is not None else 0,
        int(out.stride(0)), int(out.stride(1)), int(out.stride(2)),
        S, T, max_pages, num_pages, cap_u, n_grp_s,
    )
    ptr_q4 = (
        q_bhtd.data_ptr() | k_ptr_obj.data_ptr() | v_ptr_obj.data_ptr()
        | dummy_sc.data_ptr() | dummy_bt.data_ptr() | u_packed.data_ptr()
        | u_cnt.data_ptr() | bad_grp.data_ptr() | out.data_ptr()
    )
    q4_stride_specs = tuple(
        _triton_int_spec(rt_q4[idx_s]) for idx_s in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26, 27)
    )
    ck_q4 = (
        dev.index, 208, q_bhtd.dtype, k_ptr_obj.dtype, v_ptr_obj.dtype, dummy_sc.dtype,
        dummy_bt.dtype, u_packed.dtype, u_cnt.dtype, bad_grp.dtype, out.dtype,
        q4_stride_specs, D, float(D**-0.5), H_kv, q_per_kv, m_rows, page_sz_eff,
        16, is_paged, is_fp8, has_scale, g_blk, sub_g, check_bad, ns_q4,
    )
    triton_fast_launch(
        _QSA_TRITON_CACHE, ck_q4, _qsa_q4_gqa_kernel,
        n_grp_s, B, H_kv * sub_g, ptr_q4, rt_q4,
        {
            "D": D, "SCALE": D**-0.5, "HK": H_kv, "G": q_per_kv,
            "M_ROWS": m_rows, "PAGE_SIZE": page_sz_eff, "BK_Q": 16,
            "IS_PAGED": is_paged, "IS_FP8": is_fp8, "HAS_SCALE": has_scale,
            "G_BLK": g_blk, "SUB_G": sub_g, "CHECK_BAD": check_bad,
        },
        num_warps=4, num_stages=ns_q4,
    )
    if check_bad:
        _launch_direct_sparse_prefill(
            q_bhtd, k_ptr_obj, v_ptr_obj, sc_hnd, bt_kv, sel_c, out,
            T=T, page_size=page_size if is_paged else 1, is_paged=is_paged,
            bad_ptr=bad_grp, bad_shift=2, n_bad_s=n_grp_s,
        )
    return out


def _launch_direct_sparse_prefill(
    q_c: torch.Tensor,
    k_ptr_obj: torch.Tensor,
    v_ptr_obj: torch.Tensor,
    sc_hnd: Optional[torch.Tensor],
    bt_kv: Optional[torch.Tensor],
    sel_c: torch.Tensor,
    out: torch.Tensor,
    *,
    T: int,
    page_size: int,
    is_paged: bool,
    bad_ptr: Optional[torch.Tensor] = None,
    bad_shift: int = 2,
    n_bad_s: int = 1,
) -> None:
    """Launch dense or paged Direct sparse prefill kernel (full or bad-group/row GPU-only repair)."""
    B, H_q, S, D = q_c.shape
    H_kv = int(k_ptr_obj.shape[2] if is_paged else k_ptr_obj.shape[1])
    K = int(sel_c.shape[-1])
    q_per_kv = H_q // H_kv
    blk_h = 32 if (16 < q_per_kv <= 32) else 16
    sub_g = triton.cdiv(q_per_kv, blk_h)
    blk_k = (
        128
        if (K >= 128 and D == 256 and blk_h >= 16)
        else (64 if (blk_h == 32 or D < 256) else 32)
    )
    is_fp8 = is_paged and (k_ptr_obj.dtype == torch.float8_e4m3fn)
    has_scale = is_fp8 and sc_hnd is not None
    dummy_sc = sc_hnd if has_scale and sc_hnd is not None else k_ptr_obj
    dummy_bt = bt_kv if is_paged and bt_kv is not None else sel_c
    only_bad = bad_ptr is not None
    bad_arg = bad_ptr if only_bad else out
    max_pages = int(bt_kv.shape[1]) if is_paged and bt_kv is not None else 1
    num_pages = int(k_ptr_obj.shape[0]) if is_paged else 1
    max_blks_k = triton.cdiv(K, blk_k)
    rt_dir = (
        q_c, k_ptr_obj, v_ptr_obj, dummy_sc, dummy_bt, sel_c, bad_arg, out,
        int(q_c.stride(0)), int(q_c.stride(1)), int(q_c.stride(2)),
        int(out.stride(0)), int(out.stride(1)), int(out.stride(2)),
        int(k_ptr_obj.stride(0)),
        int(k_ptr_obj.stride(1)) if is_paged else 0,
        int(k_ptr_obj.stride(2)) if is_paged else int(k_ptr_obj.stride(1)),
        int(k_ptr_obj.stride(3)) if is_paged else int(k_ptr_obj.stride(2)),
        0 if is_paged else int(v_ptr_obj.stride(0)),
        0 if is_paged else int(v_ptr_obj.stride(1)),
        0 if is_paged else int(v_ptr_obj.stride(2)),
        int(sc_hnd.stride(0)) if has_scale and sc_hnd is not None else 0,
        int(sc_hnd.stride(1)) if has_scale and sc_hnd is not None else 0,
        int(sc_hnd.stride(2)) if has_scale and sc_hnd is not None else 0,
        int(sc_hnd.stride(3)) if has_scale and sc_hnd is not None else 0,
        int(bt_kv.stride(0)) if is_paged and bt_kv is not None else 0,
        int(bt_kv.stride(1)) if is_paged and bt_kv is not None else 0,
        int(sel_c.stride(0)), int(sel_c.stride(1)),
        K, T, max_pages, num_pages, max_blks_k, n_bad_s,
    )
    ptr_dir = (
        q_c.data_ptr() | k_ptr_obj.data_ptr() | v_ptr_obj.data_ptr() | dummy_sc.data_ptr()
        | dummy_bt.data_ptr() | sel_c.data_ptr() | bad_arg.data_ptr() | out.data_ptr()
    )
    data_stride_specs = tuple(_triton_int_spec(v_s) for v_s in rt_dir[8:21])
    if D == 256 and blk_h >= 16 and q_per_kv >= 4 and blk_k >= 64:
        ns_ds = 2 if (is_fp8 and blk_k == 128 and blk_h == 32) else 1
        ck_ds = (
            q_c.device.index, 202, q_c.dtype, k_ptr_obj.dtype, v_ptr_obj.dtype,
            dummy_sc.dtype, dummy_bt.dtype, sel_c.dtype, bad_arg.dtype, out.dtype,
            data_stride_specs, float(D**-0.5), q_per_kv, page_size, blk_k, blk_h,
            is_paged, is_fp8, has_scale, sub_g, only_bad, bad_shift if only_bad else 2, ns_ds,
        )
        triton_fast_launch(
            _QSA_TRITON_CACHE, ck_ds, _qsa_sparse_prefill_dsplit_kernel,
            S, B, H_kv * sub_g, ptr_dir, rt_dir,
            {
                "SCALE": D**-0.5, "q_per_kv": q_per_kv, "PAGE_SIZE": page_size,
                "BLK_K": blk_k, "BLK_H": blk_h, "IS_PAGED": is_paged,
                "IS_FP8": is_fp8, "HAS_SCALE": has_scale, "FP8_QK": is_fp8,
                "SUB_G": sub_g, "ONLY_BAD": only_bad, "BAD_SHIFT": bad_shift if only_bad else 2,
            },
            num_warps=4, num_stages=ns_ds,
        )
    else:
        ns_gf = 1 if (is_fp8 or blk_h == 32) else 2
        ck_gf = (
            q_c.device.index, 209, q_c.dtype, k_ptr_obj.dtype, v_ptr_obj.dtype,
            dummy_sc.dtype, dummy_bt.dtype, sel_c.dtype, bad_arg.dtype, out.dtype,
            data_stride_specs, D, float(D**-0.5), q_per_kv, page_size, blk_k, blk_h,
            is_paged, is_fp8, has_scale, sub_g, only_bad, bad_shift if only_bad else 2, ns_gf,
        )
        triton_fast_launch(
            _QSA_TRITON_CACHE, ck_gf, _qsa_sparse_prefill_grouped_fast_kernel,
            S, B, H_kv * sub_g, ptr_dir, rt_dir,
            {
                "D": D, "SCALE": D**-0.5, "q_per_kv": q_per_kv, "PAGE_SIZE": page_size,
                "BLK_K": blk_k, "BLK_H": blk_h, "IS_PAGED": is_paged,
                "IS_FP8": is_fp8, "HAS_SCALE": has_scale, "SUB_G": sub_g,
                "ONLY_BAD": only_bad, "BAD_SHIFT": bad_shift if only_bad else 2,
            },
            num_warps=4, num_stages=ns_gf,
        )


def qsa_sparse_prefill_attention(
    q_bhtd: torch.Tensor,
    k_bhtd: torch.Tensor,
    v_bhtd: torch.Tensor,
    selected: torch.Tensor,
    *,
    validate_selected: bool = False,
    assume_unique_selected: bool = False,
) -> torch.Tensor:
    """Sparse prefill attention ``[B, H_q, S, D]`` over selected token indices."""
    if q_bhtd.is_cuda:
        B, H_q, S, D = q_bhtd.shape
        H_kv = int(k_bhtd.shape[1])
        T = int(k_bhtd.shape[2])
        K = int(selected.shape[-1])
        if B == 0 or S == 0:
            return torch.empty_like(q_bhtd)
        q_per_kv = H_q // max(H_kv, 1)
        can_triton_fast = (
            _HAS_TRITON
            and torch.version.hip is None
            and q_bhtd.dtype == torch.bfloat16
            and k_bhtd.dtype == torch.bfloat16
            and v_bhtd.dtype == torch.bfloat16
            and selected.dtype == torch.int32
            and k_bhtd.shape == v_bhtd.shape
            and int(k_bhtd.shape[0]) == B
            and int(k_bhtd.shape[3]) == D
            and tuple(selected.shape[:2]) == (B, S)
            and H_kv > 0
            and H_q % H_kv == 0
            and D in (64, 128, 256)
            and torch.cuda.get_device_capability(q_bhtd.device)[0] >= 8
        )
        if can_triton_fast:
            if validate_selected and selected.numel() > 0:
                max_selected = int(selected.max().item())
                if max_selected >= T:
                    raise ValueError(
                        f"selected index {max_selected} is outside KV sequence length {T}"
                    )
            if T == 0 or K == 0:
                return torch.zeros_like(q_bhtd)
            q_c = q_bhtd.contiguous()
            k_c = k_bhtd.contiguous()
            v_c = v_bhtd.contiguous()
            sel_c = selected.contiguous()

            # 1) Dense medium-context (T <= 2048 and <= 2*K): packed 1-bit bitmap multi-query path
            # For D==256 and q_per_kv>=4 (e.g. TP4 Hq=8), Direct dsplit beats bitmap on B*S < 2048 or T < 512 with 0 syncs.
            words = triton.cdiv(T, 32)
            if (
                not torch.cuda.is_current_stream_capturing()
                and S >= 128
                and 64 <= T <= min(2048, 2 * K)
                and q_per_kv <= 16
                and (q_per_kv < 4 or D != 256 or (B * S >= 2048 and T >= 512))
                and (B * S * words) <= 32 * 1024 * 1024
            ):
                qs32 = 32 // q_per_kv
                qs16 = max(1, 16 // q_per_kv)
                if (5 <= q_per_kv <= 7) or (qs32 > qs16 and q_per_kv <= 12):
                    m_rows, qs = 32, qs32
                else:
                    m_rows, qs = 16, qs16
                kt = 64 if T >= 64 else 32
                check_bad_bm = not assume_unique_selected
                packed = torch.zeros((B, S, words), device=q_c.device, dtype=torch.int32)
                ends = torch.empty((B, S), device=q_c.device, dtype=torch.int32)
                bad_bm = torch.zeros((B * S,), device=q_c.device, dtype=torch.int32) if check_bad_bm else ends
                bk = 512 if K >= 256 else max(16, triton.next_power_of_2(K))
                rt_bp = (sel_c, packed, ends, bad_bm, T, K, words)
                ptr_bp = sel_c.data_ptr() | packed.data_ptr() | ends.data_ptr() | bad_bm.data_ptr()
                ck_bp = (q_c.device.index, 210, sel_c.dtype, packed.dtype, ends.dtype, bad_bm.dtype, bk, check_bad_bm)
                triton_fast_launch(
                    _QSA_TRITON_CACHE, ck_bp, _qsa_pack_bitmap_words_kernel,
                    B * S, 1, 1, ptr_bp, rt_bp,
                    {"BK": bk, "CHECK_DUP": check_bad_bm},
                    num_warps=4,
                )
                out = torch.empty_like(q_c)
                rt_bg = (q_c, k_c, v_c, packed, ends, bad_bm, out, S, T, words)
                ptr_bg = (
                    q_c.data_ptr() | k_c.data_ptr() | v_c.data_ptr()
                    | packed.data_ptr() | ends.data_ptr() | bad_bm.data_ptr() | out.data_ptr()
                )
                ck_bg = (
                    q_c.device.index, 211, q_c.dtype, k_c.dtype, v_c.dtype, packed.dtype, ends.dtype,
                    bad_bm.dtype, out.dtype, H_q, H_kv, D, float(D**-0.5), q_per_kv, qs, m_rows, kt, False, check_bad_bm,
                )
                triton_fast_launch(
                    _QSA_TRITON_CACHE, ck_bg, _qsa_bitmap_packed_gqa_kernel,
                    B, H_kv, triton.cdiv(S, qs), ptr_bg, rt_bg,
                    {
                        "HQ": H_q, "HK": H_kv, "D": D, "SCALE": D**-0.5, "G": q_per_kv,
                        "QS": qs, "M_ROWS": m_rows, "KT": kt, "CHECK_EMPTY": False, "CHECK_BAD": check_bad_bm,
                    },
                    num_warps=4, num_stages=1,
                )
                if check_bad_bm:
                    _launch_direct_sparse_prefill(
                        q_c, k_c, v_c, None, None, sel_c, out,
                        T=T, page_size=1, is_paged=False,
                        bad_ptr=bad_bm, bad_shift=0, n_bad_s=S,
                    )
                return out

            # 2) Sparse long-context Q4 4-query quad-dedup TensorCore kernel
            # For D==256 and q_per_kv>=4, Direct dsplit beats Q4 on small work (B*S < 1024 or T <= 2048 or K < 256).
            if D != 256 or q_per_kv < 4 or (B * S >= 1024 and T > 2048 and K >= 256):
                q4_out = _try_run_q4_prefill(
                    q_c,
                    k_c,
                    v_c,
                    None,
                    None,
                    sel_c,
                    T=T,
                    page_size=4,
                    is_paged=False,
                    assume_unique_selected=assume_unique_selected,
                )
                if q4_out is not None:
                    return q4_out

            # 3) Fast direct grouped GQA TensorCore path (BLK_H=32 for q_per_kv=32, else BLK_H=16)
            if S >= 16 and K >= 32 and q_per_kv >= 3:
                out = torch.empty_like(q_c)
                _launch_direct_sparse_prefill(
                    q_c, k_c, v_c, None, None, sel_c, out,
                    T=T, page_size=1, is_paged=False,
                )
                return out

        from rtp_llm.models_py.kernels.cuda.qsa.sparse_fmha import (
            sparse_prefill_attn,
        )

        return sparse_prefill_attn(
            q_bhtd.contiguous(),
            k_bhtd.contiguous(),
            v_bhtd.contiguous(),
            selected.contiguous(),
            validate_selected=validate_selected,
        )

    # Exact PyTorch reference fallback (CPU / non-CUDA tests)
    bsz, h_q, seq_len, head_dim = q_bhtd.shape
    h_kv = int(k_bhtd.shape[1])
    total_kv = int(k_bhtd.shape[2])
    q_per_kv = h_q // h_kv
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.zeros_like(q_bhtd)
    k_f = k_bhtd.float()
    v_f = v_bhtd.float()
    q_f = q_bhtd.float()
    for b in range(bsz):
        for s in range(seq_len):
            sel_row = selected[b, s]
            valid_mask = (sel_row >= 0) & (sel_row < total_kv)
            if not bool(valid_mask.any()):
                continue
            safe_idx = sel_row.clamp(0, max(total_kv - 1, 0)).long()
            k_sel = k_f[b, :, safe_idx, :]  # [H_kv, K, D]
            v_sel = v_f[b, :, safe_idx, :]  # [H_kv, K, D]
            if q_per_kv > 1:
                k_sel = k_sel.repeat_interleave(q_per_kv, dim=0)
                v_sel = v_sel.repeat_interleave(q_per_kv, dim=0)
            scores = torch.einsum("hd,hkd->hk", q_f[b, :, s, :], k_sel) * scale
            scores = scores.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            probs = probs.masked_fill(~valid_mask.unsqueeze(0), 0.0)
            out[b, :, s, :] = torch.einsum("hk,hkd->hd", probs, v_sel).to(q_bhtd.dtype)
    return out


def qsa_sparse_paged_prefill_attention(
    q_bhtd: torch.Tensor,
    kv_cache_base: torch.Tensor,
    kv_block_table: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    kv_heads: int,
    head_dim: int,
    max_ctx_len: int,
    kv_scale: Optional[torch.Tensor] = None,
    assume_unique_selected: bool = False,
) -> torch.Tensor:
    """Zero-copy sparse prefill directly against paged MHA KV pool (Q4 dedup + GQA TC fallback)."""
    B, H_q, S, D = q_bhtd.shape
    if B == 0 or S == 0:
        return torch.empty_like(q_bhtd)
    K = int(selected.shape[-1])
    if max_ctx_len <= 0 or K == 0:
        return torch.zeros_like(q_bhtd)

    q_per_kv = H_q // max(kv_heads, 1)
    kv_hnd, sc_hnd = _view_mha_kv_and_scale_hnd(
        kv_cache_base,
        kv_scale,
        kv_heads=kv_heads,
        page_size=page_size,
        head_dim=head_dim,
    )
    if (
        _HAS_TRITON
        and q_bhtd.is_cuda
        and torch.version.hip is None
        and q_bhtd.dtype == torch.bfloat16
        and kv_hnd.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and selected.dtype == torch.int32
        and kv_block_table.dtype == torch.int32
        and D in (64, 128, 256)
        and kv_heads > 0
        and H_q % kv_heads == 0
        and (q_per_kv <= 16 or K >= 32)
        and page_size > 0
        and (page_size & (page_size - 1)) == 0
        and torch.cuda.get_device_capability(q_bhtd.device)[0] >= 8
    ):
        q_c = q_bhtd if q_bhtd.stride(-1) == 1 else q_bhtd.contiguous()
        sel_c = selected if selected.stride(-1) == 1 else selected.contiguous()
        if D != 256 or q_per_kv < 4 or (B * S >= 1024 and max_ctx_len > 2048 and K >= 256):
            q4_out = _try_run_q4_prefill(
                q_c,
                kv_hnd,
                kv_hnd,
                sc_hnd,
                kv_block_table,
                sel_c,
                T=max_ctx_len,
                page_size=page_size,
                is_paged=True,
                assume_unique_selected=assume_unique_selected,
            )
            if q4_out is not None:
                return q4_out

        out = torch.empty_like(q_c)
        _launch_direct_sparse_prefill(
            q_c, kv_hnd, kv_hnd, sc_hnd, kv_block_table, sel_c, out,
            T=max_ctx_len, page_size=page_size, is_paged=True,
        )
        return out

    from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import (
        sparse_paged_gqa_attn,
    )

    kv_lens_2d = torch.full((B, S), max_ctx_len, dtype=torch.int32, device=q_bhtd.device)
    return sparse_paged_gqa_attn(
        q_bhtd,
        kv_hnd,
        kv_block_table,
        kv_lens_2d,
        selected,
        page_size=page_size,
        kv_head_num=kv_heads,
        kv_scale=sc_hnd,
        validate_selected=False,
    )


def qsa_sparse_paged_decode_attention(
    q_bhsd: torch.Tensor,
    kv_cache_base: torch.Tensor,
    kv_block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    kv_heads: int,
    head_dim: int,
    kv_scale: Optional[torch.Tensor] = None,
    output_gate: Optional[torch.Tensor] = None,
    is_cuda_graph: bool = False,
) -> torch.Tensor:
    """Sparse paged GQA/MQA attention ``[B, H_q, S, D]`` using fast GQA TensorCore Split-KV."""
    B, H_q, S, D = q_bhsd.shape
    if B == 0 or S == 0:
        if output_gate is not None:
            return torch.empty((B, S, H_q, D), device=q_bhsd.device, dtype=q_bhsd.dtype).transpose(
                1, 2
            )
        return torch.empty_like(q_bhsd)

    K = int(selected.shape[-1])
    kv_hnd, sc_hnd = _view_mha_kv_and_scale_hnd(
        kv_cache_base,
        kv_scale,
        kv_heads=kv_heads,
        page_size=page_size,
        head_dim=head_dim,
    )
    q_per_kv = H_q // max(kv_heads, 1)

    if output_gate is not None and not (
        output_gate.shape == (B, S, H_q, D)
        and output_gate.dtype == q_bhsd.dtype
        and output_gate.device == q_bhsd.device
        and output_gate.is_contiguous()
    ):
        raise ValueError("QSA output gate must be contiguous [B,S,H,D] on the query device")

    if (
        _HAS_TRITON
        and q_bhsd.is_cuda
        and torch.version.hip is None
        and q_bhsd.dtype == torch.bfloat16
        and kv_hnd.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and kv_block_table.dtype == torch.int32
        and kv_lens.dtype == torch.int32
        and selected.dtype == torch.int32
        and kv_heads > 0
        and H_q % kv_heads == 0
        and q_per_kv >= 1
        and D in (64, 128, 256)
        and page_size > 0
        and (page_size & (page_size - 1)) == 0
        and int(kv_hnd.stride(-1)) == 1
        and torch.cuda.get_device_capability(q_bhsd.device)[0] >= 8
    ):
        if K == 0:
            if output_gate is not None:
                return torch.zeros(
                    (B, S, H_q, D), device=q_bhsd.device, dtype=q_bhsd.dtype
                ).transpose(1, 2)
            return torch.zeros_like(q_bhsd)

        q_c = q_bhsd if q_bhsd.stride(-1) == 1 else q_bhsd.contiguous()
        sel_c = selected if selected.stride(-1) == 1 else selected.contiguous()
        kv_lens_2d = kv_lens.unsqueeze(1) if kv_lens.ndim == 1 else kv_lens
        is_fp8 = kv_hnd.dtype == torch.float8_e4m3fn
        has_scale = is_fp8 and sc_hnd is not None

        m_rows = 32 if (16 < q_per_kv <= 32) else 16
        sub_g = triton.cdiv(q_per_kv, m_rows)
        m_ctas = B * S * kv_heads * sub_g
        if not is_fp8:
            if K >= 256 and D == 256:
                if m_rows == 32:
                    if m_ctas <= 16:
                        target_sp, kt, nw, stg = 16, 64, 4, 1
                    elif m_ctas <= 128:
                        target_sp, kt, nw, stg = (4 if S == 1 else 2), 64, 4, 1
                    else:
                        target_sp, kt, nw, stg = 4, 32, 4, 2
                else:
                    kt, nw, stg = 64, 2, 1
                    if m_ctas <= 16:
                        target_sp = 16
                    elif m_ctas <= 64:
                        target_sp = 8
                    elif m_ctas <= 128:
                        target_sp = 4
                    elif m_ctas <= 512:
                        target_sp = 1
                    else:
                        target_sp = 4
            elif m_ctas <= 16:
                target_sp, kt, nw, stg = 32, 32, 4, 2
            elif m_ctas <= 128:
                target_sp, kt, nw, stg = 4, (64 if K >= 256 else 32), 4, 2
            else:
                target_sp, kt, nw, stg = 4, 32, 4, 2
        else:
            if K >= 512 and D == 256:
                kt, nw = 128, 4
                if m_ctas <= 16:
                    target_sp, stg = 16, 1
                elif m_ctas <= 64:
                    target_sp, stg = 4, 1
                elif m_ctas <= 128:
                    target_sp, stg = 2, 1
                elif m_ctas <= 512:
                    target_sp, stg = (1 if m_rows == 32 else 4), 1
                else:
                    target_sp, stg = 2, 2
            elif m_ctas <= 16:
                target_sp, kt, nw, stg = 16, 32, 4, 2
            elif m_ctas <= 128:
                if S > 1:
                    target_sp, kt, nw, stg = 16, 32, 4, 1
                else:
                    target_sp, kt, nw, stg = 4, (64 if K >= 256 else 32), 4, 2
            else:
                target_sp, kt, nw, stg = 4, 32, 4, 2

        splits = min(target_sp, triton.next_power_of_2(triton.cdiv(K, kt)))
        with_gate = output_gate is not None
        if with_gate:
            out = torch.empty(
                (B, S, H_q, D), device=q_c.device, dtype=q_c.dtype
            ).transpose(1, 2)
        else:
            out = torch.empty_like(q_c)

        dummy_sc = sc_hnd if has_scale and sc_hnd is not None else kv_hnd
        dummy_gate = output_gate if with_gate and output_gate is not None else out

        use_dsplit = D == 256 and m_rows == 32 and kt >= 64
        if splits == 1:
            if use_dsplit:
                _qsa_gqa_paged_split_tc_dsplit_kernel[(m_ctas, 1)](
                    q_c,
                    kv_hnd,
                    dummy_sc,
                    kv_block_table,
                    kv_lens_2d,
                    sel_c,
                    out,
                    out,
                    out,
                    dummy_gate,
                    int(q_c.stride(0)),
                    int(q_c.stride(1)),
                    int(q_c.stride(2)),
                    int(kv_hnd.stride(0)),
                    int(kv_hnd.stride(1)),
                    int(kv_hnd.stride(2)),
                    int(kv_hnd.stride(3)),
                    int(sc_hnd.stride(0)) if has_scale and sc_hnd is not None else 0,
                    int(sc_hnd.stride(1)) if has_scale and sc_hnd is not None else 0,
                    int(sc_hnd.stride(2)) if has_scale and sc_hnd is not None else 0,
                    int(sc_hnd.stride(3)) if has_scale and sc_hnd is not None else 0,
                    int(kv_block_table.stride(0)),
                    int(kv_block_table.stride(1)),
                    int(kv_lens_2d.stride(0)),
                    int(kv_lens_2d.stride(1)),
                    int(sel_c.stride(0)),
                    int(sel_c.stride(1)),
                    int(out.stride(0)),
                    int(out.stride(1)),
                    int(out.stride(2)),
                    S,
                    K,
                    int(kv_block_table.shape[1]),
                    int(kv_hnd.shape[0]),
                    HK=kv_heads,
                    G=q_per_kv,
                    M_ROWS=m_rows,
                    PAGE_SIZE=page_size,
                    SCALE=D**-0.5,
                    KT=kt,
                    NUM_SPLITS=1,
                    WITH_GATE=with_gate,
                    IS_FP8=is_fp8,
                    HAS_SCALE=has_scale,
                    FP8_QK=is_fp8,
                    SUB_G=sub_g,
                    num_warps=nw,
                    num_stages=stg,
                )
            else:
                _qsa_gqa_paged_split_tc_kernel[(m_ctas, 1)](
                    q_c,
                    kv_hnd,
                    dummy_sc,
                    kv_block_table,
                    kv_lens_2d,
                    sel_c,
                    out,
                    out,
                    out,
                    dummy_gate,
                    int(q_c.stride(0)),
                    int(q_c.stride(1)),
                    int(q_c.stride(2)),
                    int(kv_hnd.stride(0)),
                    int(kv_hnd.stride(1)),
                    int(kv_hnd.stride(2)),
                    int(kv_hnd.stride(3)),
                    int(sc_hnd.stride(0)) if has_scale and sc_hnd is not None else 0,
                    int(sc_hnd.stride(1)) if has_scale and sc_hnd is not None else 0,
                    int(sc_hnd.stride(2)) if has_scale and sc_hnd is not None else 0,
                    int(sc_hnd.stride(3)) if has_scale and sc_hnd is not None else 0,
                    int(kv_block_table.stride(0)),
                    int(kv_block_table.stride(1)),
                    int(kv_lens_2d.stride(0)),
                    int(kv_lens_2d.stride(1)),
                    int(sel_c.stride(0)),
                    int(sel_c.stride(1)),
                    int(out.stride(0)),
                    int(out.stride(1)),
                    int(out.stride(2)),
                    S,
                    K,
                    int(kv_block_table.shape[1]),
                    int(kv_hnd.shape[0]),
                    HK=kv_heads,
                    G=q_per_kv,
                    M_ROWS=m_rows,
                    PAGE_SIZE=page_size,
                    D=D,
                    SCALE=D**-0.5,
                    KT=kt,
                    NUM_SPLITS=1,
                    WITH_GATE=with_gate,
                    IS_FP8=is_fp8,
                    HAS_SCALE=has_scale,
                    SUB_G=sub_g,
                    num_warps=nw,
                    num_stages=stg,
                )
            return out

        p_out = torch.empty((B, S, H_q, splits, D), dtype=torch.float32, device=q_c.device)
        p_m = torch.empty((B, S, H_q, splits), dtype=torch.float32, device=q_c.device)
        p_z = torch.empty((B, S, H_q, splits), dtype=torch.float32, device=q_c.device)
        if use_dsplit:
            _qsa_gqa_paged_split_tc_dsplit_kernel[(m_ctas, splits)](
                q_c,
                kv_hnd,
                dummy_sc,
                kv_block_table,
                kv_lens_2d,
                sel_c,
                p_out,
                p_m,
                p_z,
                dummy_gate,
                int(q_c.stride(0)),
                int(q_c.stride(1)),
                int(q_c.stride(2)),
                int(kv_hnd.stride(0)),
                int(kv_hnd.stride(1)),
                int(kv_hnd.stride(2)),
                int(kv_hnd.stride(3)),
                int(sc_hnd.stride(0)) if has_scale and sc_hnd is not None else 0,
                int(sc_hnd.stride(1)) if has_scale and sc_hnd is not None else 0,
                int(sc_hnd.stride(2)) if has_scale and sc_hnd is not None else 0,
                int(sc_hnd.stride(3)) if has_scale and sc_hnd is not None else 0,
                int(kv_block_table.stride(0)),
                int(kv_block_table.stride(1)),
                int(kv_lens_2d.stride(0)),
                int(kv_lens_2d.stride(1)),
                int(sel_c.stride(0)),
                int(sel_c.stride(1)),
                int(out.stride(0)),
                int(out.stride(1)),
                int(out.stride(2)),
                S,
                K,
                int(kv_block_table.shape[1]),
                int(kv_hnd.shape[0]),
                HK=kv_heads,
                G=q_per_kv,
                M_ROWS=m_rows,
                PAGE_SIZE=page_size,
                SCALE=D**-0.5,
                KT=kt,
                NUM_SPLITS=splits,
                WITH_GATE=False,
                IS_FP8=is_fp8,
                HAS_SCALE=has_scale,
                FP8_QK=is_fp8,
                SUB_G=sub_g,
                num_warps=nw,
                num_stages=stg,
            )
        else:
            _qsa_gqa_paged_split_tc_kernel[(m_ctas, splits)](
                q_c,
                kv_hnd,
                dummy_sc,
                kv_block_table,
                kv_lens_2d,
                sel_c,
                p_out,
                p_m,
                p_z,
                dummy_gate,
                int(q_c.stride(0)),
                int(q_c.stride(1)),
                int(q_c.stride(2)),
                int(kv_hnd.stride(0)),
                int(kv_hnd.stride(1)),
                int(kv_hnd.stride(2)),
                int(kv_hnd.stride(3)),
                int(sc_hnd.stride(0)) if has_scale and sc_hnd is not None else 0,
                int(sc_hnd.stride(1)) if has_scale and sc_hnd is not None else 0,
                int(sc_hnd.stride(2)) if has_scale and sc_hnd is not None else 0,
                int(sc_hnd.stride(3)) if has_scale and sc_hnd is not None else 0,
                int(kv_block_table.stride(0)),
                int(kv_block_table.stride(1)),
                int(kv_lens_2d.stride(0)),
                int(kv_lens_2d.stride(1)),
                int(sel_c.stride(0)),
                int(sel_c.stride(1)),
                int(out.stride(0)),
                int(out.stride(1)),
                int(out.stride(2)),
                S,
                K,
                int(kv_block_table.shape[1]),
                int(kv_hnd.shape[0]),
                HK=kv_heads,
                G=q_per_kv,
                M_ROWS=m_rows,
                PAGE_SIZE=page_size,
                D=D,
                SCALE=D**-0.5,
                KT=kt,
                NUM_SPLITS=splits,
                WITH_GATE=False,
                IS_FP8=is_fp8,
                HAS_SCALE=has_scale,
                SUB_G=sub_g,
                num_warps=nw,
                num_stages=stg,
            )
        sp_pad = max(1, triton.next_power_of_2(splits))
        _qsa_gqa_paged_merge_kernel[(B * S * H_q,)](
            p_out,
            p_m,
            p_z,
            out,
            dummy_gate,
            int(out.stride(0)),
            int(out.stride(1)),
            int(out.stride(2)),
            S,
            HQ=H_q,
            D=D,
            NUM_SPLITS=splits,
            SP_PAD=sp_pad,
            WITH_GATE=with_gate,
            num_warps=4,
        )
        return out

    from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import (
        sparse_paged_gqa_attn,
    )

    in_capture = bool(
        is_cuda_graph
        and (
            torch.cuda.is_current_stream_capturing()
            or os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1"
        )
    )
    return sparse_paged_gqa_attn(
        q_bhsd,
        kv_hnd,
        kv_block_table,
        kv_lens,
        selected,
        page_size=page_size,
        kv_head_num=kv_heads,
        kv_scale=sc_hnd,
        graph_capture=in_capture,
        output_gate=output_gate,
        selected_has_count=False,
        validate_selected=False,
    )


__all__ = [
    "build_qsa_neox_cos_sin",
    "qsa_paged_score",
    "qsa_prefill_score",
    "qsa_select_causal_from_projected",
    "qsa_sparse_paged_decode_attention",
    "qsa_sparse_paged_prefill_attention",
    "qsa_sparse_prefill_attention",
    "score_and_select_qsa_paged",
    "view_indexer_kv_pool",
    "view_indexer_state_pool",
    "view_mha_kv_hnd",
    "write_qsa_decode_cache_gpu_",
    "write_qsa_verify_cache_gpu_",
    "write_qsa_prefill_cache",
]

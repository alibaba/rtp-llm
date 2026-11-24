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

    _HAS_TRITON = True
except Exception:  # pragma: no cover - pure-Python / no-triton fallback
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

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
    inv_freq = float(rope_theta) ** (
        -2.0 * torch.arange(half, dtype=torch.float32, device=device) / rotary_dim
    )
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
        bm = max(64, 4 * n_heads)
        bn = 64
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
            num_warps=4,
            num_stages=1,
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
        grid = (m_rows, triton.cdiv(max_ctx_len, _PAGED_SCORE_BN))
        _qsa_paged_score_kernel[grid](
            q_c,
            pool_flat,
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
    """Zero-sync token index expansion wrapping ``select_qsa_paged_tokens``."""
    return select_qsa_paged_tokens(
        block_scores,
        visible_lengths.to(dtype=torch.int32).reshape(-1),
        compress_ratio=compress_ratio,
        token_budget=token_budget,
        validate_lengths=False,
        scores_are_masked=scores_are_masked,
    )


def qsa_select_causal_from_projected(
    indexer: Qwen4ExpQSAIndexer,
    q_unrot: torch.Tensor,
    raw_keys: torch.Tensor,
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
    *,
    past_raw_keys: Optional[torch.Tensor] = None,
    k_norm_gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select causal QSA tokens ``[B, S, max_selected]`` for standalone/cold prefill.

    Supports both ``indexer.k_norm_gamma`` (source layer) and an overridden
    ``k_norm_gamma`` (YoCo consumer scoring against source raw keys), and uses
    the parameterized ``H_IDX: tl.constexpr`` Triton prefill scorer on CUDA.
    """
    bsz, seq_len, n_heads, head_dim = q_unrot.shape
    ratio = indexer.compress_ratio
    device = q_unrot.device

    all_raw = (
        torch.cat([past_raw_keys, raw_keys], dim=1)
        if past_raw_keys is not None
        else raw_keys
    )
    past_len = int(all_raw.shape[1]) - seq_len
    q_rot = apply_partial_rope(
        q_unrot, cos_all[:, past_len:].unsqueeze(2), sin_all[:, past_len:].unsqueeze(2)
    )

    num_blocks = int(all_raw.shape[1]) // ratio
    positions = past_len + torch.arange(seq_len, dtype=torch.int32, device=device)
    visible_1d = (positions + 1).unsqueeze(0).expand(bsz, seq_len).reshape(-1)
    if num_blocks == 0:
        empty_scores = torch.empty((bsz * seq_len, 0), dtype=torch.float32, device=device)
        return _select_top_tokens(
            empty_scores,
            visible_1d,
            compress_ratio=ratio,
            token_budget=indexer.token_budget,
            scores_are_masked=False,
        ).view(bsz, seq_len, indexer.max_selected)

    gamma = k_norm_gamma if k_norm_gamma is not None else indexer.k_norm_gamma
    groups = all_raw[:, : num_blocks * ratio].unflatten(1, (num_blocks, ratio))
    pooled = groups.float().mean(dim=2).to(all_raw.dtype)
    pooled = exact_head_rms_norm(pooled, gamma, indexer.norm_eps)
    starts = torch.arange(num_blocks, device=device) * ratio
    block_keys = apply_partial_rope(
        pooled, cos_all.index_select(1, starts), sin_all.index_select(1, starts)
    )

    bytes_per_query = max(1, bsz * num_blocks * (n_heads + 1) * 4)
    chunk_size = max(1, _MAX_SCORE_WORKSPACE_BYTES // bytes_per_query)
    if seq_len <= chunk_size:
        scores = qsa_prefill_score(q_rot, block_keys)
        return _select_top_tokens(
            scores.reshape(bsz * seq_len, num_blocks),
            visible_1d,
            compress_ratio=ratio,
            token_budget=indexer.token_budget,
            scores_are_masked=False,
        ).view(bsz, seq_len, indexer.max_selected)

    out = torch.empty(
        (bsz, seq_len, indexer.max_selected), dtype=torch.int32, device=device
    )
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        sub_scores = qsa_prefill_score(q_rot[:, start:end], block_keys)
        sub_vis = (
            (past_len + 1 + torch.arange(start, end, dtype=torch.int32, device=device))
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
        out[:, start:end].copy_(sub_sel)
    return out


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
) -> torch.Tensor:
    """Zero-sync paged QSA scoring + token selection for prefill reuse & decode.

    Args:
        q_unrot_bf16: Normed unrotated indexer query ``[B, S, H_idx, D_idx]``.
        positions: Token positions ``[B, S]`` or ``[B]``.
        kv_pool: ``indexer_kv`` view ``[pool_blocks, entries_per_block, D_idx]``.
        kv_table: ``indexer_kv`` block table ``[B, max_blocks]`` int32.
        visible_lens_dev: Visible token lengths ``[B, S]`` or ``[B]`` int32.
    Returns:
        Selected token indices ``[B, S, token_budget + compress_ratio - 1]`` int32.
    """
    bsz, seq_len, n_heads, head_dim = q_unrot_bf16.shape
    pos_2d = positions.unsqueeze(1) if positions.ndim == 1 else positions
    cos, sin = build_qsa_neox_cos_sin(
        pos_2d, rotary_dim, rope_theta, q_unrot_bf16.dtype
    )
    q_rot = apply_partial_rope(q_unrot_bf16, cos.unsqueeze(2), sin.unsqueeze(2))

    vis_2d = (
        visible_lens_dev.unsqueeze(1)
        if visible_lens_dev.ndim == 1
        else visible_lens_dev
    ).to(dtype=torch.int32)
    ctx_lens = torch.div(vis_2d, compress_ratio, rounding_mode="floor").to(torch.int32)

    if max_ctx_len is None:
        kv_entries_per_block = int(kv_pool.shape[1])
        max_ctx_len = int(kv_table.shape[1]) * kv_entries_per_block

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
) -> None:
    """Write raw indexer keys (state ring) and pooled keys (kv pool) for prefill/suffix."""
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
    pos = seq_lens_before_dev.to(dtype=torch.int64)
    block_starts = pos - torch.remainder(pos, ratio)
    block_cos, block_sin = build_qsa_neox_cos_sin(
        block_starts, rotary_dim, rope_theta, raw_keys.dtype
    )

    if (
        _HAS_TRITON
        and raw_keys.is_cuda
        and ratio == 4
        and head_dim == 128
        and rotary_dim == 64
        and kv_tokens_per_block >= 4
        and kv_tokens_per_block % 4 == 0
        and state_tokens_per_block >= 4
    ):
        from rtp_llm.models_py.kernels.cuda.qsa.indexer_decode import (
            _write_decode_key,
        )

        valid_arg = (
            valid.to(dtype=torch.int8).contiguous()
            if valid is not None
            else seq_lens_before_dev
        )
        _write_decode_key[(bsz,)](
            raw_keys.contiguous(),
            pos.to(dtype=torch.int32).contiguous(),
            valid_arg,
            block_cos.contiguous(),
            block_sin.contiguous(),
            k_norm_gamma,
            kv_pool,
            kv_table.contiguous(),
            state_pool,
            state_table.contiguous(),
            KV_TABLE_WIDTH=int(kv_table.shape[1]),
            STATE_TABLE_WIDTH=int(state_table.shape[1]),
            KV_POOL_BLOCKS=int(kv_pool.shape[0]),
            STATE_POOL_BLOCKS=int(state_pool.shape[0]),
            KV_TOKENS_PER_BLOCK=kv_tokens_per_block,
            STATE_TOKENS_PER_BLOCK=state_tokens_per_block,
            KV_ENTRIES_PER_BLOCK=kv_tokens_per_block // ratio,
            D=head_dim,
            ROTARY=rotary_dim,
            EPS=float(norm_eps),
            HAS_VALID=valid is not None,
        )
        return

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


def qsa_sparse_prefill_attention(
    q_bhtd: torch.Tensor,
    k_bhtd: torch.Tensor,
    v_bhtd: torch.Tensor,
    selected: torch.Tensor,
    *,
    validate_selected: bool = False,
) -> torch.Tensor:
    """Sparse prefill attention ``[B, H_q, S, D]`` over selected token indices."""
    if q_bhtd.is_cuda:
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
    output_gate: Optional[torch.Tensor] = None,
    is_cuda_graph: bool = False,
) -> torch.Tensor:
    """Sparse paged GQA/MQA attention ``[B, H_q, S, D]`` wrapping ``sparse_paged_gqa_attn``."""
    from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import (
        sparse_paged_gqa_attn,
    )

    kv_hnd = view_mha_kv_hnd(
        kv_cache_base, kv_heads=kv_heads, page_size=page_size, head_dim=head_dim
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
    "qsa_sparse_prefill_attention",
    "score_and_select_qsa_paged",
    "view_indexer_kv_pool",
    "view_indexer_state_pool",
    "view_mha_kv_hnd",
    "write_qsa_decode_cache_gpu_",
    "write_qsa_prefill_cache",
]

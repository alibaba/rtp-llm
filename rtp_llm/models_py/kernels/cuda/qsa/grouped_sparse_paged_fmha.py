# SPDX-License-Identifier: Apache-2.0
"""Tensor Core paged QSA with grouped heads and split KV reduction."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qsa.chunk_paged_gqa import (
    _can_use_direct_fp8,
    _fp8_to_bf16,
    _supports_tensor_core_arch,
)

_MAX_GRID_X = 2**31 - 1


def _row_grid(rows):
    """Cover linear rows with legal CUDA axes and at most y-1 padding CTAs."""
    y = triton.cdiv(rows, _MAX_GRID_X)
    if y > 65535:
        raise ValueError("QSA row count exceeds the CUDA grid capacity")
    return triton.cdiv(rows, y), y


def _short_config(batch, queries, kv_heads, *, arch=12, bf16=False):
    """Cold-L2 (splits, KV tile, warps, stages, interleaved tiles).

    Sparse short grids need interleaved splits; large SM120 grids benefit
    from contiguous tiles. BF16 keeps its measured contiguous BK64 policy
    at medium occupancy; the remaining policy is shared with FP8.
    A single stage keeps the 128-key BF16 tile within SM120 shared memory.
    """
    groups = batch * queries * kv_heads
    if bf16 and (
        (arch == 12 and 12 < groups <= 64) or (arch == 10 and 48 < groups <= 64)
    ):
        return 8, 64, 4, 2, False
    if arch == 10:
        if groups <= 12:
            return 32, 128, 4, 1, True
        if groups <= 32:
            return 16, 128, 4, 1, True
        if groups <= 64:
            return 8, 64, 2, 1, True
        return 4, 64, 2, 1, True
    if groups > 48:
        return 4, 128, 4, 1, False
    if groups > 20:
        return 16, 128, 4, 1, True
    if groups > 12:
        if queries == 4 and batch * kv_heads < 8:
            return 32, 32, 4, 1, True
        return 16, 32, 4, 1, True
    if queries not in (1, 4) or (batch == 1 and kv_heads == 2):
        return 32, (128 if groups <= 4 else 32), 4, 1, True
    if batch == 1:
        return 32, (128 if queries == 4 else 64), 4, 1, True
    if (batch, queries, kv_heads) == (4, 1, 1):
        return 32, 128, 4, 1, True
    return 32, 32, 4, 1, True


def preferred_splits(q, cache, selected, *, page_size, block_k):
    """Shared BF16/FP8 TP1/TP2 short-window launch policy."""
    if not (
        q.ndim == 4
        and cache.ndim == 5
        and q.dtype == torch.bfloat16
        and cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and cache.shape[2] in (1, 2)
        and q.shape[1] == 12 * cache.shape[2]
        and q.shape[3] == 256
        and q.shape[0] in (1, 2, 4, 8, 16)
        and 1 <= q.shape[2] <= 15
        and selected.shape[:2] == (q.shape[0], q.shape[2])
        and selected.shape[-1] in (2051, 2052)
        and page_size == 128
        and block_k == 64
        and q.is_cuda
        and torch.version.hip is None
        and _supports_tensor_core_arch(torch.cuda.get_device_capability(q.device))
    ):
        return None
    return _short_config(
        q.shape[0],
        q.shape[2],
        cache.shape[2],
        arch=torch.cuda.get_device_capability(q.device)[0],
        bf16=cache.dtype == torch.bfloat16,
    )[0]


@triton.jit
def _grouped_sparse_paged_split_kernel(
    q,
    cache,
    scale,
    table,
    lengths,
    selected,
    partial,
    QB: tl.constexpr,
    QH: tl.constexpr,
    QS: tl.constexpr,
    CP: tl.constexpr,
    CKV: tl.constexpr,
    CH: tl.constexpr,
    CT: tl.constexpr,
    BTW: tl.constexpr,
    S: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    WIDTH: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    NB: tl.constexpr,
    HQ: tl.constexpr,
    HG: tl.constexpr,
    FP8_CACHE: tl.constexpr,
    SP: tl.constexpr,
    SKV: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    ROTATE_TILES: tl.constexpr,
    FP16_PV: tl.constexpr,
    DRAFT_CU=None,
    DRAFT_INPUT: tl.constexpr = False,
    DRAFT_QS: tl.constexpr = 0,
    DRAFT_LENGTHS=None,
    DRAFT_PREFIX=None,
    DIRECT_FP8: tl.constexpr = False,
    UNIT_SCALE: tl.constexpr = False,
    D: tl.constexpr = 256,
    WIDE_ADDRESS: tl.constexpr = False,
    FLAT_GRID: tl.constexpr = False,
    BATCH: tl.constexpr = 1,
):
    if FLAT_GRID:
        linear = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
            tl.int64
        ) * tl.num_programs(0)
        groups = (HQ // HG) * tl.cdiv(HG, 16) * S
        group_query = linear % groups
        batch = linear // groups
    else:
        batch = tl.program_id(0)
        if WIDE_ADDRESS:
            batch = batch.to(tl.int64)
        group_query = tl.program_id(1)
    if not FLAT_GRID or batch < BATCH:
        query = group_query % S
        head_tiles = tl.cdiv(HG, 16)
        group = group_query // (S * head_tiles)
        split = tl.program_id(2)
        head = (group_query // S % head_tiles) * 16 + tl.arange(0, 16)
        global_head = group * HG + head
        dim = tl.arange(0, max(16, D))
        cols = tl.arange(0, BK)
        if DRAFT_INPUT:
            active = query < tl.load(DRAFT_LENGTHS + batch)
            source = tl.where(active, tl.load(DRAFT_CU + batch) + query, 0)
            query_base = source * DRAFT_QS
            selected_row = source
        else:
            query_base = batch * QB + query * QS
            selected_row = batch * S + query
        query_values = tl.load(
            q + query_base + global_head[:, None] * QH + dim[None, :],
            mask=(head[:, None] < HG) & (dim[None, :] < D),
            other=0,
        )
        if DRAFT_INPUT:
            length = tl.where(active, tl.load(DRAFT_PREFIX + batch) + query + 1, 0)
        else:
            length = tl.load(lengths + batch * S + query)
        row = selected + selected_row * ROW_WIDTH
        count = (
            tl.minimum(tl.maximum(tl.load(row + WIDTH), 0), WIDTH)
            if HAS_COUNT
            else WIDTH
        )
        maximum = tl.full((16,), float("-inf"), tl.float32)
        normalizer = tl.zeros((16,), tl.float32)
        acc = tl.zeros((16, max(16, D)), tl.float32)
        # Round-robin tiles keep all useful splits active when 2051 slots need
        # 33 tiles. Contiguous ceil-sized partitions used only 17 of 32 splits.
        if ROTATE_TILES:
            limit = tl.maximum(0, tl.cdiv(count - split * BK, NS * BK))
        else:
            limit = tl.maximum(0, tl.minimum(NB, tl.cdiv(count - split * NB * BK, BK)))
        limit = tl.where(length > 0, limit, 0)
        for step in range(limit):
            if ROTATE_TILES:
                offset = (split + step * NS) * BK + cols
            else:
                offset = (split * NB + step) * BK + cols
            token = tl.load(row + offset, mask=offset < count, other=-1)
            logical = token // PAGE
            physical = tl.load(
                table + batch * BTW + logical,
                mask=(token >= 0) & (token < length) & (logical < BTW),
                other=0,
            ).to(tl.int64)
            valid = (
                (offset < count)
                & (token >= 0)
                & (token < length)
                & (logical < BTW)
                & (physical > 0)
                & (physical < PAGES)
            )
            kp = (
                cache
                + physical[:, None] * CP
                + group * CH
                + (token % PAGE)[:, None] * CT
                + dim[None, :]
            )
            key = tl.load(kp, mask=valid[:, None] & (dim[None, :] < D), other=0.0)
            if FP8_CACHE:
                if DIRECT_FP8:
                    key = _fp8_to_bf16(key)
            if FP8_CACHE and not UNIT_SCALE:
                sp = scale + physical * SP + group * SH + (token % PAGE) * ST
                kscale = tl.load(sp, mask=valid, other=1.0)
                key = (key.to(tl.float32) * kscale[:, None]).to(tl.bfloat16)
            else:
                key = key.to(tl.bfloat16)
            score = tl.dot(query_values, tl.trans(key)) * (D**-0.5)
            score = tl.where(valid[None, :], score, float("-inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1))
            alpha = tl.where(normalizer > 0, tl.exp(maximum - new_max), 0.0)
            prob = tl.where(valid[None, :], tl.exp(score - new_max[:, None]), 0.0)
            value = tl.load(
                kp + CKV, mask=valid[:, None] & (dim[None, :] < D), other=0.0
            )
            if FP8_CACHE:
                if DIRECT_FP8:
                    value = _fp8_to_bf16(value)
            if FP8_CACHE and not UNIT_SCALE:
                vscale = tl.load(sp + SKV, mask=valid, other=1.0)
                value = value.to(tl.float32) * vscale[:, None]
            # Use one PV product with FP32 accumulation. The explicit FP16 option
            # falls back to a single BF16 product for values outside FP16 range.
            value_bf16 = value.to(tl.bfloat16)
            if FP16_PV:
                safe_fp16 = (
                    tl.max(tl.max(tl.abs(value_bf16.to(tl.float32)), 1), 0) <= 65504.0
                )
                if safe_fp16:
                    acc = tl.dot(
                        prob.to(tl.float16),
                        value_bf16.to(tl.float16),
                        acc * alpha[:, None],
                    )
                else:
                    acc = tl.dot(prob.to(tl.bfloat16), value_bf16, acc * alpha[:, None])
            else:
                acc = tl.dot(prob.to(tl.bfloat16), value_bf16, acc * alpha[:, None])
            normalizer = normalizer * alpha + tl.sum(prob, 1)
            maximum = new_max
        row = (batch * HQ + global_head) * S + query
        if NS == 1:
            output = tl.where(normalizer[:, None] > 0, acc / normalizer[:, None], 0.0)
            tl.store(
                partial + row[:, None] * D + dim[None, :],
                output,
                mask=(head[:, None] < HG) & (dim[None, :] < D),
            )
        else:
            base = (row * NS + split) * (D + 2)
            tl.store(
                partial + base[:, None] + dim[None, :],
                acc,
                mask=(head[:, None] < HG) & (dim[None, :] < D),
            )
            tl.store(partial + base + D, maximum, mask=head < HG)
            tl.store(partial + base + D + 1, normalizer, mask=head < HG)


def grouped_sparse_paged_split(
    q,
    cache,
    table,
    lengths,
    selected,
    partial,
    *,
    page_size,
    splits,
    scale=None,
    selected_has_count=False,
    fp16_pv=False,
    unit_scale=False,
    block_k=64,
):
    """Write BF16 output for one split, otherwise FP32 split states.

    K/V are shared across each query head group. BF16 and FP8 caches use a
    single BF16 PV with FP32 accumulation by default; arbitrary scales keep
    the BF16 dequant boundary.
    ``unit_scale`` certifies the entire visible cache, including prefixes.
    The caller validates metadata and selects the verified shape domain.
    """
    batch, heads, query_len, dim = q.shape
    row_width = selected.shape[-1]
    width = row_width - int(selected_has_count)
    kv_heads = cache.shape[2]
    fp8 = cache.dtype == torch.float8_e4m3fn
    direct_fp8 = fp8 and unit_scale and _can_use_direct_fp8(q.device)
    strides = scale.stride() if fp8 and not unit_scale else (0, 0, 0, 0)
    tuned = splits == preferred_splits(
        q, cache, selected, page_size=page_size, block_k=block_k
    )
    block_k, num_warps, num_stages = max(32, block_k), 4, 2
    rotate_tiles = False
    if dim > 256:
        block_k, num_stages = 32, 1
    if tuned and not fp16_pv:
        _, block_k, num_warps, num_stages, rotate_tiles = _short_config(
            batch,
            query_len,
            kv_heads,
            arch=torch.cuda.get_device_capability(q.device)[0],
            bf16=cache.dtype == torch.bfloat16,
        )
    groups = kv_heads * triton.cdiv(heads // kv_heads, 16) * query_len
    flat_grid = groups > 65535
    grid = (
        (*_row_grid(batch * groups), splits) if flat_grid else (batch, groups, splits)
    )
    _grouped_sparse_paged_split_kernel[grid](
        q,
        cache,
        cache if scale is None else scale,
        table,
        lengths,
        selected,
        partial,
        QB=q.stride(0),
        QH=q.stride(1),
        QS=q.stride(2),
        CP=cache.stride(0),
        CKV=cache.stride(1),
        CH=cache.stride(2),
        CT=cache.stride(3),
        BTW=table.shape[1],
        S=query_len,
        PAGE=page_size,
        PAGES=cache.shape[0],
        WIDTH=width,
        ROW_WIDTH=row_width,
        HAS_COUNT=selected_has_count,
        BK=block_k,
        NS=splits,
        NB=triton.cdiv(width, block_k * splits),
        HQ=heads,
        HG=heads // kv_heads,
        D=dim,
        WIDE_ADDRESS=max(
            q.stride(0) * batch, table.numel(), selected.numel(), partial.numel()
        )
        >= 2**31,
        FP8_CACHE=fp8,
        SP=strides[0],
        SKV=strides[1],
        SH=strides[2],
        ST=strides[3],
        ROTATE_TILES=rotate_tiles,
        FP16_PV=fp16_pv,
        DIRECT_FP8=direct_fp8,
        FLAT_GRID=flat_grid,
        BATCH=batch,
        UNIT_SCALE=fp8 and unit_scale,
        num_warps=num_warps,
        num_stages=num_stages,
        enable_fp_fusion=False,
    )

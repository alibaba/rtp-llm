# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Hopper V4.1 scoring over packed FP4 keys with BF16 tensor-core math.

Adapted from SGLang's kernels/ops/attention/dsv4/fp4_indexer.py. RTP keeps
its own block-table and ratio-2 half-page addressing. No capacity-sized
dequantized keys or [query,head,key] intermediate is materialized.

Candidate addressing follows SGLang PR #40860, candidate_bf16_mqa.py at
f983aca2d2f04f87bb504235728156441888d6f1. Unlike its compact output, retain
RTP's logical-width logits so native TopK dispatch and exact ties do not
change. Only candidate dot products are evaluated; dense initialization
and selection remain in this conservative first optimization.
"""

import os

import torch
import triton
import triton.language as tl

from ._v41_fp4_triton import _e2m1_to_float, _ue8m0_to_float


@triton.jit
def _score_kernel(
    q_ptr,
    q_scale_ptr,
    key_ptr,
    key_scale_ptr,
    weight_ptr,
    table_ptr,
    start_ptr,
    end_ptr,
    out_ptr,
    CAPACITY: tl.constexpr,
    HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    PAGED: tl.constexpr,
    PHYSICAL: tl.constexpr,
    LOGICAL: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    candidate_ptr=None,
    candidate_end_ptr=None,
    CANDIDATES: tl.constexpr = False,
    CANDIDATE_BLOCKS: tl.constexpr = 0,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(start_ptr + row)
    end = tl.load(end_ptr + row)
    visible = (cols >= begin) & (cols < end) & (cols < CAPACITY)
    active_tile = (tl.program_id(1) * BLOCK < end) & (
        (tl.program_id(1) + 1) * BLOCK > begin
    )
    if CANDIDATES:
        sparse_end = tl.load(candidate_end_ptr + row)
        sparse_valid = cols < sparse_end
        block = tl.load(
            candidate_ptr + row * CANDIDATE_BLOCKS + cols // 8,
            sparse_valid,
            other=-1,
        )
        cols = block * 8 + cols % 8
        visible = sparse_valid & (cols >= begin) & (cols < end) & (cols < CAPACITY)
        active_tile = tl.program_id(1) * BLOCK < sparse_end
    if active_tile:
        dims = tl.arange(0, 128)
        heads = tl.arange(0, HEADS)
        q_bytes = tl.load(
            q_ptr + (row * HEADS + heads[:, None]) * 64 + dims[None, :] // 2
        ).to(tl.uint8)
        q_code = (q_bytes >> ((dims[None, :] % 2) * 4)) & 15
        q_scales = tl.load(q_scale_ptr + row * HEADS + heads)
        q_exponent = (q_scales[:, None] >> ((dims[None, :] // 32) * 8)) & 255
        q = (_e2m1_to_float(q_code) * _ue8m0_to_float(q_exponent)).to(tl.bfloat16)
        if PAGED:
            page = tl.load(
                table_ptr
                + (row // QUERY_LEN) * TABLE_STRIDE
                + (cols // LOGICAL) * TABLE_COL_STRIDE,
                visible,
                other=0,
            ).to(tl.int64)
            offset = cols % LOGICAL
            # Preserve RTP's visible zero-K sentinel contract. Nonpositive
            # table entries never read storage and score zero, not -inf.
            valid = visible & (page > 0)
            key_bytes = tl.load(
                key_ptr
                + page[:, None] * PAGE_STRIDE
                + offset[:, None] * 64
                + dims[None, :] // 2,
                valid[:, None],
                other=0,
            ).to(tl.uint8)
            key_exponent = tl.load(
                key_ptr
                + page[:, None] * PAGE_STRIDE
                + PHYSICAL * 64
                + offset[:, None] * 4
                + dims[None, :] // 32,
                valid[:, None],
                other=127,
            )
        else:
            key_bytes = tl.load(
                key_ptr + cols[:, None] * 64 + dims[None, :] // 2,
                visible[:, None],
                other=0,
            ).to(tl.uint8)
            scales = tl.load(key_scale_ptr + cols, visible, other=0x7F7F7F7F)
            key_exponent = (scales[:, None] >> ((dims[None, :] // 32) * 8)) & 255
        key_code = (key_bytes >> ((dims[None, :] % 2) * 4)) & 15
        keys = (_e2m1_to_float(key_code) * _ue8m0_to_float(key_exponent)).to(
            tl.bfloat16
        )
        dots = tl.dot(q, tl.trans(keys)).to(tl.bfloat16).to(tl.float32)
        weights = tl.load(weight_ptr + row * HEADS + heads).to(tl.float32)
        weighted = (tl.maximum(dots, 0.0) * weights[:, None]).to(tl.bfloat16)
        scores = tl.sum(weighted.to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
        if CANDIDATES:
            # Prepared candidates are unique: no duplicate scatter races.
            # All other dense columns were initialized to -inf by the caller.
            tl.store(out_ptr + row * CAPACITY + cols, scores, visible)
        else:
            tl.store(
                out_ptr + row * CAPACITY + cols,
                tl.where(visible, scores, -float("inf")),
                cols < CAPACITY,
            )
    else:
        if not CANDIDATES:
            tl.store(out_ptr + row * CAPACITY + cols, -float("inf"), cols < CAPACITY)


def _score(
    q,
    q_scales,
    keys,
    key_scales,
    weights,
    starts,
    ends,
    capacity,
    *,
    table=None,
    logical_entries=None,
    candidate_plan=None,
):
    heads = q.shape[-2]
    if heads not in (32, 64) or q.shape[-1] != 64:
        raise ValueError("SM90 V4.1 indexer requires 32/64 heads with packed dim 64")
    rows = q.numel() // (heads * 64)
    output = (
        torch.empty((rows, capacity), dtype=torch.float32, device=q.device)
        if candidate_plan is None
        else torch.full(
            (rows, capacity), -torch.inf, dtype=torch.float32, device=q.device
        )
    )
    if rows == 0 or capacity == 0:
        return output
    paged = table is not None
    if paged and (keys.dtype != torch.uint8 or keys.shape[-1] != 68):
        raise ValueError("SM90 indexer requires the 68B per-page planar FP4 pool")
    # Validated two-query K reuse: exact scores/TopK, graph replay and 8-H20
    # CP8/EP8 128K smoke; paired full-prefill latency improved by 5.24%.
    # Preserve each original M32 dot/head-reduction tree. BK64 regressed;
    # select only BK128 and keep paged/sparse/H64/small shapes unchanged.
    # Explicit 0 retains the old producer for diagnosis and A/B measurements.
    if (
        not paged
        and candidate_plan is None
        and heads == 32
        and rows >= 512
        and 32768 <= capacity <= 131072
        and rows * capacity <= 64 * 1024 * 1024
        and os.environ.get("DSV41_SM90_MULTIQUERY_INDEXER", "1") == "1"
        and torch.cuda.get_device_capability(q.device) == (9, 0)
    ):
        from ._v41_sm90_indexer_multiquery import dense_score_two_queries

        return dense_score_two_queries(
            q.reshape(rows, heads, 64).contiguous(),
            q_scales.reshape(rows, heads).contiguous(),
            keys,
            key_scales,
            weights.to(torch.bfloat16).reshape(rows, heads).contiguous(),
            starts.contiguous(),
            ends.contiguous(),
            block=128,
            out=output,
        )
    scoring_width = (
        capacity if candidate_plan is None else candidate_plan.sparse_columns
    )
    # Validated dense-producer launch only. Keep paged decode, candidate
    # scatter, H64 and small shapes on their original geometry/rounding.
    block = (
        128
        if not paged
        and candidate_plan is None
        and heads == 32
        and rows >= 512
        and capacity >= 1024
        else 64
    )
    _score_kernel[(rows, triton.cdiv(scoring_width, block))](
        q.contiguous(),
        q_scales.contiguous(),
        keys,
        key_scales,
        weights.to(torch.bfloat16).contiguous(),
        table if paged else starts,
        starts.contiguous(),
        ends.contiguous(),
        output,
        CAPACITY=capacity,
        HEADS=heads,
        QUERY_LEN=q.shape[1] if paged else 1,
        PAGED=paged,
        PHYSICAL=keys.shape[1] if paged else 1,
        LOGICAL=logical_entries if paged else 1,
        PAGE_STRIDE=keys.stride(0) if paged else 1,
        TABLE_STRIDE=table.stride(0) if paged else 1,
        TABLE_COL_STRIDE=table.stride(1) if paged else 1,
        BLOCK=block,
        **(
            {
                "candidate_ptr": candidate_plan.sparse_indices,
                "candidate_end_ptr": candidate_plan.end,
                "CANDIDATES": True,
                "CANDIDATE_BLOCKS": candidate_plan.sparse_indices.shape[1],
            }
            if candidate_plan is not None
            else {}
        ),
        num_warps=4,
    )
    return output


def candidate_score(q, q_scales, keys, key_scales, weights, plan):
    """Score a unique candidate prefix into baseline-shaped FP32 logits.

    The plan must describe one request's contiguous, logically restored K.
    CP zigzag query ordering is carried by per-row causal bounds; no global
    positions are inferred from row order. Packed FP4 and every existing
    BF16 rounding boundary remain unchanged. No DeepGEMM/DeepSelect is used.
    """
    if plan.row_key_offsets is not None or plan.block_size != 8:
        raise ValueError("SM90 candidate scoring requires request-local block-8 keys")
    if keys.shape[0] != plan.key_count or q.shape[0] != plan.rows:
        raise ValueError("SM90 candidate plan does not match query/key shape")
    return _score(
        q,
        q_scales,
        keys,
        key_scales,
        weights,
        plan.row_ks,
        plan.row_ke,
        plan.key_count,
        candidate_plan=plan,
    )


def dense_score(q, q_scales, keys, key_scales, weights, starts, ends):
    return _score(
        q,
        q_scales,
        keys.contiguous(),
        key_scales.contiguous(),
        weights,
        starts,
        ends,
        keys.shape[0],
    )


def paged_score(q, q_scales, pool, weights, table, lengths, capacity, logical_entries):
    if capacity > table.shape[1] * logical_entries:
        raise ValueError("Indexer capacity exceeds block-table coverage")
    lengths = lengths.reshape(-1).to(torch.int32).clamp(0, capacity)
    return _score(
        q,
        q_scales,
        pool,
        pool,
        weights,
        torch.zeros_like(lengths),
        lengths,
        capacity,
        table=table,
        logical_entries=logical_entries,
    )

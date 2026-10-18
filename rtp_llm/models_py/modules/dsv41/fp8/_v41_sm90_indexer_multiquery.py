# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Bounded two-query tile adaptation of the existing SM90 score reference.

Derived from _v41_sm90_indexer._score_kernel (itself SGLang FP4 indexer-derived).
Only the query tile/packed-K reuse changes: decode K once, then retain TWO
independent 32-head dot/reduction expressions. The earlier concatenated M64
variant is retained only for failure diagnostics, never selected by the public
backend wrapper. The production dispatcher selects only the validated BK128
large-dense shape range. This is not a new attention algorithm; paged and
sparse-candidate addressing retain the original implementation.

Boundaries remain FP4*g32 UE8M0 -> BF16, BF16 dot -> BF16, ReLU * BF16
weights -> BF16, FP32 32-head sum -> BF16 -> FP32. No new quantization or
candidate/topk policy. Dense H32/D128 only; paged/candidate addressing excluded.
"""

import torch
import triton
import triton.language as tl

from ._v41_fp4_triton import _e2m1_to_float, _ue8m0_to_float
from ._v41_sm90_indexer import _score_kernel


@triton.jit
def _score_stage_probe(
    q_ptr,
    qs_ptr,
    key_ptr,
    ks_ptr,
    weight_ptr,
    raw_ptr,
    dot_ptr,
    weighted_ptr,
    sum_ptr,
    score_ptr,
    ROW_BASE: tl.constexpr,
    COL_BASE: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Diagnostic-only instrumentation of the SAME M32/M64 tile expressions.

    No timing/production use. The test must verify emitted final scores against
    uninstrumented full-shape launches before trusting intermediate values.
    Inputs are full valid tiles from a failing fixture (no addressing masks).
    """
    heads = tl.arange(0, QUERY_ROWS * 32)
    dims = tl.arange(0, 128)
    cols = COL_BASE + tl.arange(0, BLOCK)
    q_bytes = tl.load(
        q_ptr + (ROW_BASE * 32 + heads[:, None]) * 64 + dims[None, :] // 2
    ).to(tl.uint8)
    q_code = (q_bytes >> ((dims[None, :] % 2) * 4)) & 15
    qs = tl.load(qs_ptr + ROW_BASE * 32 + heads)
    qe = (qs[:, None] >> ((dims[None, :] // 32) * 8)) & 255
    q = (_e2m1_to_float(q_code) * _ue8m0_to_float(qe)).to(tl.bfloat16)
    kb = tl.load(key_ptr + cols[:, None] * 64 + dims[None, :] // 2).to(tl.uint8)
    kc = (kb >> ((dims[None, :] % 2) * 4)) & 15
    ks = tl.load(ks_ptr + cols)
    ke = (ks[:, None] >> ((dims[None, :] // 32) * 8)) & 255
    keys = (_e2m1_to_float(kc) * _ue8m0_to_float(ke)).to(tl.bfloat16)
    raw = tl.dot(q, tl.trans(keys))
    dots = raw.to(tl.bfloat16).to(tl.float32)
    weights = tl.load(weight_ptr + ROW_BASE * 32 + heads).to(tl.float32)
    weighted = (tl.maximum(dots, 0.0) * weights[:, None]).to(tl.bfloat16).to(tl.float32)
    if QUERY_ROWS == 1:
        sums = tl.sum(weighted, axis=0)
        output_offsets = tl.arange(0, BLOCK)
    else:
        sums = tl.sum(tl.reshape(weighted, (2, 32, BLOCK)), axis=1)
        output_offsets = tl.arange(0, 2)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    offsets = heads[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    tl.store(raw_ptr + offsets, raw)
    tl.store(dot_ptr + offsets, dots)
    tl.store(weighted_ptr + offsets, weighted)
    tl.store(sum_ptr + output_offsets, sums)
    tl.store(score_ptr + output_offsets, sums.to(tl.bfloat16).to(tl.float32))


@triton.jit(do_not_specialize=["ROWS"])
def _score_two_queries_concat_rejected_kernel(
    q_ptr,
    qs_ptr,
    key_ptr,
    ks_ptr,
    weight_ptr,
    start_ptr,
    end_ptr,
    out_ptr,
    ROWS,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Existing 32-head GEMM M axis becomes two adjacent groups of 32 heads.
    # K decode and K operand are shared; K=128 dot traversal is unchanged.
    group = tl.program_id(0)
    query_rows = group * 2 + tl.arange(0, 2)
    begins = tl.load(start_ptr + query_rows, query_rows < ROWS, other=CAPACITY)
    ends = tl.load(end_ptr + query_rows, query_rows < ROWS, other=0)
    union_begin = tl.min(begins, axis=0)
    union_end = tl.max(ends, axis=0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    visible = (
        (query_rows[:, None] < ROWS)
        & (cols[None, :] >= begins[:, None])
        & (cols[None, :] < ends[:, None])
        & (cols[None, :] < CAPACITY)
    )
    active_tile = (tl.program_id(1) * BLOCK < union_end) & (
        (tl.program_id(1) + 1) * BLOCK > union_begin
    )
    if active_tile:
        dims = tl.arange(0, 128)
        query_heads = tl.arange(0, 64)
        rows = group * 2 + query_heads // 32
        flat_heads = rows * 32 + query_heads % 32
        q_bytes = tl.load(
            q_ptr + flat_heads[:, None] * 64 + dims[None, :] // 2,
            rows[:, None] < ROWS,
            other=0,
        ).to(tl.uint8)
        q_code = (q_bytes >> ((dims[None, :] % 2) * 4)) & 15
        q_scales = tl.load(qs_ptr + flat_heads, rows < ROWS, other=0x7F7F7F7F)
        q_exponent = (q_scales[:, None] >> ((dims[None, :] // 32) * 8)) & 255
        q = (_e2m1_to_float(q_code) * _ue8m0_to_float(q_exponent)).to(tl.bfloat16)

        # A conservative union is valid even for disjoint CP zigzag bounds.
        # Per-query visibility is restored on store; no causal order inferred.
        key_visible = (cols >= union_begin) & (cols < union_end) & (cols < CAPACITY)
        key_bytes = tl.load(
            key_ptr + cols[:, None] * 64 + dims[None, :] // 2,
            key_visible[:, None],
            other=0,
        ).to(tl.uint8)
        key_scales = tl.load(ks_ptr + cols, key_visible, other=0x7F7F7F7F)
        key_exponent = (key_scales[:, None] >> ((dims[None, :] // 32) * 8)) & 255
        key_code = (key_bytes >> ((dims[None, :] % 2) * 4)) & 15
        keys = (_e2m1_to_float(key_code) * _ue8m0_to_float(key_exponent)).to(
            tl.bfloat16
        )

        dots = tl.dot(q, tl.trans(keys)).to(tl.bfloat16).to(tl.float32)
        weights = tl.load(weight_ptr + flat_heads, rows < ROWS, other=0).to(tl.float32)
        weighted = (tl.maximum(dots, 0.0) * weights[:, None]).to(tl.bfloat16)
        # Do not combine the two queries' reductions. Same 32 input heads,
        # FP32 reduction and final BF16 boundary as the existing score kernel.
        per_query = tl.reshape(weighted.to(tl.float32), (2, 32, BLOCK))
        scores = tl.sum(per_query, axis=1).to(tl.bfloat16).to(tl.float32)
        tl.store(
            out_ptr + query_rows[:, None] * CAPACITY + cols[None, :],
            tl.where(visible, scores, -float("inf")),
            (query_rows[:, None] < ROWS) & (cols[None, :] < CAPACITY),
        )
    else:
        tl.store(
            out_ptr + query_rows[:, None] * CAPACITY + cols[None, :],
            -float("inf"),
            (query_rows[:, None] < ROWS) & (cols[None, :] < CAPACITY),
        )


@triton.jit(do_not_specialize=["ROWS"])
def _score_two_queries_kernel(
    q_ptr,
    qs_ptr,
    key_ptr,
    ks_ptr,
    weight_ptr,
    start_ptr,
    end_ptr,
    out_ptr,
    ROWS,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Share K decode but retain original M32 dot and axis0 reduction.

    The rejected concat variant changes MMA2 to WGMMA3 and the head reduction
    tree. Diag03 proved identical per-head weighted BF16 values but differing
    FP32 reductions. Keep the exact original expression shape for BOTH rows;
    compiled MMA/reduction layout and same-fixture exact gates must verify it.
    """
    row_base = tl.program_id(0) * 2
    query_rows = row_base + tl.arange(0, 2)
    begins = tl.load(start_ptr + query_rows, query_rows < ROWS, other=CAPACITY)
    ends = tl.load(end_ptr + query_rows, query_rows < ROWS, other=0)
    union_begin = tl.min(begins, axis=0)
    union_end = tl.max(ends, axis=0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    union_active = (tl.program_id(1) * BLOCK < union_end) & (
        (tl.program_id(1) + 1) * BLOCK > union_begin
    )
    if union_active:
        dims = tl.arange(0, 128)
        heads = tl.arange(0, 32)
        key_visible = (cols >= union_begin) & (cols < union_end) & (cols < CAPACITY)
        key_bytes = tl.load(
            key_ptr + cols[:, None] * 64 + dims[None, :] // 2,
            key_visible[:, None],
            other=0,
        ).to(tl.uint8)
        key_code = (key_bytes >> ((dims[None, :] % 2) * 4)) & 15
        key_scales = tl.load(ks_ptr + cols, key_visible, other=0x7F7F7F7F)
        key_exponent = (key_scales[:, None] >> ((dims[None, :] // 32) * 8)) & 255
        keys = (_e2m1_to_float(key_code) * _ue8m0_to_float(key_exponent)).to(
            tl.bfloat16
        )
        for query in tl.static_range(2):
            row = row_base + query
            begin = tl.load(start_ptr + row, row < ROWS, other=CAPACITY)
            end = tl.load(end_ptr + row, row < ROWS, other=0)
            visible = (cols >= begin) & (cols < end) & (cols < CAPACITY)
            active = (
                (row < ROWS)
                & (tl.program_id(1) * BLOCK < end)
                & ((tl.program_id(1) + 1) * BLOCK > begin)
            )
            if active:
                q_bytes = tl.load(
                    q_ptr + (row * 32 + heads[:, None]) * 64 + dims[None, :] // 2
                ).to(tl.uint8)
                q_code = (q_bytes >> ((dims[None, :] % 2) * 4)) & 15
                q_scales = tl.load(qs_ptr + row * 32 + heads)
                q_exponent = (q_scales[:, None] >> ((dims[None, :] // 32) * 8)) & 255
                q = (_e2m1_to_float(q_code) * _ue8m0_to_float(q_exponent)).to(
                    tl.bfloat16
                )
                dots = tl.dot(q, tl.trans(keys)).to(tl.bfloat16).to(tl.float32)
                weights = tl.load(weight_ptr + row * 32 + heads).to(tl.float32)
                weighted = (tl.maximum(dots, 0.0) * weights[:, None]).to(tl.bfloat16)
                scores = (
                    tl.sum(weighted.to(tl.float32), axis=0)
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                tl.store(
                    out_ptr + row * CAPACITY + cols,
                    tl.where(visible, scores, -float("inf")),
                    cols < CAPACITY,
                )
            else:
                tl.store(
                    out_ptr + row * CAPACITY + cols,
                    -float("inf"),
                    (row < ROWS) & (cols < CAPACITY),
                )
    else:
        tl.store(
            out_ptr + query_rows[:, None] * CAPACITY + cols[None, :],
            -float("inf"),
            (query_rows[:, None] < ROWS) & (cols[None, :] < CAPACITY),
        )


def dense_score_two_queries(
    q, qs, keys, ks, weights, starts, ends, *, block=128, out=None
):
    """Backend for the bounded dispatcher and explicit component benchmarks.

    M=1 deliberately uses the existing path and is reported as FALLBACK_M1 in
    benchmarks. BF16 weights are accepted without an extra conversion. Optional
    preallocated output isolates producer time in local A/B measurements.
    """
    if block not in (64, 128):
        raise ValueError("Only BQ2/BK64-or-128/warps4 are under evaluation")
    if q.ndim != 3 or q.shape[1:] != (32, 64) or keys.ndim != 2 or keys.shape[1] != 64:
        raise ValueError("Dense FP4 H32/D128 contiguous rows only")
    rows, capacity = q.shape[0], keys.shape[0]
    if rows * capacity >= 2**31:
        raise ValueError("Candidate output indexing requires rows * capacity < 2**31")
    if qs.shape != (rows, 32) or ks.shape != (capacity,) or weights.shape != (rows, 32):
        raise ValueError("Inconsistent quantization/weight shape")
    tensors = (q, qs, keys, ks, weights, starts, ends)
    if any(
        not t.is_cuda or t.device != q.device or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("Candidate requires same-device contiguous CUDA tensors")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("This candidate is validated only for SM90")
    if q.dtype not in (torch.int8, torch.uint8) or keys.dtype not in (
        torch.int8,
        torch.uint8,
    ):
        raise ValueError("FP4 payload must be packed int8/uint8 bytes")
    if qs.dtype != torch.int32 or ks.dtype != torch.int32:
        raise ValueError("FP4 g32 scales must be packed int32")
    if starts.dtype not in (torch.int32, torch.int64) or ends.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Causal bounds must be int32/int64")
    if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "Weights must be BF16/FP16/FP32, converted at the original BF16 boundary"
        )
    if starts.shape != (rows,) or ends.shape != (rows,):
        raise ValueError("Per-query causal starts/ends required")
    if out is None:
        out = torch.empty((rows, capacity), device=q.device, dtype=torch.float32)
    if (
        out.shape != (rows, capacity)
        or out.dtype != torch.float32
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError("Invalid score output buffer")
    if not rows or not capacity:
        return out
    if rows == 1:
        _score_kernel[(rows, triton.cdiv(capacity, 64))](
            q,
            qs,
            keys,
            ks,
            weights.to(torch.bfloat16),
            starts,
            starts,
            ends,
            out,
            CAPACITY=capacity,
            HEADS=32,
            QUERY_LEN=1,
            PAGED=False,
            PHYSICAL=1,
            LOGICAL=1,
            PAGE_STRIDE=1,
            TABLE_STRIDE=1,
            TABLE_COL_STRIDE=1,
            BLOCK=64,
            num_warps=4,
        )
        return out
    _score_two_queries_kernel[(triton.cdiv(rows, 2), triton.cdiv(capacity, block))](
        q,
        qs,
        keys,
        ks,
        weights.to(torch.bfloat16),
        starts,
        ends,
        out,
        ROWS=rows,
        CAPACITY=capacity,
        BLOCK=block,
        num_warps=4,
    )
    return out

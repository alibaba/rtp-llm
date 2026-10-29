"""Fused operand copies and output assembly for SM90 FlashMLA queries."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["B", "QUERY"])
    def _copy_single_query_kernel(
        q_ptr,
        indices_ptr,
        extra_ptr,
        out_q_ptr,
        out_indices_ptr,
        out_extra_ptr,
        B,
        QUERY,
        q_stride_b: tl.constexpr,
        q_stride_s: tl.constexpr,
        indices_stride_b: tl.constexpr,
        indices_stride_s: tl.constexpr,
        extra_stride_b: tl.constexpr,
        extra_stride_s: tl.constexpr,
        Q_ROW: tl.constexpr,
        TOPK: tl.constexpr,
        EXTRA_TOPK: tl.constexpr,
        HAS_EXTRA: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

        q_count = B * Q_ROW
        q_mask = offsets < q_count
        q_batch = offsets // Q_ROW
        q_inner = offsets % Q_ROW
        q = tl.load(
            q_ptr + q_batch * q_stride_b + QUERY * q_stride_s + q_inner,
            mask=q_mask,
        )
        tl.store(out_q_ptr + offsets, q, mask=q_mask)

        index_count = B * TOPK
        index_mask = offsets < index_count
        index_batch = offsets // TOPK
        index_inner = offsets % TOPK
        index = tl.load(
            indices_ptr
            + index_batch * indices_stride_b
            + QUERY * indices_stride_s
            + index_inner,
            mask=index_mask,
        )
        tl.store(out_indices_ptr + offsets, index, mask=index_mask)

        if HAS_EXTRA:
            extra_count = B * EXTRA_TOPK
            extra_mask = offsets < extra_count
            extra_batch = offsets // EXTRA_TOPK
            extra_inner = offsets % EXTRA_TOPK
            extra = tl.load(
                extra_ptr
                + extra_batch * extra_stride_b
                + QUERY * extra_stride_s
                + extra_inner,
                mask=extra_mask,
            )
            tl.store(out_extra_ptr + offsets, extra, mask=extra_mask)

    @triton.jit(do_not_specialize=["B"])
    def _join_query_outputs_kernel(
        outputs,
        lses,
        out_ptr,
        lse_ptr,
        B,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        QUERIES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        query_id = tl.program_id(1)
        output_mask = offsets < B * HEADS * DIM
        output_offsets = (
            (offsets // (HEADS * DIM) * QUERIES + query_id) * HEADS * DIM
            + offsets % (HEADS * DIM)
        )
        lse_mask = offsets < B * HEADS
        for query in tl.static_range(QUERIES):
            if query_id == query:
                value = tl.load(outputs[query] + offsets, mask=output_mask)
                tl.store(out_ptr + output_offsets, value, mask=output_mask)
                lse = tl.load(lses[query] + offsets, mask=lse_mask)
                tl.store(lse_ptr + offsets * QUERIES + query_id, lse, mask=lse_mask)
else:
    _copy_single_query_kernel = None
    _join_query_outputs_kernel = None


def can_fuse_query_copy(q, indices, extra_indices=None):
    """Leave zero-copy B1, CPU and unsupported layouts on the native path."""
    if (
        triton is None
        or q.ndim != 4
        or q.shape[0] <= 1
        or q.shape[1] <= 1
        or q.dtype != torch.bfloat16
        or not q.is_cuda
        or q.shape[2] == 0
        or q.shape[3] == 0
        or q.stride(-1) != 1
        or q.stride(-2) != q.shape[-1]
        or q.stride(1) != q.shape[2] * q.shape[3]
        or indices is None
    ):
        return False
    return all(
        value.ndim == 3
        and value.shape[:2] == q.shape[:2]
        and value.shape[2] > 0
        and value.dtype == torch.int32
        and value.device == q.device
        and value.stride(-1) == 1
        and value.stride(1) == value.shape[2]
        for value in (indices, extra_indices)
        if value is not None
    )


def allocate_query_scratch(q, indices, extra_indices=None):
    """Allocate one reusable [B,1,...] scratch set, not S full copies."""
    if q.ndim != 4 or indices.ndim != 3:
        raise ValueError("q/indices must be [B,S,H,D] and [B,S,K]")
    if q.shape[:2] != indices.shape[:2]:
        raise ValueError("q and indices must have the same B/S dimensions")
    if extra_indices is not None and extra_indices.shape[:2] != q.shape[:2]:
        raise ValueError("extra indices must have the same B/S dimensions")
    return (
        torch.empty((q.shape[0], 1, *q.shape[2:]), dtype=q.dtype, device=q.device),
        torch.empty(
            (indices.shape[0], 1, indices.shape[2]),
            dtype=indices.dtype,
            device=indices.device,
        ),
        None
        if extra_indices is None
        else torch.empty(
            (extra_indices.shape[0], 1, extra_indices.shape[2]),
            dtype=extra_indices.dtype,
            device=extra_indices.device,
        ),
    )


def copy_query_into_scratch(q, indices, extra_indices, query, scratch):
    """Copy q and both index rows with one launch, preserving every bit."""
    if triton is None:
        raise RuntimeError("Triton is required by fused FlashMLA query copy")
    out_q, out_indices, out_extra = scratch
    if not (q.is_cuda and indices.is_cuda):
        raise ValueError("fused FlashMLA query copy requires CUDA tensors")
    if q.dtype != torch.bfloat16 or indices.dtype != torch.int32:
        raise ValueError("fused FlashMLA query copy requires BF16 q and int32 indices")
    if not 0 <= int(query) < q.shape[1]:
        raise ValueError("query index is out of range")
    q_row = q.shape[2] * q.shape[3]
    if (
        q.stride(-1) != 1
        or q.stride(-2) != q.shape[-1]
        or q.stride(1) != q_row
    ):
        raise ValueError("q query rows must be contiguous")
    if indices.stride(-1) != 1 or indices.stride(1) != indices.shape[2]:
        raise ValueError("indices query rows must be contiguous")
    if extra_indices is not None:
        if (
            extra_indices.dtype != torch.int32
            or extra_indices.stride(-1) != 1
            or extra_indices.stride(1) != extra_indices.shape[2]
        ):
            raise ValueError("extra indices query rows must be contiguous int32")
        if extra_indices.device != q.device:
            raise ValueError("extra indices must be on q's device")
    if indices.device != q.device:
        raise ValueError("indices must be on q's device")
    expected = (
        (q.shape[0], 1, *q.shape[2:]),
        (indices.shape[0], 1, indices.shape[2]),
        None
        if extra_indices is None
        else (extra_indices.shape[0], 1, extra_indices.shape[2]),
    )
    for actual, shape in zip(scratch, expected):
        if (actual is None) != (shape is None):
            raise ValueError("scratch presence does not match extra indices")
        if actual is not None and (actual.shape != shape or not actual.is_contiguous()):
            raise ValueError("invalid fused FlashMLA query scratch")
    if out_q.dtype != q.dtype or out_q.device != q.device:
        raise ValueError("q scratch dtype/device mismatch")
    if out_indices.dtype != indices.dtype or out_indices.device != q.device:
        raise ValueError("index scratch dtype/device mismatch")
    if out_extra is not None and (
        out_extra.dtype != extra_indices.dtype or out_extra.device != q.device
    ):
        raise ValueError("extra scratch dtype/device mismatch")

    topk = indices.shape[2]
    extra_topk = 0 if extra_indices is None else extra_indices.shape[2]
    count = max(q.shape[0] * q_row, q.shape[0] * topk, q.shape[0] * extra_topk)
    dummy_in = indices if extra_indices is None else extra_indices
    dummy_out = out_indices if out_extra is None else out_extra
    _copy_single_query_kernel[(triton.cdiv(count, 256),)](
        q,
        indices,
        dummy_in,
        out_q,
        out_indices,
        dummy_out,
        q.shape[0],
        int(query),
        q.stride(0),
        q.stride(1),
        indices.stride(0),
        indices.stride(1),
        dummy_in.stride(0),
        dummy_in.stride(1),
        Q_ROW=q_row,
        TOPK=topk,
        EXTRA_TOPK=extra_topk,
        HAS_EXTRA=extra_indices is not None,
        BLOCK=256,
        num_warps=4,
    )
    return scratch


def join_query_outputs(outputs, lses):
    """Assemble BF16 output and FP32 LSE in one launch, without arithmetic."""
    if triton is None or len(outputs) < 2 or len(lses) != len(outputs):
        return None
    first = outputs[0]
    if (
        first.ndim != 4
        or first.shape[1] != 1
        or first.dtype != torch.bfloat16
        or not first.is_cuda
        or any(size == 0 for size in first.shape)
    ):
        return None
    batch, _, heads, dim = first.shape
    if not all(
        value.shape == first.shape
        and value.dtype == first.dtype
        and value.device == first.device
        and value.is_contiguous()
        for value in outputs
    ) or not all(
        value.shape == (batch, heads, 1)
        and value.dtype == torch.float32
        and value.device == first.device
        and value.is_contiguous()
        for value in lses
    ):
        return None
    queries = len(outputs)
    output = torch.empty(
        (batch, queries, heads, dim), dtype=first.dtype, device=first.device
    )
    lse = torch.empty((batch, heads, queries), dtype=lses[0].dtype, device=first.device)
    _join_query_outputs_kernel[(triton.cdiv(batch * heads * dim, 1024), queries)](
        tuple(outputs),
        tuple(lses),
        output,
        lse,
        batch,
        HEADS=heads,
        DIM=dim,
        QUERIES=queries,
        BLOCK=1024,
        num_warps=4,
    )
    return output, lse


__all__ = [
    "can_fuse_query_copy",
    "allocate_query_scratch",
    "copy_query_into_scratch",
    "join_query_outputs",
]

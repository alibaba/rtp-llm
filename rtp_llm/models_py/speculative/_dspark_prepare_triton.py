"""Forward-local DSpark integer preparation; unsupported inputs keep eager math.

The row mapper implements searchsorted(right=True), including duplicate starts.
The proposal preparer preserves the int32 cast before position arithmetic.
Neither helper reads device values on the host or retains per-forward tensors.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _dspark_query_inputs_kernel(
    ids,
    prefixes,
    query_ids,
    query_positions,
    prefix32,
    active,
    B: tl.constexpr,
    WIDTH: tl.constexpr,
    NOISE: tl.constexpr,
    STORE_PREFIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    request = row // WIDTH
    column = row % WIDTH
    valid = request < B
    # Match Tensor.to(int32), then Tensor.to(int64), in the eager path.
    prefix = tl.load(prefixes + request, valid, 0).to(tl.int32)
    anchor = tl.load(ids + request * WIDTH, valid & (column == 0), 0).to(tl.int32)
    tl.store(query_ids + row, tl.where(column == 0, anchor, NOISE), valid)
    tl.store(query_positions + row, prefix.to(tl.int64) + column, valid)
    first = valid & (column == 0)
    tl.store(active + request, prefix > 0, first)
    if STORE_PREFIX:
        tl.store(prefix32 + request, prefix, first)


@triton.jit
def _dspark_commit_rows_kernel(
    starts,
    lengths,
    ends,
    requests,
    positions,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = row < ROWS
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), B, tl.int32)
    for _ in tl.static_range(STEPS):
        searching = lo < hi
        mid = (lo + hi) // 2
        value = tl.load(starts + mid, mask & searching & (mid < B), 0).to(tl.int64)
        right = value <= row
        lo = tl.where(searching & right, mid + 1, lo)
        hi = tl.where(searching & ~right, mid, hi)
    request = lo - 1
    safe = tl.minimum(tl.maximum(request, 0), B - 1)
    start = tl.load(starts + safe, mask, 0).to(tl.int64)
    length = tl.load(lengths + safe, mask, 0).to(tl.int64)
    end = tl.load(ends + safe, mask, 0).to(tl.int64)
    valid = (request >= 0) & (row >= start) & (row < start + length)
    position = end - length + row - start
    tl.store(requests + row, tl.where(valid, request, -1), mask)
    tl.store(positions + row, tl.where(valid, position, -1), mask)


def _integers_on_device(tensors):
    first = tensors[0]
    return first.device.type == "cuda" and all(
        t.device == first.device
        and t.dtype in (torch.int32, torch.int64)
        and t.is_contiguous()
        for t in tensors
    )


def try_prepare_proposal(input_ids, prefix_lengths, *, width, noise_token_id):
    """Return (ids[B,W], positions[B,W], prefix[B], active[B]) or None."""
    if (
        os.environ.get("DSV4_FUSED_DSPARK_QUERY_INPUTS", "1") == "0"
        or width <= 0
        or input_ids.numel() % width
        or input_ids.numel() >= 2**31
        or prefix_lengths.ndim != 1
        or not -(2**31) <= noise_token_id < 2**31
        or not _integers_on_device((input_ids, prefix_lengths))
    ):
        return None
    batch = input_ids.numel() // width
    if prefix_lengths.numel() < batch:
        return None
    device = input_ids.device
    query_ids = torch.empty((batch, width), dtype=torch.int32, device=device)
    positions = torch.empty((batch, width), dtype=torch.int64, device=device)
    active = torch.empty(batch, dtype=torch.bool, device=device)
    prefix32 = (
        prefix_lengths[:batch]
        if prefix_lengths.dtype == torch.int32
        else torch.empty(batch, dtype=torch.int32, device=device)
    )
    if batch:
        with torch.cuda.device(device):
            _dspark_query_inputs_kernel[(triton.cdiv(batch * width, 256),)](
                input_ids,
                prefix_lengths,
                query_ids,
                positions,
                prefix32,
                active,
                B=batch,
                WIDTH=width,
                NOISE=int(noise_token_id),
                STORE_PREFIX=prefix_lengths.dtype != torch.int32,
                BLOCK=256,
                num_warps=4,
            )
    return query_ids, positions, prefix32, active


def try_map_context_rows(starts, lengths, committed_ends, row_count):
    """Exact non-CP packed/dense row mapping; starts must be sorted ascending.

    Duplicate starts select the last equal entry, as in torch.searchsorted.
    Inputs belong to this forward; the returned int32 tensors never alias them.
    """
    if (
        os.environ.get("DSV4_FUSED_DSPARK_COMMIT_ROWS", "1") == "0"
        or row_count < 0
        or row_count >= 2**31
        or starts.numel() >= 2**30
        or any(t.dtype != torch.int64 for t in (starts, lengths, committed_ends))
        or starts.ndim != 1
        or lengths.shape != starts.shape
        or committed_ends.shape != starts.shape
        or not _integers_on_device((starts, lengths, committed_ends))
    ):
        return None
    batch = starts.numel()
    if not batch and row_count:
        return None
    requests = torch.empty(row_count, dtype=torch.int32, device=starts.device)
    positions = torch.empty_like(requests)
    if row_count:
        with torch.cuda.device(starts.device):
            _dspark_commit_rows_kernel[(triton.cdiv(row_count, 256),)](
                starts,
                lengths,
                committed_ends,
                requests,
                positions,
                B=batch,
                ROWS=row_count,
                STEPS=(batch + 1).bit_length(),
                BLOCK=256,
                num_warps=4,
            )
    return requests, positions

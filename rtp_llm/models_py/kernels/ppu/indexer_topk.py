"""Exact FP32 score ordering with deterministic token-index tie breaking."""

import torch
import triton
import triton.language as tl


@triton.jit
def _encode_keys(
    SCORES,
    STARTS,
    ENDS,
    KEYS,
    N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start, end = tl.load(STARTS + row), tl.load(ENDS + row)
    valid = (col < N) & (col >= start) & (col < end)
    score = tl.load(SCORES + row * S0 + col * S1, valid, 0.0)
    # Normalize signed zero, then map signed IEEE bits to monotone signed ints.
    bits = score.to(tl.int32, bitcast=True)
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    key = (ordered << 32) | (0xFFFFFFFF - col.to(tl.int64))
    key = tl.where(valid, key, -9223372036854775808)
    tl.store(KEYS + row * N + col, key, col < N)


@triton.jit
def _finish(
    INDICES,
    VALUES,
    STARTS,
    OUTPUT,
    K: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    selected = col < COUNT
    index = tl.load(INDICES + row * COUNT + col, selected, 0)
    value = tl.load(VALUES + row * COUNT + col, selected, -9223372036854775808)
    start = tl.load(STARTS + row)
    index = tl.where(
        selected & (value != -9223372036854775808), index - start, 2147483647
    ).to(tl.int32)
    index = tl.sort(index, descending=False)
    index = tl.where(index == 2147483647, -1, index)
    tl.store(OUTPUT + row * K + col, index, col < K)


@triton.jit
def _ordered_uint(score):
    bits = score.to(tl.int32, bitcast=True)
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.uint32) ^ 0x80000000


@triton.jit
def _stable_topk_small_kernel(
    SCORES,
    STARTS,
    ENDS,
    OUTPUT,
    N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    K: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    start, end = tl.load(STARTS + row), tl.load(ENDS + row)
    begin = tl.maximum(start, 0)
    count = tl.maximum(tl.minimum(end, N) - begin, 0)
    if count <= K:
        col = tl.arange(0, K)
        tl.store(OUTPUT + row * K + col, tl.where(col < count, begin + col - start, -1))
    else:
        ia = tl.arange(0, A)
        va = (ia < N) & (ia >= start) & (ia < end)
        ka = _ordered_uint(tl.load(SCORES + row * S0 + ia * S1, va, 0.0))
        if N > A:
            ib = A + tl.arange(0, B)
            vb = (ib < N) & (ib >= start) & (ib < end)
            kb = _ordered_uint(tl.load(SCORES + row * S0 + ib * S1, vb, 0.0))
        threshold = tl.full((), 0, tl.uint32)
        bit = tl.full((), 0x80000000, tl.uint32)
        for _ in range(32):
            candidate = threshold | bit
            above = tl.sum((va & (ka >= candidate)).to(tl.int32), 0)
            if N > A:
                above += tl.sum((vb & (kb >= candidate)).to(tl.int32), 0)
            threshold = tl.where(above >= K, candidate, threshold)
            bit = bit >> 1
        aa, ea = va & (ka > threshold), va & (ka == threshold)
        above_count = tl.sum(aa.to(tl.int32), 0)
        if N > A:
            ab, eb = vb & (kb > threshold), vb & (kb == threshold)
            above_count += tl.sum(ab.to(tl.int32), 0)
        sa = aa | (ea & (tl.cumsum(ea.to(tl.int32), 0) <= K - above_count))
        pa = tl.cumsum(sa.to(tl.int32), 0) - 1
        tl.store(OUTPUT + row * K + pa, ia - start, sa)
        if N > A:
            equal_prefix = tl.sum(ea.to(tl.int32), 0)
            sb = ab | (
                eb & (equal_prefix + tl.cumsum(eb.to(tl.int32), 0) <= K - above_count)
            )
            pb = tl.sum(sa.to(tl.int32), 0) + tl.cumsum(sb.to(tl.int32), 0) - 1
            tl.store(OUTPUT + row * K + pb, ib - start, sb)


def _stable_topk_small(logits, starts, ends, output):
    # A power-of-two prefix plus a small tail avoids doubling register lanes
    # at the production 34816-column width. Exact FP32 keys resolve all ties.
    rows, columns = logits.shape
    a = 1 << (columns.bit_length() - 1)
    b = triton.next_power_of_2(max(columns - a, 1))
    if rows:
        _stable_topk_small_kernel[(rows,)](
            logits,
            starts,
            ends,
            output,
            columns,
            logits.stride(0),
            logits.stride(1),
            2048,
            a,
            b,
            num_warps=16,
        )


@triton.jit(do_not_specialize=["COLUMNS"])
def _all_indices(
    STARTS,
    ENDS,
    OFFSETS,
    OUTPUT,
    COLUMNS,
    K: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    start, end = tl.load(STARTS + row), tl.load(ENDS + row)
    begin = tl.maximum(start, 0)
    count = tl.maximum(tl.minimum(end, COLUMNS) - begin, 0)
    offset = tl.load(OFFSETS + row) if HAS_OFFSETS else 0
    index = tl.where(col < count, begin + col - start + offset, -1)
    tl.store(OUTPUT + row * K + col, index.to(tl.int32), col < K)


def all_token_indices(row_starts, row_ends, output, columns, offsets=None):
    """Emit canonical indices when every row's valid keys fit in TopK.

    The caller must establish that bound from CPU sequence metadata. Offsets
    match the ragged CP workspace mapping; no host read of row lengths occurs.
    """
    rows, topk = output.shape
    if rows and topk:
        _all_indices[(rows,)](
            row_starts,
            row_ends,
            output if offsets is None else offsets,
            output,
            columns,
            topk,
            offsets is not None,
            triton.next_power_of_2(topk),
            num_warps=4,
        )


def stable_topk(logits, row_starts, row_ends, output, max_seq_len=None):
    """Write request-local indices; equal scores prefer smaller token indices.

    Integer keys retain every FP32 score bit. No epsilon is added to scores.
    The selected indices are sorted in token order with trailing -1 padding.
    """
    rows, columns = logits.shape
    count = min(output.shape[1], columns)
    if rows == 0 or count == 0:
        output.fill_(-1)
        return
    if columns <= output.shape[1] or (
        max_seq_len is not None and 0 <= max_seq_len <= output.shape[1]
    ):
        all_token_indices(row_starts, row_ends, output, columns)
        return
    if rows <= 64 and 8192 <= columns <= 34816 and output.shape[1] == 2048:
        # Per-row device bounds skip score selection for short decode rows,
        # including when one Graph bucket also serves longer rows. The exact
        # threshold and token-order tie break stay on device for long rows.
        # Wider contexts and larger row chunks keep the original 64-bit path.
        _stable_topk_small(logits, row_starts, row_ends, output)
        return
    keys = torch.empty((rows, columns), dtype=torch.int64, device=logits.device)
    _encode_keys[(rows, triton.cdiv(columns, 256))](
        logits,
        row_starts,
        row_ends,
        keys,
        columns,
        logits.stride(0),
        logits.stride(1),
        256,
    )
    values, indices = torch.topk(keys, count, dim=-1, sorted=False)
    _finish[(rows,)](
        indices,
        values,
        row_starts,
        output,
        output.shape[1],
        count,
        triton.next_power_of_2(output.shape[1]),
        num_warps=4,
    )

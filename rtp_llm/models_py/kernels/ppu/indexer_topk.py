"""Exact FP32 score ordering with deterministic token-index tie breaking."""

import torch
import triton
import triton.language as tl


@triton.jit
def _encode_keys(SCORES, STARTS, ENDS, KEYS, N: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, BLOCK: tl.constexpr):
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
def _finish(INDICES, VALUES, STARTS, OUTPUT, K: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    selected = col < COUNT
    index = tl.load(INDICES + row * COUNT + col, selected, 0)
    value = tl.load(VALUES + row * COUNT + col, selected, -9223372036854775808)
    start = tl.load(STARTS + row)
    index = tl.where(selected & (value != -9223372036854775808), index - start, -1)
    tl.store(OUTPUT + row * K + col, index.to(tl.int32), col < K)


def stable_topk(logits, row_starts, row_ends, output):
    """Write request-local indices; equal scores prefer smaller token indices.

    Integer keys retain every FP32 score bit. No epsilon is added to scores.
    The caller canonicalizes the selected indices for sparse-attention order.
    """
    rows, columns = logits.shape
    count = min(output.shape[1], columns)
    if rows == 0 or count == 0:
        output.fill_(-1)
        return
    keys = torch.empty((rows, columns), dtype=torch.int64, device=logits.device)
    _encode_keys[(rows, triton.cdiv(columns, 256))](
        logits, row_starts, row_ends, keys, columns, logits.stride(0), logits.stride(1), 256,
    )
    values, indices = torch.topk(keys, count, dim=-1, sorted=False)
    _finish[(rows, triton.cdiv(output.shape[1], 256))](
        indices, values, row_starts, output, output.shape[1], count, 256,
    )

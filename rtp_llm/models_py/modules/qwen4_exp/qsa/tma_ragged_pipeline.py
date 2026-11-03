"""Packed prefill over request-local QSA selections and validated CU metadata."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .tma_pipeline import _KV_LAYOUT, _execute_prefill


@triton.jit(do_not_specialize=["B", "WORDS"])
def _build_ragged_membership(Ids, Mask, Cu, B, WORDS, MAX_WORDS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    low = 0
    high = B
    while low < high:
        mid = (low + high) // 2
        if row >= tl.load(Cu + mid + 1):
            low = mid + 1
        else:
            high = mid
    local_s = tl.load(Cu + low + 1) - tl.load(Cu + low)
    word = tl.arange(0, MAX_WORDS)
    tl.store(Mask + row * WORDS + word, 0, word < WORDS)
    tl.debug_barrier()
    block = tl.arange(0, 512)
    start = tl.load(Ids + row * 2051 + block * 4)
    bits = tl.full((512,), 15, tl.uint32) << (start % 32)
    tl.atomic_or(
        Mask + row * WORDS + start // 32,
        bits,
        (start >= 0) & (start + 3 < local_s),
        sem="relaxed",
    )
    tail = tl.arange(0, 4)
    index = tl.load(Ids + row * 2051 + 2048 + tail, tail < 3, other=-1)
    tl.atomic_or(
        Mask + row * WORDS + index // 32,
        tl.full((4,), 1, tl.uint32) << (index % 32),
        (tail < 3) & (index >= 0) & (index < local_s),
        sem="relaxed",
    )


@g.jit
def _add(a, b):
    return a + b


@g.jit
def _locate(Cu, B, POSITIONS: gl.constexpr, G: gl.constexpr):
    layout: gl.constexpr = BlockedLayout([1], [32], [4], [0])
    b = gl.arange(0, 32, layout=layout)
    starts = gl.load(Cu + b, b < B, other=0)
    ends = gl.load(Cu + b + 1, b < B, other=0)
    lengths = ends - starts
    counts = gl.cdiv(lengths, POSITIONS)
    cumulative = gl.associative_scan(counts, 0, _add)
    work = gl.num_programs(1 if G == 4 else 0) - 1 - gl.program_id(1 if G == 4 else 0)
    request = gl.sum(((work >= cumulative) & (b < B)).to(gl.int32), 0)
    base = gl.sum(gl.where(b == request, starts, 0), 0)
    length = gl.sum(gl.where(b == request, lengths, 0), 0)
    first = gl.sum(gl.where(b == request, cumulative - counts, 0), 0)
    return base, length, work - first


@g.jit(do_not_specialize=["B", "WORDS"])
def _packed_prefill(
    Q,
    K,
    V,
    Mask,
    O,
    Cu,
    B,
    WORDS,
    G: gl.constexpr,
    QM: gl.constexpr,
    QH: gl.constexpr,
    QS: gl.constexpr,
    OS: gl.constexpr,
    PIPE: gl.constexpr,
):
    positions: gl.constexpr = QM // G
    base, length, tile = _locate(Cu, B, positions, G)
    _execute_prefill(
        Q,
        K,
        V,
        Mask,
        O,
        length,
        WORDS,
        256,
        G,
        QM,
        32,
        positions,
        QH,
        QS,
        256,
        OS,
        PIPE,
        QM // 16,
        QM == 32,
        base,
        tile,
        2 if G != 6 else 1,
    )


def run_prefill(q, k, v, selected, cu, lengths):
    _, heads, tokens, dim = q.shape
    longest = max(lengths)
    qm, group = (
        (16, 12) if heads == 12 and longest <= 512 else (32, 4 if heads == 12 else 6)
    )
    words = triton.cdiv(longest, 32)
    membership = torch.empty((tokens, words), dtype=torch.int32, device=q.device)
    _build_ragged_membership[(tokens,)](
        selected,
        membership,
        cu,
        len(lengths),
        words,
        16 if longest <= 512 else 128,
        num_warps=4,
    )
    output = torch.empty(
        (1, tokens, heads, dim), dtype=q.dtype, device=q.device
    ).transpose(1, 2)
    key = TensorDescriptor(k, [tokens, dim], [k.stride(2), 1], [32, dim], _KV_LAYOUT)
    value = TensorDescriptor(v, [tokens, dim], [v.stride(2), 1], [32, dim], _KV_LAYOUT)
    positions = qm // group
    # Cover every positive partition with the same total, including graph replays.
    # Surplus work resolves to an empty request and performs no Q/K/V reads.
    tiles = len(lengths) + (tokens - len(lengths)) // positions
    grid = (tiles, heads // group)
    if group == 4:
        grid = grid[::-1]
    _packed_prefill[grid](
        q,
        key,
        value,
        membership,
        output,
        cu,
        len(lengths),
        words,
        group,
        qm,
        q.stride(1),
        q.stride(2),
        output.stride(2),
        heads == 6,
        num_warps=4,
    )
    return output.transpose(1, 2).reshape(tokens, heads * dim)

"""Batched prefill from request-local page tables and validated GPU metadata."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, NVMMASharedLayout
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .tma_paged_pipeline import _execute_prefill


@triton.jit(do_not_specialize=["B", "WORDS"])
def _build_paged_membership(Ids, Mask, Cu, Prefixes, B, WORDS, MAX_WORDS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    low = 0
    high = B
    while low < high:
        mid = (low + high) // 2
        if row >= tl.load(Cu + mid + 1):
            low = mid + 1
        else:
            high = mid
    local_s = tl.load(Prefixes + low) + tl.load(Cu + low + 1) - tl.load(Cu + low)
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
    return request, base, length, work - first


@g.jit(do_not_specialize=["B", "WORDS", "WIDTH", "PAGES"])
def _batched_paged_prefill(
    Q,
    Cache,
    Table,
    Scales,
    Mask,
    O,
    Cu,
    Prefixes,
    B,
    WORDS,
    WIDTH,
    PAGES,
    TABLE_ROW: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    SCALE_PAGE: gl.constexpr,
    SCALE_KV: gl.constexpr,
    FP8: gl.constexpr,
    CACHE_ROWS: gl.constexpr,
    G: gl.constexpr,
    QM: gl.constexpr,
    QS: gl.constexpr,
    OS: gl.constexpr,
    PIPE: gl.constexpr,
):
    positions: gl.constexpr = QM // G
    request, base, length, tile = _locate(Cu, B, positions, G)
    prefix = gl.load(Prefixes + request)
    _execute_prefill(
        Q + base.to(gl.int64) * QS,
        Cache,
        Cache,
        Mask + base.to(gl.int64) * WORDS,
        O + base.to(gl.int64) * OS,
        length,
        WORDS,
        256,
        G,
        QM,
        32,
        positions,
        256,
        QS,
        256,
        OS,
        PIPE,
        QM // 16,
        True,
        tile,
        Table + request.to(gl.int64) * TABLE_ROW,
        prefix,
        WIDTH,
        PAGES,
        Scales,
        PAGE_SIZE,
        SCALE_PAGE,
        SCALE_KV,
        FP8,
        CACHE_ROWS,
    )


def run_prefill(
    q, cache, table, selected, cu, prefixes, lengths, prefix_host, page_size, scale
):
    tokens, width = q.shape
    heads = width // 256
    dim = 256
    longest = max(lengths)
    bound = max(prefix_host) + longest
    qm, group = (
        (16, 12) if heads == 12 and longest <= 512 else (32, 4 if heads == 12 else 6)
    )
    words = triton.cdiv(bound, 32)
    membership = torch.empty((tokens, words), dtype=torch.int32, device=q.device)
    _build_paged_membership[(tokens,)](
        selected,
        membership,
        cu,
        prefixes,
        len(lengths),
        words,
        16 if bound <= 512 else 128,
        num_warps=4,
    )
    fp8 = cache.dtype == torch.float8_e4m3fn
    cache_rows = cache.stride(0) // dim
    descriptor = TensorDescriptor(
        cache,
        [cache.shape[0] * cache_rows, dim],
        [dim, 1],
        [32, dim],
        NVMMASharedLayout(128, 8 if fp8 else 16),
    )
    output = torch.empty((tokens, width), dtype=q.dtype, device=q.device)
    tiles = sum(triton.cdiv(s, qm // group) for s in lengths)
    grid = (tiles, heads // group)
    if group == 4:
        grid = grid[::-1]
    _batched_paged_prefill[grid](
        q,
        descriptor,
        table,
        cache if scale is None else scale,
        membership,
        output,
        cu,
        prefixes,
        len(lengths),
        words,
        table.shape[1],
        cache.shape[0],
        table.stride(0),
        page_size,
        0 if scale is None else scale.stride(0),
        0 if scale is None else scale.stride(1),
        fp8,
        cache_rows,
        group,
        qm,
        q.stride(0),
        output.stride(0),
        (not fp8 or heads == 6) and bound <= 2048,
        num_warps=4,
    )
    return output

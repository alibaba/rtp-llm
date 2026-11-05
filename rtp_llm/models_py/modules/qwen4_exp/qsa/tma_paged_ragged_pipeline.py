"""Batched prefill from request-local page tables and validated GPU metadata."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import NVMMASharedLayout
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .membership import _find_request, _write_membership
from .ragged_metadata import _locate
from .tma_paged_pipeline import _execute_prefill


@triton.jit(do_not_specialize=["B", "WORDS"])
def _build_paged_membership(Ids, Mask, Cu, Prefixes, B, WORDS, MAX_WORDS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    request = _find_request(Cu, row, B)
    local_s = (
        tl.load(Prefixes + request) + tl.load(Cu + request + 1) - tl.load(Cu + request)
    )
    _write_membership(Ids, Mask, row, local_s, WORDS, MAX_WORDS)


@g.jit(do_not_specialize=["B", "WORDS", "WIDTH", "PAGES", "TABLE_ROW"])
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
    TABLE_ROW,
    PAGE_SIZE: gl.constexpr,
    SCALE_PAGE: gl.constexpr,
    SCALE_KV: gl.constexpr,
    FP8: gl.constexpr,
    SCALED: gl.constexpr,
    CACHE_ROWS: gl.constexpr,
    G: gl.constexpr,
    QM: gl.constexpr,
    QS: gl.constexpr,
    OS: gl.constexpr,
    PIPE: gl.constexpr,
    FRONTIER: gl.constexpr,
):
    positions: gl.constexpr = QM // G
    request, base, length, tile = _locate(Cu, B, positions, G)
    prefix = gl.load(Prefixes + request, request < B, other=0)
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
        SCALED,
        CACHE_ROWS,
        FRONTIER,
    )


def run_prefill(
    q, cache, table, selected, cu, prefixes, lengths, prefix_host, page_size, scale
):
    tokens, width = q.shape
    heads = width // 256
    dim = 256
    longest = max(lengths)
    bound = max(prefix_host) + longest
    scaled = cache.dtype == torch.float8_e4m3fn
    qm, group = (
        (16, 12) if heads == 12 and longest <= 512 else (32, 4 if heads == 12 else 6)
    )
    words = triton.cdiv(bound, 32)
    membership = torch.empty((tokens, words), dtype=torch.int32, device=q.device)
    if scaled and bound >= 1024:
        from .prefill_staging import stage_paged_kv

        cache, table, scale = stage_paged_kv(
            cache, table, scale, bound, page_size, cu, prefixes, selected, membership
        )
    else:
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
    # Positive partitions of a fixed token count can round to different tile counts.
    # Bound all such partitions so replay can change CU contents within the plan.
    tiles = len(lengths) + (tokens - len(lengths)) // (qm // group)
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
        scaled,
        cache_rows,
        group,
        qm,
        q.stride(0),
        output.stride(0),
        (not scaled or heads == 6) and bound <= 2048,
        heads == 12 or bound > 2048,
        num_warps=4,
    )
    return output

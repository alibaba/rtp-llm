"""Query-tiled Q8K4 prefill block scores over RTP's packed NVFP4 ABI."""

import torch
import triton
import triton.language as tl

from ..decode.nvfp4_q8_index_score import _load_index_page_fp8

# These values change with live request shapes, not the kernel's tile geometry.
# Runtime scalars avoid compiling a new IndexScore binary for every P step.
_DYNAMIC_GEOMETRY_PARAMETERS = (
    "TOTAL_Q",
    "MAX_PAGES",
    "PHYSICAL_PAGES",
    "TABLE_SIZE",
    "OSTRIDE_H",
    "OSTRIDE_Q",
)


@triton.jit(do_not_specialize=_DYNAMIC_GEOMETRY_PARAMETERS)
def _prefill_score_kernel(
    Q,
    K,
    S,
    CUQ,
    KLENS,
    PREFIX,
    CUPAGES,
    PAGES,
    OUT,
    TOTAL_Q,
    HEADS: tl.constexpr,
    MAX_PAGES,
    PHYSICAL_PAGES,
    TABLE_SIZE,
    QSTRIDE: tl.constexpr,
    HSTRIDE: tl.constexpr,
    KSTRIDE: tl.constexpr,
    SSTRIDE: tl.constexpr,
    OSTRIDE_H,
    OSTRIDE_Q,
    TILE_Q: tl.constexpr,
):
    tile = tl.program_id(0)
    segment_head = tl.program_id(1)
    block = tl.program_id(2)
    segment = segment_head // HEADS
    head = segment_head % HEADS
    start = tl.load(CUQ + segment)
    stop = tl.load(CUQ + segment + 1)
    local_row = tile * TILE_Q + tl.arange(0, TILE_Q)
    row = start + local_row
    row_ok = (row < stop) & (row >= 0) & (row < TOTAL_Q)
    if tile * TILE_Q >= stop - start:
        return
    length = tl.load(KLENS + segment)
    prefix = tl.load(PREFIX + segment)
    page_start = tl.load(CUPAGES + segment)
    page_end = tl.load(CUPAGES + segment + 1)
    page_offset = page_start + block
    logical_ok = (block * 128 < length) & (page_offset < page_end)
    physical = tl.load(
        PAGES + page_offset,
        mask=logical_ok & (page_offset >= 0) & (page_offset < TABLE_SIZE),
        other=-1,
    ).to(tl.int64)
    page_ok = logical_ok & (physical >= 0) & (physical < PHYSICAL_PAGES)
    d = tl.arange(0, 128)
    query = tl.load(
        Q
        + row[:, None].to(tl.int64) * QSTRIDE
        + head.to(tl.int64) * HSTRIDE
        + d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    key = _load_index_page_fp8(
        K, S, tl.where(page_ok, physical, 0), page_ok, KSTRIDE, SSTRIDE, True
    )
    dot = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
    position = block * 128 + tl.arange(0, 128)
    visible = (
        page_ok
        & row_ok[:, None]
        & (position[None, :] < length)
        & (position[None, :] <= prefix + local_row[:, None])
    )
    score = tl.max(tl.where(visible, dot, float("-inf")), axis=1)
    tl.store(
        OUT + head.to(tl.int64) * OSTRIDE_H + row.to(tl.int64) * OSTRIDE_Q + block,
        score,
        mask=row_ok,
    )


@triton.jit(do_not_specialize=_DYNAMIC_GEOMETRY_PARAMETERS)
def _prefill_two_page_score_kernel(
    Q,
    K,
    S,
    CUQ,
    KLENS,
    PREFIX,
    CUPAGES,
    PAGES,
    OUT,
    TOTAL_Q,
    HEADS: tl.constexpr,
    MAX_PAGES,
    PHYSICAL_PAGES,
    TABLE_SIZE,
    QSTRIDE: tl.constexpr,
    HSTRIDE: tl.constexpr,
    KSTRIDE: tl.constexpr,
    SSTRIDE: tl.constexpr,
    OSTRIDE_H,
    OSTRIDE_Q,
    TILE_Q: tl.constexpr,
):
    # Reuse one Q tile across two independent 128-token page reductions.
    # Keeping the MMA width unchanged preserves Q8K4 rounding semantics.
    tile = tl.program_id(0)
    segment_head = tl.program_id(1)
    first_block = tl.program_id(2) * 2
    segment = segment_head // HEADS
    head = segment_head % HEADS
    start = tl.load(CUQ + segment)
    stop = tl.load(CUQ + segment + 1)
    local_row = tile * TILE_Q + tl.arange(0, TILE_Q)
    row = start + local_row
    row_ok = (row < stop) & (row >= 0) & (row < TOTAL_Q)
    if tile * TILE_Q >= stop - start:
        return
    length = tl.load(KLENS + segment)
    prefix = tl.load(PREFIX + segment)
    page_start = tl.load(CUPAGES + segment)
    page_end = tl.load(CUPAGES + segment + 1)
    d = tl.arange(0, 128)
    query = tl.load(
        Q
        + row[:, None].to(tl.int64) * QSTRIDE
        + head.to(tl.int64) * HSTRIDE
        + d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    for page in tl.static_range(2):
        block = first_block + page
        page_offset = page_start + block
        logical_ok = (
            (block * 128 < length) & (page_offset < page_end) & (block < MAX_PAGES)
        )
        physical = tl.load(
            PAGES + page_offset,
            mask=logical_ok & (page_offset >= 0) & (page_offset < TABLE_SIZE),
            other=-1,
        ).to(tl.int64)
        page_ok = logical_ok & (physical >= 0) & (physical < PHYSICAL_PAGES)
        # A page beyond the last live query is invisible to this entire tile.
        # Avoid packed-K decoding and MMA, but still overwrite its score rows
        # so reused chunk/graph buffers cannot inherit a previous launch.
        last_query = prefix + tl.minimum((tile + 1) * TILE_Q, stop - start) - 1
        if page_ok & (block * 128 <= last_query):
            key = _load_index_page_fp8(
                K,
                S,
                physical,
                page_ok,
                KSTRIDE,
                SSTRIDE,
                True,
            )
            dot = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
            position = block * 128 + tl.arange(0, 128)
            visible = (
                row_ok[:, None]
                & (position[None, :] < length)
                & (position[None, :] <= prefix + local_row[:, None])
            )
            score = tl.max(tl.where(visible, dot, float("-inf")), axis=1)
        else:
            score = tl.full((TILE_Q,), float("-inf"), tl.float32)
        tl.store(
            OUT + head.to(tl.int64) * OSTRIDE_H + row.to(tl.int64) * OSTRIDE_Q + block,
            score,
            mask=row_ok & (block < MAX_PAGES),
        )


@triton.jit(do_not_specialize=_DYNAMIC_GEOMETRY_PARAMETERS)
def _prefill_shared_heads_score_kernel(
    Q,
    K,
    S,
    CUQ,
    KLENS,
    PREFIX,
    CUPAGES,
    PAGES,
    OUT,
    TOTAL_Q,
    HEADS: tl.constexpr,
    MAX_PAGES,
    PHYSICAL_PAGES,
    TABLE_SIZE,
    QSTRIDE: tl.constexpr,
    HSTRIDE: tl.constexpr,
    KSTRIDE: tl.constexpr,
    SSTRIDE: tl.constexpr,
    OSTRIDE_H,
    OSTRIDE_Q,
    TILE_Q: tl.constexpr,
):
    # Put the four independent index heads in the MMA row dimension. Each
    # page still has its original 128-token reduction and Q8K8 dot semantics.
    # No scores are reduced across heads, and no extra workspace is required.
    tile = tl.program_id(0)
    segment = tl.program_id(1)
    first_block = tl.program_id(2) * 2
    start = tl.load(CUQ + segment)
    stop = tl.load(CUQ + segment + 1)
    lane = tl.arange(0, TILE_Q * HEADS)
    local_row = tile * TILE_Q + lane // HEADS
    head = lane % HEADS
    row = start + local_row
    row_ok = (row < stop) & (row >= 0) & (row < TOTAL_Q)
    if tile * TILE_Q >= stop - start:
        return
    length = tl.load(KLENS + segment)
    prefix = tl.load(PREFIX + segment)
    page_start = tl.load(CUPAGES + segment)
    page_end = tl.load(CUPAGES + segment + 1)
    d = tl.arange(0, 128)
    query = tl.load(
        Q
        + row[:, None].to(tl.int64) * QSTRIDE
        + head[:, None].to(tl.int64) * HSTRIDE
        + d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    last_query = prefix + tl.minimum((tile + 1) * TILE_Q, stop - start) - 1
    for page in tl.static_range(2):
        block = first_block + page
        page_offset = page_start + block
        logical_ok = (
            (block * 128 < length) & (page_offset < page_end) & (block < MAX_PAGES)
        )
        physical = tl.load(
            PAGES + page_offset,
            mask=logical_ok & (page_offset >= 0) & (page_offset < TABLE_SIZE),
            other=-1,
        ).to(tl.int64)
        page_ok = logical_ok & (physical >= 0) & (physical < PHYSICAL_PAGES)
        if page_ok & (block * 128 <= last_query):
            key = _load_index_page_fp8(K, S, physical, page_ok, KSTRIDE, SSTRIDE, True)
            dot = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
            position = block * 128 + tl.arange(0, 128)
            visible = (
                row_ok[:, None]
                & (position[None, :] < length)
                & (position[None, :] <= prefix + local_row[:, None])
            )
            score = tl.max(tl.where(visible, dot, float("-inf")), axis=1)
        else:
            score = tl.full((TILE_Q * HEADS,), float("-inf"), tl.float32)
        tl.store(
            OUT + head.to(tl.int64) * OSTRIDE_H + row.to(tl.int64) * OSTRIDE_Q + block,
            score,
            mask=row_ok & (block < MAX_PAGES),
        )


@torch.no_grad()
def q8kv4_prefill_index_score(
    q,
    packed,
    scales_mma,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    cu_page_offsets,
    kv_indices,
    out,
    *,
    max_seqlen_q,
    tile_q=None,
):
    """Write raw/unscaled FP32 max scores ``[H,totalQ,maxpages]``.

    Q is E4M3 [totalQ,H,128]. K is uint8 [physical_pages,1,128,64]
    and scales use MMA [physical_pages,1,2,32,4,4]. ``cu_page_offsets``
    partitions flat request-level ``kv_indices``; causal positions within
    each query segment start at ``prefix_lens``. Caller owns all buffers and
    computes page offsets outside this allocation-free launch boundary.

    ``max_seqlen_q`` must cover the longest segment (also on Graph replay).
    Invalid/unavailable pages and invisible keys yield -inf. No TopK rules,
    scaling, full-history dequantization, or BF16 compute are applied here.
    Long query segments use 128-row tiles to amortize packed-K decoding;
    These tiles reuse Q across two pages without an additional workspace;
    short segments retain 32-row tiles and the single-page kernel.
    """
    scales = scales_mma
    if q.ndim != 3 or q.dtype != torch.float8_e4m3fn or q.shape[2] != 128:
        raise ValueError("q must be E4M3 [totalQ,H,128]")
    if q.stride(2) != 1:
        raise ValueError("q head dimensions must be contiguous")
    if (
        packed.dtype != torch.uint8
        or packed.ndim != 4
        or packed.shape[1:] != (1, 128, 64)
    ):
        raise ValueError("packed must be uint8 [pages,1,128,64]")
    if packed.stride()[2:] != (64, 1):
        raise ValueError("packed page contents must be contiguous")
    if scales.ndim != 6 or scales.shape != (packed.shape[0], 1, 2, 32, 4, 4):
        raise ValueError("scales must be MMA [pages,1,2,32,4,4]")
    if scales.dtype not in (torch.uint8, torch.float8_e4m3fn) or scales.stride()[
        2:
    ] != (512, 16, 4, 1):
        raise ValueError("scales must have contiguous byte MMA page contents")
    segments = cu_seqlens.numel() - 1
    metadata = (cu_seqlens, seq_lens, prefix_lens, cu_page_offsets, kv_indices)
    if segments < 0 or any(
        x.dtype != torch.int32 or x.ndim != 1 or not x.is_contiguous() for x in metadata
    ):
        raise ValueError("metadata must be contiguous int32 vectors")
    if (
        seq_lens.numel() != segments
        or prefix_lens.numel() != segments
        or cu_page_offsets.numel() != segments + 1
    ):
        raise ValueError("inconsistent segment metadata sizes")
    if (
        out.ndim != 3
        or out.shape[:2] != (q.shape[1], q.shape[0])
        or out.dtype != torch.float32
        or out.stride(2) != 1
        or out.stride(1) < out.shape[2]
        or out.stride(0) < out.shape[1] * out.stride(1)
    ):
        raise ValueError(
            "out must be non-overlapping FP32 [H,totalQ,maxpages] with contiguous pages"
        )
    if not q.is_cuda or any(
        x.device != q.device for x in (packed, scales, out, *metadata)
    ):
        raise ValueError("all tensors must reside on the same CUDA device")
    automatic_tile = tile_q is None
    if automatic_tile:
        tile_q = 128 if max_seqlen_q >= 128 else 32
    if tile_q not in (16, 32, 64, 128) or max_seqlen_q < 0:
        raise ValueError("tile_q must be 16/32/64/128 and max_seqlen_q nonnegative")
    if q.shape[0] and max_seqlen_q == 0:
        raise ValueError("nonempty Q requires a positive max_seqlen_q")
    if not q.shape[0] or not q.shape[1] or not out.shape[2] or not segments:
        return out
    two_pages = tile_q == 128 and max_seqlen_q >= 128 and out.shape[2] >= 2
    kernel = _prefill_two_page_score_kernel if two_pages else _prefill_score_kernel
    shared_heads = automatic_tile and two_pages and q.shape[1] == 4
    if shared_heads:
        kernel = _prefill_shared_heads_score_kernel
        tile_q = 32
    page_tiles = triton.cdiv(out.shape[2], 2) if two_pages else out.shape[2]
    segment_tiles = segments if shared_heads else segments * q.shape[1]
    # Bound registers only for the validated H4/M128 shared-head variant.
    # Other head counts and short/explicit tiles retain their existing launch.
    launch_options = {"maxnreg": 96} if shared_heads else {}
    kernel[(triton.cdiv(max_seqlen_q, tile_q), segment_tiles, page_tiles)](
        q,
        packed,
        scales.view(torch.uint8),
        cu_seqlens,
        seq_lens,
        prefix_lens,
        cu_page_offsets,
        kv_indices,
        out,
        q.shape[0],
        q.shape[1],
        out.shape[2],
        packed.shape[0],
        kv_indices.numel(),
        q.stride(0),
        q.stride(1),
        packed.stride(0),
        scales.stride(0),
        out.stride(0),
        out.stride(1),
        tile_q,
        num_warps=4,
        **launch_options,
    )
    return out

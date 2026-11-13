"""Visible-length Indexer scoring for PPU Decode Graphs."""

from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.topk import (
    native_topk_op as _verify_topk_op,
    select_tokens,
)

_INDEXER_POOL = "indexer_kv"
_MAX_SPARSE_BLOCKS = 4096
_TOPK_TILE = 65536
# Enter the native cooperative radix path. Its medium-row implementation can
# truncate a coarse bin at 4,096 values, which is unsafe for tied block scores.
_CANDIDATE_RADIX_WIDTH = 32772
_SPARSE_INDEXER_ENV = "RTP_V41_PPU_GRAPH_SPARSE_INDEXER"
_VERIFY_FUSED_INDEXER_ENV = "RTP_V41_PPU_VERIFY_FUSED_INDEXER"
_VERIFY_FUSED_TOPK_ENV = "RTP_V41_PPU_VERIFY_FUSED_INDEXER_TOPK"
_VERIFY_COMPACT_TOPK_ENV = "RTP_V41_PPU_VERIFY_COMPACT_TOPK"


def indexer_scores(
    query, head_weights, *, cache, source_layer, ratio, positions, valid_mask
) -> torch.Tensor:
    """Return dense scores for Verify and callers requiring a single tensor."""
    tiles = list(
        _indexer_score_tiles(
            query, head_weights, cache=cache, source_layer=source_layer, ratio=ratio,
            visible=(positions + 1) // ratio, valid_mask=valid_mask,
        )
    )
    return _join_tiles([logits for logits, _, _ in tiles])


def _join_tiles(tiles):
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=1)


def _quantize_score_query(query, compact):
    if compact and query.dtype == torch.bfloat16 and query.is_contiguous() and _quant._ppu_fast_path(query):
        # The existing native quantizer converts each BF16 load to FP32.
        # Passing BF16 rows directly avoids materializing the same conversion.
        rows = query.view(-1, _quant.INDEXER_HEAD_DIM)
        payload, scales = _quant._quantize_mx_rows_ppu(rows)
        return payload.view(*query.shape[:-1], rows.shape[-1] // 2), scales.view(query.shape[:-1])
    return _quant.quantize_indexer_q(query)


def _candidate_flags(candidates, capacity, block_size):
    count = candidates.shape[1]
    if (
        count > 0 and count & (count - 1) == 0
        and candidates.dtype == torch.int32 and candidates.is_contiguous()
    ):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_decode_topk import build_candidate_flags

        return build_candidate_flags(candidates, capacity, block_size)
    # The CUDA helper uses a power-of-two candidate vector. Keep the original
    # bitmap construction for other candidate layouts and widths.
    nblocks = (capacity + block_size - 1) // block_size
    flags = torch.zeros(
        candidates.shape[0], nblocks + 1, dtype=torch.bool, device=candidates.device
    )
    ids = torch.where(candidates >= 0, candidates, nblocks).long()
    flags.scatter_(1, ids, True)
    return flags


def _cached_candidate_flags(candidates, capacity, block_size, shared):
    if shared is None:
        return _candidate_flags(candidates, capacity, block_size)
    key = ("indexer_candidate_flags", id(candidates), capacity, block_size)
    if key not in shared:
        shared[key] = candidates, _candidate_flags(candidates, capacity, block_size)
    return shared[key][1]


def _score_inputs(query, head_weights, pool, physical_page, compact, quantized_query=None):
    payload, scales = (
        _quantize_score_query(query, compact) if quantized_query is None else quantized_query
    )
    entry_bytes = _quant.INDEXER_K_ENTRY_BYTES
    packed_pool = pool.as_strided(
        (pool.shape[0], physical_page, 1, entry_bytes),
        (pool.stride(0), entry_bytes, entry_bytes, 1),
    )
    weights = head_weights.to(torch.bfloat16).contiguous()
    return (payload.unsqueeze(1), scales.unsqueeze(1)), packed_pool, weights


def _indexer_score_tiles(
    query, head_weights, *, cache, source_layer, ratio, visible, valid_mask,
    compact=False, fused_tiles=False, candidate_block_size=0, candidates=None, candidate_active=None,
    candidate_flags_cache=None, quantized_query=None, tile_metadata_cache=None,
):
    """Score the visible prefix in capture-sized tiles.

    The provider uses physical pages to address the planar FP4 payload and
    scales. Sparse block IDs skip the unused half of ratio-2 pages while the
    scheduler and output columns remain in compact compressed coordinates.
    """
    from deep_gemm import (
        fp8_fp4_paged_sparse_mqa_logits,
        get_paged_sparse_mqa_logits_metadata,
    )

    pool = cache.pool(source_layer, _INDEXER_POOL)
    table = cache.tables[_INDEXER_POOL][: query.shape[0]]
    physical_page = cache.entries[_INDEXER_POOL]
    raw_page = cache.spans[_INDEXER_POOL]
    if ratio not in (1, 2) or raw_page % ratio:
        raise ValueError("unsupported PPU paged FP4 Graph Indexer ratio")
    logical_page = raw_page // ratio
    tile = 16 if logical_page % 16 == 0 and physical_page % 16 == 0 else 8
    if (
        not 0 < logical_page <= physical_page
        or logical_page % tile
        or physical_page % tile
        or pool.dtype != torch.uint8
        or pool.stride(0) % 512
        or table.dtype != torch.int32
        or query.shape[1:] != (32, 128)
    ):
        raise ValueError("unsupported PPU paged FP4 Graph Indexer geometry")

    capacity = min(max(1, cache.max_seq_len // ratio), table.shape[1] * logical_page)
    kernels = _compact_kernels() if compact or fused_tiles else None
    if kernels is None:
        visible = torch.where(valid_mask, visible.clamp(0, capacity).to(torch.int32), 0)
    packed_query, packed_pool, weights = _score_inputs(
        query, head_weights, pool, physical_page, compact, quantized_query,
    )
    if kernels is None:
        rows = torch.arange(query.shape[0], dtype=torch.int32, device=query.device)
    else:
        rows = None
    logical_blocks = logical_page // tile
    physical_blocks = physical_page // tile
    chunk_pages = _MAX_SPARSE_BLOCKS // logical_blocks
    if chunk_pages == 0:
        raise ValueError("PPU Graph Indexer page exceeds the sparse scorer capacity")

    # Metadata, scores and all temporary shapes are fixed at capture. The
    # number of blocks scored per row comes from lengths on every replay.
    chunk_width = chunk_pages * logical_page
    allowed = None
    if candidates is not None:
        allowed = _cached_candidate_flags(
            candidates, capacity, candidate_block_size, candidate_flags_cache,
        )
    for start in range(0, capacity, chunk_width):
        width = min(chunk_width, capacity - start)
        first_page = start // logical_page
        last_page = first_page + (width + logical_page - 1) // logical_page
        pages = table[:, first_page:last_page]
        blocks = ((width + tile * 4 - 1) // (tile * 4)) * 4
        entry = None
        if tile_metadata_cache is not None:
            key = (
                "verify_indexer_tile_metadata", table.data_ptr(), table.shape,
                table.stride(), id(visible), id(valid_mask), pool.shape[0],
                physical_page, logical_page, tile, start, width,
            )
            entry = tile_metadata_cache.get(key)
        if entry is not None:
            safe_pages, lengths, metadata = entry
        elif kernels is None:
            valid_pages = (pages > 0) & (pages < pool.shape[0])
            safe_pages = torch.where(valid_pages, pages, 0).contiguous()
            lengths = (visible - start).clamp(0, width)
            block_ids = torch.arange(blocks, dtype=torch.int32, device=query.device)
            block_ids = (
                block_ids // logical_blocks
            ) * physical_blocks + block_ids % logical_blocks
            block_ids = block_ids.unsqueeze(0).expand(query.shape[0], -1).contiguous()
        else:
            if rows is None:
                rows = torch.empty(query.shape[0], dtype=torch.int32, device=query.device)
            safe_pages = torch.empty(pages.shape, dtype=torch.int32, device=query.device)
            lengths = torch.empty_like(rows)
            block_ids = torch.empty(
                (query.shape[0], blocks), dtype=torch.int32, device=query.device
            )
            grid = (query.shape[0], kernels.triton.cdiv(max(pages.shape[1], blocks), 256))
            kernels.prepare_tiles[grid](
                pages, visible, valid_mask, safe_pages, lengths, rows, block_ids,
                visible.stride(0), valid_mask.stride(0),
                pages.stride(0), pages.stride(1), pages.shape[1], pool.shape[0],
                logical_blocks, physical_blocks, blocks, start, width, 256,
            )
        if entry is None:
            metadata = get_paged_sparse_mqa_logits_metadata(
                lengths, safe_pages, rows, physical_page, block_ids, torch.int8, tile
            )
            if tile_metadata_cache is not None:
                tile_metadata_cache[key] = safe_pages, lengths, metadata
        scores = fp8_fp4_paged_sparse_mqa_logits(
            packed_query, packed_pool, weights, metadata, blocks, tile
        )
        if kernels is None:
            scores = scores[:, :width].float()
            columns = torch.arange(width, device=query.device)
            page_indices = (columns // logical_page).expand(query.shape[0], -1)
            valid = (columns[None, :] < lengths[:, None]) & valid_pages.gather(
                1, page_indices
            )
            yield scores.masked_fill_(~valid, float("-inf")), lengths, False
        else:
            # Match the reference tile's logical width and row stride so the
            # native persistent TopK keeps the same launch/tie geometry.
            logits = torch.empty(
                (query.shape[0], width), dtype=torch.float32, device=query.device
            )
            grid = (query.shape[0], kernels.triton.cdiv(width, 1024))
            kernels.write_logits[grid](
                scores, safe_pages, lengths, allowed, candidate_active, logits,
                scores.stride(0), safe_pages.shape[1], logical_page, start, width,
                candidate_block_size, 0 if allowed is None else allowed.shape[1], 1024,
            )
            yield logits, lengths, scores.dtype == torch.bfloat16


@lru_cache(maxsize=1)
def _compact_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def descending_score_key(score):
        # Torch ranks NaN first and treats signed zeros as equal.
        bits = score.to(tl.uint32, bitcast=True)
        magnitude = bits & 0x7FFFFFFF
        bits = tl.where(magnitude == 0, 0, bits).to(tl.uint32)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
        return tl.where(magnitude > 0x7F800000, 0xFFFFFFFF, ordered).to(tl.uint32)

    @triton.jit
    def encode_rank(score, index):
        # Exact BF16 scores and 15-bit tie order fit in finite normal floats.
        rank = (score.to(tl.uint32) << 15) | (32767 - index.to(tl.uint32))
        negative = rank < (1 << 30)
        magnitude = 0x00800000 + tl.where(negative, (1 << 30) - 1 - rank, rank - (1 << 30))
        bits = magnitude | tl.where(negative, 0x80000000, 0).to(tl.uint32)
        return bits.to(tl.uint32).to(tl.float32, bitcast=True)

    @triton.jit
    def decode_rank(encoded):
        bits = encoded.to(tl.uint32, bitcast=True)
        magnitude = (bits & 0x7FFFFFFF) - 0x00800000
        rank = tl.where(
            (bits & 0x80000000) != 0, (1 << 30) - 1 - magnitude, (1 << 30) + magnitude
        )
        return rank >> 15, (32767 - (rank & 32767)).to(tl.int32)

    @triton.jit
    def prepare_tiles(
        table, visible, valid_mask, safe_pages, lengths, rows, block_ids,
        visible_stride: tl.constexpr, valid_stride: tl.constexpr,
        row_stride: tl.constexpr, col_stride: tl.constexpr,
        page_count: tl.constexpr, pool_blocks: tl.constexpr,
        logical_blocks: tl.constexpr, physical_blocks: tl.constexpr,
        blocks: tl.constexpr, start: tl.constexpr, width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        segment = tl.program_id(1)
        column = segment * BLOCK + tl.arange(0, BLOCK)
        page = tl.load(
            table + row * row_stride + column * col_stride,
            column < page_count, other=0,
        )
        page = tl.where((page > 0) & (page < pool_blocks), page, 0)
        tl.store(safe_pages + row * page_count + column, page, column < page_count)
        block = (column // logical_blocks) * physical_blocks + column % logical_blocks
        tl.store(block_ids + row * blocks + column, block, column < blocks)
        if segment == 0:
            length = tl.load(visible + row * visible_stride)
            active = tl.load(valid_mask + row * valid_stride)
            length = tl.minimum(tl.maximum(length - start, 0), width)
            tl.store(lengths + row, tl.where(active, length, 0))
            tl.store(rows + row, row)

    @triton.jit
    def write_logits(
        source, pages, lengths, allowed, candidate_active, logits,
        score_stride: tl.constexpr, page_count: tl.constexpr,
        logical_page: tl.constexpr, start: tl.constexpr, width: tl.constexpr,
        candidate_block: tl.constexpr, candidate_stride: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        length = tl.load(lengths + row)
        live = (column < width) & (column < length)
        page = tl.load(pages + row * page_count + column // logical_page, live, other=0)
        live = live & (page != 0)
        if candidate_stride > 0:
            active = tl.load(candidate_active + row)
            permitted = tl.load(
                allowed + row * candidate_stride + (start + column) // candidate_block,
                live, other=False,
            )
            live = live & (~active | (permitted != 0))
        score = tl.load(source + row * score_stride + column, live, other=0).to(tl.float32)
        tl.store(logits + row * width + column, tl.where(live, score, -float("inf")), column < width)

    @triton.jit
    def overlay_verify_tail(
        logits, tentative, first, allowed, candidate_active,
        start: tl.constexpr, width: tl.constexpr, tail_width: tl.constexpr,
        row_stride: tl.constexpr, col_stride: tl.constexpr,
        candidate_block: tl.constexpr, candidate_stride: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        column = tl.load(first + row) + lane
        live = (lane < tail_width) & (column >= start) & (column < start + width)
        if candidate_stride > 0:
            active = tl.load(candidate_active + row)
            permitted = tl.load(
                allowed + row * candidate_stride + column // candidate_block,
                live, other=False,
            )
            live = live & (~active | (permitted != 0))
        score = tl.load(tentative + row * row_stride + lane * col_stride, live, other=0)
        tl.store(logits + row * width + column - start, score, live)

    @triton.jit
    def round_bf16(value):
        bits = value.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        return (bits << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def accumulate_verify_head(dots, weights, accum, row, head, columns, mask,
                               dot_row: tl.constexpr, dot_head: tl.constexpr,
                               dot_col: tl.constexpr, weight_row: tl.constexpr,
                               weight_head: tl.constexpr):
        dot = tl.load(dots + row * dot_row + head * dot_head + columns * dot_col, mask, other=0)
        weight = tl.load(weights + row * weight_row + head * weight_head)
        bits = weight.to(tl.uint16, bitcast=True).to(tl.uint32)
        weight = (bits << 16).to(tl.float32, bitcast=True)
        value = round_bf16(tl.where(dot <= 0, 0., dot))
        return round_bf16(value * weight + accum)

    @triton.jit
    def reduce_verify_heads(
        dots, weights, live, result,
        width: tl.constexpr, dot_row: tl.constexpr, dot_head: tl.constexpr,
        dot_col: tl.constexpr, weight_row: tl.constexpr, weight_head: tl.constexpr,
        live_row: tl.constexpr, live_col: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = columns < width
        active = tl.load(live + row * live_row + columns * live_col, mask, other=False)
        lane0 = tl.full((BLOCK,), 0., tl.float32)
        lane1 = tl.full((BLOCK,), 0., tl.float32)
        lane2 = tl.full((BLOCK,), 0., tl.float32)
        lane3 = tl.full((BLOCK,), 0., tl.float32)
        for group in range(8):
            head = group * 4
            lane0 = accumulate_verify_head(dots, weights, lane0, row, head, columns, active,
                                           dot_row, dot_head, dot_col, weight_row, weight_head)
            lane1 = accumulate_verify_head(dots, weights, lane1, row, head + 1, columns, active,
                                           dot_row, dot_head, dot_col, weight_row, weight_head)
            lane2 = accumulate_verify_head(dots, weights, lane2, row, head + 2, columns, active,
                                           dot_row, dot_head, dot_col, weight_row, weight_head)
            lane3 = accumulate_verify_head(dots, weights, lane3, row, head + 3, columns, active,
                                           dot_row, dot_head, dot_col, weight_row, weight_head)
        score = round_bf16(round_bf16(lane0 + lane2) + round_bf16(lane1 + lane3))
        tl.store(result + row * width + columns, tl.where(active, score, -float("inf")), mask)

    @triton.jit
    def prepare_sparse_candidates(
        table, candidates, candidate_active, visible, valid, logical_ids, provider_ids, lengths, rows,
        visible_stride: tl.constexpr, valid_stride: tl.constexpr, active_stride: tl.constexpr,
        page_count: tl.constexpr, pool_blocks: tl.constexpr, capacity: tl.constexpr,
        logical_per_page: tl.constexpr, physical_per_page: tl.constexpr, tile: tl.constexpr,
        count: tl.constexpr, blocks: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        length = tl.load(visible + row * visible_stride)
        live = tl.load(valid + row * valid_stride)
        active = tl.load(candidate_active + row * active_stride)
        candidate = tl.load(candidates + row * count + lane, (lane < count) & active, other=-1)
        block = tl.where(active, candidate, lane).to(tl.int32)
        page_column = block // logical_per_page
        in_range = live & (block >= 0) & (block < tl.cdiv(tl.minimum(length, capacity), tile))
        page = tl.load(
            table + row * page_count + page_column,
            in_range & (page_column < page_count), other=0,
        )
        present = (lane < count) & in_range & (page > 0) & (page < pool_blocks)
        missing = 0x7FFFFFFF
        ordered = tl.sort(tl.where(present, block, missing), descending=False)
        previous = tl.gather(ordered, tl.maximum(lane - 1, 0), axis=0)
        keep = (ordered != missing) & ((lane == 0) | (ordered != previous))
        slot = tl.cumsum(keep.to(tl.int32), axis=0) - 1
        tl.store(logical_ids + row * blocks + slot, ordered, keep)
        # Provider addresses use physical page width. Keep the original
        # compressed logical ID alongside each scheduled block for TopK.
        block = tl.where(keep, ordered, 0)
        provider = (block // logical_per_page) * physical_per_page + block % logical_per_page
        tl.store(provider_ids + row * blocks + slot, provider, keep)
        tl.store(lengths + row, tl.sum(keep.to(tl.int32), axis=0) * tile)
        tl.store(rows + row, row)

    @triton.jit
    def pack_sparse_ranks(
        scores, logical_ids, lengths, visible, valid, ranks, rank_lengths,
        score_stride: tl.constexpr, visible_stride: tl.constexpr, valid_stride: tl.constexpr,
        capacity: tl.constexpr, tile: tl.constexpr, blocks: tl.constexpr,
        padded: tl.constexpr, topk: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        segment = tl.program_id(1)
        column = segment * BLOCK + tl.arange(0, BLOCK)
        length = tl.load(visible + row * visible_stride)
        long_row = tl.load(valid + row * valid_stride) & (length > topk)
        compact_length = tl.load(lengths + row)
        present = long_row & (column < compact_length)
        logical = tl.load(logical_ids + row * blocks + column // tile, present, other=0)
        present = present & (logical * tile + column % tile < tl.minimum(length, capacity))
        score = tl.load(scores + row * score_stride + column, present, other=-float("inf"))
        encoded = encode_rank(descending_score_key(score.to(tl.float32)) >> 16, column)
        tl.store(
            ranks + row * padded + column,
            tl.where(column < blocks * tile, encoded, -float("inf")),
            (column < padded) & long_row,
        )
        if segment == 0:
            tl.store(rank_lengths + row, tl.where(long_row, padded, 0))

    @triton.jit
    def finish_sparse_selection(
        ranks, picked, logical_ids, lengths, visible, valid, output,
        visible_stride: tl.constexpr, valid_stride: tl.constexpr,
        tile: tl.constexpr, blocks: tl.constexpr, padded: tl.constexpr, topk: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, topk)
        length = tl.maximum(tl.load(visible + row * visible_stride), 0)
        length = tl.where(tl.load(valid + row * valid_stride), length, 0)
        if length <= topk:
            logical = tl.where(lane < length, lane, -1)
        else:
            index = tl.load(picked + row * topk + lane)
            encoded = tl.load(ranks + row * padded + index)
            ranked = tl.sort(encoded, descending=True)
            score, index = decode_rank(ranked)
            compact_length = tl.load(lengths + row)
            finite = (score > 0x007F) & (score < 0xFF80)
            keep = finite & (index < compact_length)
            block = tl.load(logical_ids + row * blocks + index // tile, keep, other=0)
            logical = tl.where(keep, block * tile + index % tile, -1)
        tl.store(output + row * topk + lane, logical)

    @triton.jit
    def pack_verify_compact(
        scores, logical_ids, compact_lengths, first, tentative, visible, valid,
        allowed, candidate_active, packed, native_lengths,
        score_stride: tl.constexpr, first_stride: tl.constexpr,
        tail_row: tl.constexpr, tail_col: tl.constexpr,
        visible_stride: tl.constexpr, valid_stride: tl.constexpr,
        active_stride: tl.constexpr, allowed_stride: tl.constexpr,
        blocks: tl.constexpr, tile: tl.constexpr, capacity: tl.constexpr,
        tail_width: tl.constexpr, padded: tl.constexpr, topk: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        segment = tl.program_id(1)
        column = segment * BLOCK + tl.arange(0, BLOCK)
        live_row = tl.load(valid + row * valid_stride)
        length = tl.maximum(tl.load(visible + row * visible_stride), 0)
        split = tl.load(first + row * first_stride)
        compact_length = tl.load(compact_lengths + row)
        prefix_live = live_row & (column < blocks * tile) & (column < compact_length)
        block = tl.load(logical_ids + row * blocks + column // tile, prefix_live, other=-1)
        logical = block * tile + column % tile
        prefix_live = prefix_live & (logical >= 0) & (logical < split)
        prefix_live = prefix_live & (logical < capacity) & (logical < length)
        prefix_score = tl.load(scores + row * score_stride + column, prefix_live, other=-float("inf"))
        tail = column - blocks * tile
        logical = split + tail
        tail_live = live_row & (tail >= 0) & (tail < tail_width)
        tail_live = tail_live & (logical >= 0) & (logical < capacity) & (logical < length)
        active = tl.load(candidate_active + row * active_stride)
        permitted = tl.load(allowed + row * allowed_stride + logical // tile, tail_live, other=0) != 0
        tail_live = tail_live & (~active | permitted)
        tail_score = tl.load(tentative + row * tail_row + tail * tail_col, tail_live, other=-float("inf"))
        score = tl.where(prefix_live, prefix_score.to(tl.float32), tail_score)
        tl.store(packed + row * padded + column, score, column < padded)
        if segment == 0:
            # Cooperative radix avoids the bounded coarse-bin buffers in the
            # native small/medium paths. Padding never becomes a logical ID.
            tl.store(native_lengths + row, tl.where(live_row & (length > topk), padded, 0))

    @triton.jit
    def finish_verify_compact(
        packed, picked, logical_ids, compact_lengths, first, visible, valid, output,
        first_stride: tl.constexpr, visible_stride: tl.constexpr, valid_stride: tl.constexpr,
        blocks: tl.constexpr, tile: tl.constexpr, capacity: tl.constexpr,
        tail_width: tl.constexpr, padded: tl.constexpr, topk: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        length = tl.maximum(tl.load(visible + row * visible_stride), 0)
        length = tl.where(tl.load(valid + row * valid_stride), length, 0)
        if length <= topk:
            logical = tl.where(lane < length, lane, -1).to(tl.int64)
        else:
            position = tl.load(picked + row * topk + lane, lane < topk, other=-1)
            keep = (lane < topk) & (position >= 0) & (position < padded)
            score = tl.load(packed + row * padded + position, keep, other=-float("inf"))
            keep = keep & (tl.abs(score) < float("inf"))
            split = tl.load(first + row * first_stride)
            compact_length = tl.load(compact_lengths + row)
            prefix = keep & (position < blocks * tile) & (position < compact_length)
            block = tl.load(logical_ids + row * blocks + position // tile, prefix, other=-1)
            prefix_id = block * tile + position % tile
            prefix = prefix & (prefix_id >= 0) & (prefix_id < split)
            tail = position - blocks * tile
            tail_live = keep & (tail >= 0) & (tail < tail_width)
            logical = tl.where(prefix, prefix_id, tl.where(tail_live, split + tail, -1))
            logical = tl.where((logical >= 0) & (logical < capacity) & (logical < length), logical, -1).to(tl.int64)
        tl.store(output + row * topk + lane, logical.to(tl.int32), lane < topk)

    @triton.jit
    def verify_sparse_map(
        logical_ids, lengths, lookup,
        blocks: tl.constexpr, dense_blocks: tl.constexpr, tile: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        tl.store(lookup + row * dense_blocks + lane, -1, lane < dense_blocks)
        # Each row has exactly one CTA. Complete its initialization before
        # the compact valid blocks overwrite their own entries.
        tl.debug_barrier()
        length = tl.load(lengths + row)
        logical = tl.load(logical_ids + row * blocks + lane, lane < blocks, other=-1)
        keep = (lane < blocks) & (lane * tile < length) & (logical >= 0) & (logical < dense_blocks)
        tl.store(lookup + row * dense_blocks + logical, lane, keep)

    @triton.jit
    def write_verify_sparse_logits(
        scores, lookup, visible, valid, logits,
        score_stride: tl.constexpr, dense_blocks: tl.constexpr,
        visible_stride: tl.constexpr, valid_stride: tl.constexpr,
        start: tl.constexpr, width: tl.constexpr, tile: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = start + tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = (column < start + width) & tl.load(valid + row * valid_stride)
        live = live & (column < tl.load(visible + row * visible_stride))
        compact = tl.load(lookup + row * dense_blocks + column // tile, live, other=-1)
        live = live & (compact >= 0)
        score = tl.load(scores + row * score_stride + compact * tile + column % tile, live, other=0)
        tl.store(logits + row * width + column - start,
                 tl.where(live, score.to(tl.float32), -float("inf")), column < start + width)

    @triton.jit
    def collect_tile(
        logits, local_lengths, selected, merged_logits, merged_ids,
        width: tl.constexpr, start: tl.constexpr, topk: tl.constexpr,
        merge_width: tl.constexpr, offset: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        local_id = tl.load(selected + row * topk + column, column < topk, other=-1)
        length = tl.load(local_lengths + row)
        keep = (local_id >= 0) & (local_id < length) & (column < topk)
        score = tl.load(logits + row * width + local_id, keep, other=0)
        destination = row * merge_width + offset + column
        tl.store(merged_logits + destination, tl.where(keep, score, -float("inf")), column < topk)
        tl.store(merged_ids + destination, local_id + start, column < topk)

    @triton.jit
    def finish_topk(
        values, ranks, merged_ids, visible, valid, selected,
        visible_stride: tl.constexpr, valid_stride: tl.constexpr,
        topk: tl.constexpr, merge_width: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        length = tl.maximum(tl.load(visible + row * visible_stride), 0)
        length = tl.where(tl.load(valid + row * valid_stride), length, 0)
        if length <= topk:
            ids = tl.where(column < length, column, -1)
        else:
            position = row * topk + column
            score = tl.load(values + position, column < topk, other=-float("inf"))
            if merge_width == topk:
                index = column
            else:
                index = tl.load(ranks + position, column < topk, other=0)
            ids = tl.load(merged_ids + row * merge_width + index, column < topk, other=-1)
            ids = tl.where(tl.abs(score) < float("inf"), ids, -1)
        tl.store(selected + row * topk + column, ids, column < topk)

    @triton.jit
    def verify_topk_lengths(
        visible, valid, lengths,
        rows: tl.constexpr, tiles: tl.constexpr, last_width: tl.constexpr,
        visible_stride: tl.constexpr, valid_stride: tl.constexpr,
        TILE: tl.constexpr, BLOCK: tl.constexpr,
    ):
        tile = tl.program_id(0)
        row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = tl.load(valid + row * valid_stride, row < rows, other=False)
        length = tl.load(visible + row * visible_stride, row < rows, other=0)
        length = tl.where(live, tl.maximum(length, 0), 0).to(tl.int32)
        width = tl.where(tile + 1 == tiles, last_width, TILE)
        length = tl.minimum(tl.maximum(length - tile * TILE, 0), width)
        tl.store(lengths + tile * rows + row, length, row < rows)

    @triton.jit
    def merge_topk(
        values, merged_ids, visible, valid, selected,
        visible_stride: tl.constexpr, valid_stride: tl.constexpr,
        topk: tl.constexpr, merge_width: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = tl.arange(0, BLOCK)
        length = tl.maximum(tl.load(visible + row * visible_stride), 0)
        length = tl.where(tl.load(valid + row * valid_stride), length, 0)
        if length <= topk:
            ids = tl.where(column < length, column, -1)
        else:
            score = tl.load(
                values + row * merge_width + column, column < merge_width,
                other=-float("inf"),
            )
            # Resolve equal scores in input order without changing ranking.
            order = descending_score_key(score) >> 16
            key = (order << 12) | (4095 - column.to(tl.uint32))
            ranked = tl.sort(key, descending=True)
            index = 4095 - (ranked & 4095)
            rank_score = ranked >> 12
            finite = (rank_score > 0x007F) & (rank_score < 0xFF80)
            ids = tl.load(
                merged_ids + row * merge_width + index,
                (column < topk) & (index < merge_width) & finite, other=-1,
            )
        tl.store(selected + row * topk + column, ids, column < topk)

    @triton.jit
    def pool_candidate_tile(
        logits, visible, scores, lengths,
        visible_stride: tl.constexpr, width: tl.constexpr, block_size: tl.constexpr,
        block_start: tl.constexpr, tile_blocks: tl.constexpr, write_blocks: tl.constexpr,
        padded_width: tl.constexpr, TILE: tl.constexpr,
    ):
        row = tl.program_id(0)
        segment = tl.program_id(1)
        block = segment * TILE + tl.arange(0, TILE)
        offset = tl.arange(0, triton.next_power_of_2(block_size))
        column = block[:, None] * block_size + offset[None, :]
        values = tl.load(
            logits + row * width + column,
            (block[:, None] < tile_blocks) & (offset[None, :] < block_size) & (column < width),
            other=-float("inf"),
        )
        score = tl.max(descending_score_key(values) >> 16, axis=1)
        length = tl.load(visible + row * visible_stride)
        index = block_start + block
        newest = tl.maximum(length - 1, 0) // block_size
        score = tl.where(
            index == newest, tl.where(length > 0, 0xFF80, 0x007F), score
        )

        # Provider scores widened from BF16 occupy only the upper 16 bits.
        # Rank by those exact bits and then by ascending logical block ID.
        encoded = encode_rank(score, index)
        tl.store(
            scores + row * padded_width + index,
            tl.where(block < tile_blocks, encoded, -float("inf")),
            block < write_blocks,
        )
        if block_start == 0:
            if segment == 0:
                tl.store(lengths + row, padded_width)

    @triton.jit
    def finish_candidates(
        scores, picked, output, count: tl.constexpr, padded_width: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, count)
        index = tl.load(picked + row * count + lane)
        encoded = tl.load(scores + row * padded_width + index)
        ranked = tl.sort(encoded, descending=True)
        score, index = decode_rank(ranked)
        # Legacy candidate publication accepts +inf (the pinned newest block)
        # but drops -inf and NaN after selection without refilling.
        keep = (score > 0x007F) & (score < 0xFFFF)
        tl.store(output + row * count + lane, tl.where(keep, index, -1))

    return SimpleNamespace(
        triton=triton, prepare_tiles=prepare_tiles, write_logits=write_logits,
        overlay_verify_tail=overlay_verify_tail, reduce_verify_heads=reduce_verify_heads,
        prepare_sparse_candidates=prepare_sparse_candidates, pack_sparse_ranks=pack_sparse_ranks,
        finish_sparse_selection=finish_sparse_selection,
        pack_verify_compact=pack_verify_compact, finish_verify_compact=finish_verify_compact,
        verify_sparse_map=verify_sparse_map, write_verify_sparse_logits=write_verify_sparse_logits,
        collect_tile=collect_tile, finish_topk=finish_topk, merge_topk=merge_topk,
        verify_topk_lengths=verify_topk_lengths,
        pool_candidate_tile=pool_candidate_tile, finish_candidates=finish_candidates,
    )


def _candidate_tiles(score_tiles, visible, block_size, topk_blocks):
    from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import select_candidate_blocks

    tiles = [logits for logits, _, _ in score_tiles]
    if any(tile.shape[1] % block_size for tile in tiles[:-1]):
        return select_candidate_blocks(_join_tiles(tiles), visible, block_size, topk_blocks)
    width = sum(tile.shape[1] for tile in tiles)
    nblocks = (width + block_size - 1) // block_size
    if (
        8192 < nblocks <= 32768
        and block_size <= 16
        and topk_blocks in (512, 1024, 2048)
        and all(bf16 for _, _, bf16 in score_tiles)
    ):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        kernels = _compact_kernels()
        rows = visible.numel()
        device = visible.device
        padded = _CANDIDATE_RADIX_WIDTH
        scores = torch.empty((rows, padded), dtype=torch.float32, device=device)
        lengths = torch.empty(rows, dtype=torch.int32, device=device)
        picked = torch.empty((rows, topk_blocks), dtype=torch.int32, device=device)
        result = torch.empty_like(picked)
        workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
        block_start = 0
        for index, logits in enumerate(tiles):
            blocks = kernels.triton.cdiv(logits.shape[1], block_size)
            write_blocks = padded - block_start if index == len(tiles) - 1 else blocks
            kernels.pool_candidate_tile[(rows, kernels.triton.cdiv(write_blocks, 128))](
                logits, visible, scores, lengths, visible.stride(0), logits.shape[1],
                block_size, block_start, blocks, write_blocks, padded, 128, num_warps=4,
            )
            block_start += blocks
        rtp_llm_ops.dsv4_persistent_topk(
            scores, lengths, picked, workspace, topk_blocks, padded,
        )
        kernels.finish_candidates[(rows,)](
            scores, picked, result, topk_blocks, padded, num_warps=8,
        )
        return result

    if len(tiles) == 1:
        return select_candidate_blocks(tiles[0], visible, block_size, topk_blocks)

    maxima = []
    for logits in tiles:
        padding = (-logits.shape[1]) % block_size
        if padding:
            logits = torch.nn.functional.pad(logits, (0, padding), value=-torch.inf)
        maxima.append(logits.view(logits.shape[0], -1, block_size).amax(-1))
    # The shared candidate rule runs on one score per block. Tile boundaries
    # match candidate block boundaries, so the global reduction order is intact.
    visible_blocks = (visible + block_size - 1) // block_size
    return select_candidate_blocks(_join_tiles(maxima), visible_blocks, 1, topk_blocks)


def _select_compact_topk(score_tiles, visible, valid_mask, topk):
    tiles = [logits for logits, _, _ in score_tiles]
    if tiles[-1].shape[1] > _TOPK_TILE or any(
        tile.shape[1] != _TOPK_TILE for tile in tiles[:-1]
    ):
        # Keep the established tile boundaries for nonstandard page geometry.
        return select_topk(_join_tiles(tiles), visible, valid_mask, topk)

    from rtp_llm.ops.compute_ops import rtp_llm_ops

    kernels = _compact_kernels()
    rows = visible.numel()
    device = visible.device
    merged_width = len(tiles) * topk
    tile_scores = torch.empty((rows, merged_width), dtype=torch.float32, device=device)
    tile_indices = torch.empty((rows, merged_width), dtype=torch.int32, device=device)
    selected = torch.empty((rows, topk), dtype=torch.int32, device=device)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    grid = (rows, kernels.triton.cdiv(topk, 256))
    for index, (logits, local_lengths, _) in enumerate(score_tiles):
        start = index * _TOPK_TILE
        width = logits.shape[1]
        rtp_llm_ops.dsv4_persistent_topk(
            logits, local_lengths, selected, workspace, topk, width,
        )
        kernels.collect_tile[grid](
            logits, local_lengths, selected, tile_scores, tile_indices,
            width, start, topk, merged_width, index * topk, 256,
        )
    if topk < merged_width <= 4096 and all(bf16 for _, _, bf16 in score_tiles):
        kernels.merge_topk[(rows,)](
            tile_scores, tile_indices, visible, valid_mask, selected,
            visible.stride(0), valid_mask.stride(0), topk, merged_width,
            kernels.triton.next_power_of_2(merged_width), num_warps=8,
        )
    else:
        values, indices = (
            (tile_scores, None) if len(tiles) == 1 else tile_scores.topk(topk, dim=1)
        )
        kernels.finish_topk[grid](
            values, indices, tile_indices, visible, valid_mask, selected,
            visible.stride(0), valid_mask.stride(0), topk, merged_width, 256,
        )
    return selected


def _select_verify_topk_tiled(score_tiles, visible, valid_mask, topk):
    """Keep Verify's native tile and Torch merge order without dense copies.

    The committed scorer's lengths exclude the tentative journal. TopK must
    instead use each row's current visible length, just as select_topk does
    after concatenating and re-slicing the dense logits. Marking these tiles
    non-BF16 deliberately retains the original Torch topk merge on ties.
    """
    if (
        not visible.is_cuda or topk not in (512, 1024, 2048)
        or score_tiles[-1][0].shape[1] > _TOPK_TILE
        or any(tile.shape[1] != _TOPK_TILE for tile, _, _ in score_tiles[:-1])
    ):
        return select_topk(_join_tiles([tile for tile, _, _ in score_tiles]), visible, valid_mask, topk)
    kernels = _compact_kernels()
    rows, count = visible.numel(), len(score_tiles)
    lengths = torch.empty((count, rows), dtype=torch.int32, device=visible.device)
    kernels.verify_topk_lengths[(count, kernels.triton.cdiv(rows, 128))](
        visible, valid_mask, lengths, rows, count, score_tiles[-1][0].shape[1],
        visible.stride(0), valid_mask.stride(0), _TOPK_TILE, 128,
    )
    tiles = [(tile, lengths[index], False) for index, (tile, _, _) in enumerate(score_tiles)]
    return _select_compact_topk(tiles, visible, valid_mask, topk)


def _select_verify_topk(score_tiles, visible, valid_mask, topk, *, bf16_tail_provenance=False):
    """Use native cooperative TopK once over the full Verify score domain.

    The provider's sparse scoring limit is not a native TopK width limit.
    Native TopK computes its cooperative CTA geometry from the complete row
    stride and bounds resident CTAs. Equal-score indices may differ from the
    tiled path: native collection order is unspecified. Consumers map each ID
    independently before attention, so no selected rank is required. Keep the
    native order and replace nonfinite selections by -1 without refill.
    Short rows retain their ordinal contract.
    """
    if (
        len(score_tiles) == 1 or not visible.is_cuda or not _quant.is_ppu()
        or visible.numel() != 32 or topk != 512
        or any(tile.dtype != torch.float32 for tile, _, _ in score_tiles)
    ):
        return _select_verify_topk_tiled(score_tiles, visible, valid_mask, topk)
    scores = _join_tiles([tile for tile, _, _ in score_tiles])
    rows, width = scores.shape
    kernels = _compact_kernels()
    lengths = torch.empty(rows, dtype=torch.int32, device=scores.device)
    kernels.verify_topk_lengths[(1, kernels.triton.cdiv(rows, 128))](
        visible, valid_mask, lengths, rows, 1, width,
        visible.stride(0), valid_mask.stride(0), width, 128,
    )
    selected = torch.empty((rows, topk), dtype=torch.int32, device=scores.device)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=scores.device)
    native_topk, _ = _verify_topk_op(
        rows, scores.device, topk,
        bf16_provenance=bf16_tail_provenance and all(bf16 for _, _, bf16 in score_tiles),
    )
    native_topk(scores, lengths, selected, workspace, topk, width)
    values = torch.empty((rows, topk), dtype=torch.float32, device=scores.device)
    logical_ids = torch.empty_like(selected)
    grid = (rows, kernels.triton.cdiv(topk, 256))
    kernels.collect_tile[grid](
        scores, lengths, selected, values, logical_ids, width, 0, topk, topk, 0, 256,
    )
    kernels.finish_topk[grid](
        values, None, logical_ids, visible, valid_mask, selected,
        visible.stride(0), valid_mask.stride(0), topk, topk, 256,
    )
    return selected


def _sparse_candidate_scores(
    query, head_weights, cache, source_layer, ratio, visible, valid_mask,
    topk, tile, candidates, candidate_active, inactive_limit, shared, quantized_query=None,
    max_compact_width=32768,
):
    batch = query.shape[0]
    if (
        ratio not in (1, 2)
        or tile not in (8, 16)
        or candidates.ndim != 2
        or candidates.shape[0] != batch
        or candidates.dtype != torch.int32
        or candidates.device != query.device
        or not candidates.is_contiguous()
        or candidate_active is None
        or candidate_active.shape != (batch,)
        or candidate_active.dtype != torch.bool
        or candidate_active.device != query.device
    ):
        return None
    count = candidates.shape[1]
    blocks = ((count + 3) // 4) * 4
    width = blocks * tile
    if (
        not 0 < blocks <= _MAX_SPARSE_BLOCKS
        or not topk <= width <= max_compact_width
        or not 0 <= inactive_limit <= count * tile
    ):
        return None
    table = cache.tables[_INDEXER_POOL][:batch]
    physical_page = cache.entries[_INDEXER_POOL]
    raw_page = cache.spans[_INDEXER_POOL]
    logical_page = raw_page // ratio
    pool = cache.pool(source_layer, _INDEXER_POOL)
    if (
        raw_page % ratio
        or not 0 < logical_page <= physical_page
        or logical_page % tile
        or physical_page % tile
        or pool.dtype != torch.uint8
        or pool.stride(0) % 512
        or table.dtype != torch.int32
        or not table.is_contiguous()
        or query.shape[1:] != (32, 128)
    ):
        return None
    capacity = min(max(1, cache.max_seq_len // ratio), table.shape[1] * logical_page)
    if capacity <= width:
        return None

    from deep_gemm import (
        fp8_fp4_paged_sparse_mqa_logits,
        get_paged_sparse_mqa_logits_metadata,
    )
    kernels = _compact_kernels()
    device = query.device
    key = (
        "indexer_sparse_schedule", id(candidates), id(candidate_active), id(visible),
        id(valid_mask), table.data_ptr(), table.shape, physical_page, logical_page,
        pool.shape[0], capacity, tile,
    )
    schedule = None if shared is None else shared.get(key)
    if schedule is None:
        logical_ids = torch.empty((batch, blocks), dtype=torch.int32, device=device)
        provider_ids = torch.empty_like(logical_ids)
        lengths = torch.empty(batch, dtype=torch.int32, device=device)
        rows = torch.empty_like(lengths)
        kernels.prepare_sparse_candidates[(batch,)](
            table, candidates, candidate_active, visible, valid_mask,
            logical_ids, provider_ids, lengths, rows,
            visible.stride(0), valid_mask.stride(0), candidate_active.stride(0),
            table.shape[1], pool.shape[0], capacity, logical_page // tile,
            physical_page // tile, tile, count, blocks, kernels.triton.next_power_of_2(count),
            num_warps=8,
        )
        metadata = get_paged_sparse_mqa_logits_metadata(
            lengths, table, rows, physical_page, provider_ids, torch.int8, tile,
        )
        schedule = metadata, logical_ids, lengths, candidates, candidate_active, visible, valid_mask
        if shared is not None:
            shared[key] = schedule
    metadata, logical_ids, lengths, *_ = schedule
    packed_query, packed_pool, weights = _score_inputs(
        query, head_weights, pool, physical_page, True, quantized_query,
    )
    scores = fp8_fp4_paged_sparse_mqa_logits(
        packed_query, packed_pool, weights, metadata, blocks, tile,
    )
    return SimpleNamespace(
        scores=scores, logical_ids=logical_ids, lengths=lengths,
        blocks=blocks, tile=tile, capacity=capacity, kernels=kernels,
    )


def _pack_verify_compact_logits(
    scored, first, tentative, visible, valid_mask, allowed, candidate_active, topk,
    *, cooperative_bf16=False,
):
    kernels = scored.kernels
    width = scored.blocks * scored.tile + tentative.shape[1]
    padded = ((max(width, 0 if cooperative_bf16 else _CANDIDATE_RADIX_WIDTH) + 3) // 4) * 4
    packed = torch.empty((visible.numel(), padded), dtype=torch.float32, device=visible.device)
    lengths = torch.empty(visible.numel(), dtype=torch.int32, device=visible.device)
    kernels.pack_verify_compact[(visible.numel(), kernels.triton.cdiv(padded, 1024))](
        scored.scores, scored.logical_ids, scored.lengths, first, tentative,
        visible, valid_mask, allowed, candidate_active, packed, lengths,
        scored.scores.stride(0), first.stride(0), *tentative.stride(),
        visible.stride(0), valid_mask.stride(0), candidate_active.stride(0), allowed.shape[1],
        scored.blocks, scored.tile, scored.capacity, tentative.shape[1], padded, topk, 1024,
    )
    return packed, lengths


def _select_verify_compact_topk(
    scored, first, tentative, visible, valid_mask, allowed, candidate_active, topk,
    *, bf16_tail_provenance=False,
):
    """Select plain scores, then map compact positions to logical KV IDs."""
    native_topk, cooperative_bf16 = _verify_topk_op(
        visible.numel(), visible.device, topk,
        bf16_provenance=bf16_tail_provenance and scored.scores.dtype == torch.bfloat16,
    )
    packed, lengths = _pack_verify_compact_logits(
        scored, first, tentative, visible, valid_mask, allowed, candidate_active, topk,
        cooperative_bf16=cooperative_bf16,
    )
    rows, padded = packed.shape
    picked = torch.empty((rows, topk), dtype=torch.int32, device=packed.device)
    output = torch.empty_like(picked)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=packed.device)
    native_topk(packed, lengths, picked, workspace, topk, padded)
    scored.kernels.finish_verify_compact[(rows, scored.kernels.triton.cdiv(topk, 256))](
        packed, picked, scored.logical_ids, scored.lengths, first, visible, valid_mask, output,
        first.stride(0), visible.stride(0), valid_mask.stride(0), scored.blocks,
        scored.tile, scored.capacity, tentative.shape[1], padded, topk, 256,
    )
    return output


def _select_verify_sparse_indices(
    query, head_weights, *, cache, source_layer, ratio, positions, valid_mask,
    topk, candidate_block_size, candidates, candidate_active, candidate_inactive_limit, shared,
):
    """Consume existing candidates without materializing dense Verify tiles."""
    if (
        not _quant.is_ppu() or query.shape[0] != 32 or topk != 512
        or candidates is None or candidate_inactive_limit is None
        or os.environ.get(_SPARSE_INDEXER_ENV, "1") == "0"
    ):
        return None
    from rtp_llm.models_py.modules.dsv41.ppu.graph_attention import score_indexer_entries

    requests = cache.prefixes.numel()
    if requests == 0 or query.shape[0] % requests:
        return None
    tail_width = (query.shape[0] // requests + ratio - 1) // ratio
    prefix_key = ("verify_indexer_prefix", id(cache), ratio)
    if prefix_key not in shared:
        shared[prefix_key] = cache.prefixes[cache.requests] // ratio
    first = shared[prefix_key]
    visible_key = ("verify_indexer_visible", id(positions), ratio)
    if visible_key not in shared:
        shared[visible_key] = (positions + 1) // ratio
    visible = shared[visible_key]
    native = cache.native_cache
    table = getattr(cache, "row_tables", {}).get(_INDEXER_POOL)
    if table is None:
        table = native.tables[_INDEXER_POOL][cache.requests]
    paged_cache = SimpleNamespace(
        pool=native.pool, tables={_INDEXER_POOL: table}, entries=native.entries,
        spans=native.spans, max_seq_len=native.max_seq_len,
    )
    quantized_query = _quant.quantize_indexer_q(query)
    weights = head_weights.to(torch.bfloat16).contiguous()
    scored = _sparse_candidate_scores(
        query, weights, paged_cache, source_layer, ratio, first, valid_mask,
        topk, candidate_block_size, candidates, candidate_active, candidate_inactive_limit,
        shared, quantized_query, max_compact_width=_MAX_SPARSE_BLOCKS * candidate_block_size,
    )
    if scored is None:
        return None
    offsets = torch.arange(tail_width, dtype=torch.int64, device=query.device)
    tentative = score_indexer_entries(
        query, weights, cache=cache, source_layer=source_layer, ratio=ratio,
        positions=positions, valid_mask=valid_mask, entries=first[:, None] + offsets,
        quantized_query=quantized_query, score_reducer=verify_indexer_head_reduce,
    )
    allowed = _cached_candidate_flags(candidates, scored.capacity, candidate_block_size, shared)
    return _select_verify_compact_topk(
        scored, first, tentative, visible, valid_mask, allowed, candidate_active, topk,
        bf16_tail_provenance=True,
    )


def _select_sparse_candidates(
    query, head_weights, cache, source_layer, ratio, visible, valid_mask,
    topk, tile, candidates, candidate_active, inactive_limit, shared,
):
    scored = _sparse_candidate_scores(
        query, head_weights, cache, source_layer, ratio, visible, valid_mask,
        topk, tile, candidates, candidate_active, inactive_limit, shared,
    )
    if scored is None:
        return None
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    scores, logical_ids, lengths = scored.scores, scored.logical_ids, scored.lengths
    blocks, capacity, kernels = scored.blocks, scored.capacity, scored.kernels
    batch, device = query.shape[0], query.device
    padded = _CANDIDATE_RADIX_WIDTH
    ranks = torch.empty((batch, padded), dtype=torch.float32, device=device)
    rank_lengths = torch.empty_like(lengths)
    picked = torch.empty((batch, topk), dtype=torch.int32, device=device)
    result = torch.empty_like(picked)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    grid = (batch, kernels.triton.cdiv(padded, 1024))
    kernels.pack_sparse_ranks[grid](
        scores, logical_ids, lengths, visible, valid_mask, ranks, rank_lengths,
        scores.stride(0), visible.stride(0), valid_mask.stride(0),
        capacity, tile, blocks, padded, topk, 1024,
    )
    rtp_llm_ops.dsv4_persistent_topk(ranks, rank_lengths, picked, workspace, topk, padded)
    kernels.finish_sparse_selection[(batch,)](
        ranks, picked, logical_ids, lengths, visible, valid_mask, result,
        visible.stride(0), valid_mask.stride(0), tile, blocks, padded, topk, num_warps=8,
    )
    return result


def _verify_sparse_score_tiles(
    query, head_weights, *, cache, source_layer, ratio, visible, valid_mask,
    topk, candidate_block_size, candidates, candidate_active,
    inactive_limit, shared, quantized_query,
):
    """Score candidate blocks, then restore Verify's exact dense tile layout.

    Restoring the original logical columns preserves every per-token score
    and permits the independent tiled TopK reference. Tentative journal
    scores are overlaid by _verify_score_tiles after this committed producer.
    """
    scored = _sparse_candidate_scores(
        query, head_weights, cache, source_layer, ratio, visible, valid_mask,
        topk, candidate_block_size, candidates, candidate_active, inactive_limit,
        shared, quantized_query,
    )
    if scored is None:
        return None
    kernels = scored.kernels
    rows, capacity, tile = query.shape[0], scored.capacity, scored.tile
    dense_blocks = (capacity + tile - 1) // tile
    key = ("verify_sparse_lookup", id(scored.logical_ids), capacity, tile)
    lookup = None if shared is None else shared.get(key)
    if lookup is None:
        lookup = torch.empty((rows, dense_blocks), dtype=torch.int32, device=query.device)
        kernels.verify_sparse_map[(rows,)](
            scored.logical_ids, scored.lengths, lookup, scored.blocks, dense_blocks,
            tile, kernels.triton.next_power_of_2(max(scored.blocks, dense_blocks)),
            num_warps=8,
        )
        if shared is not None:
            shared[key] = lookup
    widths = [min(_TOPK_TILE, capacity - start) for start in range(0, capacity, _TOPK_TILE)]
    lengths = torch.empty((len(widths), rows), dtype=torch.int32, device=query.device)
    kernels.verify_topk_lengths[(len(widths), kernels.triton.cdiv(rows, 128))](
        visible, valid_mask, lengths, rows, len(widths), widths[-1],
        visible.stride(0), valid_mask.stride(0), _TOPK_TILE, 128,
    )
    tiles = []
    for index, width in enumerate(widths):
        logits = torch.empty((rows, width), dtype=torch.float32, device=query.device)
        kernels.write_verify_sparse_logits[(rows, kernels.triton.cdiv(width, 1024))](
            scored.scores, lookup, visible, valid_mask, logits, scored.scores.stride(0),
            dense_blocks, visible.stride(0), valid_mask.stride(0), index * _TOPK_TILE,
            width, tile, 1024,
        )
        tiles.append((logits, lengths[index], scored.scores.dtype == torch.bfloat16))
    return tiles


def select_visible_indices(
    query, head_weights, *, cache, source_layer, ratio, positions, valid_mask,
    topk, candidate_block_size=0, candidate_topk_blocks=0, candidates=None,
    candidate_active=None, visible=None, candidate_flags_cache=None,
    candidate_inactive_limit=None,
):
    """Ordinary Decode selects from native tiles and publishes logical IDs.

    The caller may share the unmasked visible length for the same positions
    and ratio, and candidate metadata within one forward over immutable IDs.
    A proven upper bound on unfiltered rows enables sparse candidate scoring.
    """
    from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import (
        mask_candidate_logits,
        select_candidate_blocks,
    )

    if visible is None:
        visible = (positions + 1) // ratio
    # A one-row slice can already be contiguous with the full score stride.
    # Keep its original native TopK launch geometry.
    compact = query.is_cuda and query.shape[0] > 1 and topk in (512, 1024, 2048)
    if (
        compact and candidates is not None and candidate_topk_blocks == 0
        and candidate_inactive_limit is not None
        and os.environ.get(_SPARSE_INDEXER_ENV, "1") != "0"
    ):
        selected = _select_sparse_candidates(
            query, head_weights, cache, source_layer, ratio, visible, valid_mask,
            topk, candidate_block_size, candidates, candidate_active,
            candidate_inactive_limit, candidate_flags_cache,
        )
        if selected is not None:
            return selected, None
    score_tiles = list(
        _indexer_score_tiles(
            query, head_weights, cache=cache, source_layer=source_layer, ratio=ratio,
            visible=visible, valid_mask=valid_mask, compact=compact,
            candidate_block_size=candidate_block_size, candidates=candidates if compact else None,
            candidate_active=candidate_active, candidate_flags_cache=candidate_flags_cache,
        )
    )
    tiles = [logits for logits, _, _ in score_tiles]
    if compact:
        published = (
            _candidate_tiles(score_tiles, visible, candidate_block_size, candidate_topk_blocks)
            if candidate_topk_blocks > 0 else None
        )
        return _select_compact_topk(score_tiles, visible, valid_mask, topk), published
    scores = _join_tiles(tiles)
    published = (
        select_candidate_blocks(scores, visible, candidate_block_size, candidate_topk_blocks)
        if candidate_topk_blocks > 0 else None
    )
    if candidates is not None:
        masked = mask_candidate_logits(scores.clone(), candidates, candidate_block_size)
        scores = torch.where(candidate_active[:, None], masked, scores)
    return select_topk(scores, visible, valid_mask, topk), published


def verify_indexer_head_reduce(dots, weights_bf16, live):
    """Run the original four-lane BF16 chain without contracting mul/add."""
    kernels = _compact_kernels()
    rows, _, width = dots.shape
    scores = torch.empty((rows, width), dtype=torch.float32, device=dots.device)
    block = min(32, kernels.triton.next_power_of_2(width))
    kernels.reduce_verify_heads[(rows, kernels.triton.cdiv(width, block))](
        dots, weights_bf16, live, scores,
        width, *dots.stride(), *weights_bf16.stride(), *live.stride(),
        block, num_warps=1, enable_fp_fusion=False,
    )
    return scores


def _verify_score_tiles(
    query, head_weights, *, cache, source_layer, ratio, positions, valid_mask,
    fused, candidates=None, candidate_active=None, candidate_block_size=0, shared=None,
    candidate_inactive_limit=None, topk=None,
):
    """Score the paged prefix and write the journal scores into their native tiles."""
    from rtp_llm.models_py.modules.dsv41.ppu.graph_attention import score_indexer_entries

    native = cache.native_cache
    prefix_key = ("verify_indexer_prefix", id(cache), ratio)
    if shared is not None and prefix_key in shared:
        first_tentative = shared[prefix_key]
    else:
        first_tentative = cache.prefixes[cache.requests] // ratio
        if shared is not None:
            shared[prefix_key] = first_tentative
    verify_width = query.shape[0] // cache.prefixes.numel()
    tail_width = (verify_width + ratio - 1) // ratio
    table = getattr(cache, "row_tables", {}).get(_INDEXER_POOL) if fused else None
    if table is None:
        table = native.tables[_INDEXER_POOL][cache.requests]
    paged_cache = SimpleNamespace(
        pool=native.pool,
        tables={_INDEXER_POOL: table},
        entries=native.entries,
        spans=native.spans,
        max_seq_len=native.max_seq_len,
    )
    quantized_query = None
    reducer = None
    if shared is not None:
        quantized_query = _quant.quantize_indexer_q(query)
        head_weights = head_weights.to(torch.bfloat16).contiguous()
        reducer = verify_indexer_head_reduce
    tiles = None
    if (fused and candidates is not None and candidate_inactive_limit is not None
            and topk in (512, 1024, 2048)
            and os.environ.get(_SPARSE_INDEXER_ENV, "1") != "0"):
        tiles = _verify_sparse_score_tiles(
            query, head_weights, cache=paged_cache, source_layer=source_layer, ratio=ratio,
            visible=first_tentative, valid_mask=valid_mask, topk=topk,
            candidate_block_size=candidate_block_size, candidates=candidates,
            candidate_active=candidate_active, inactive_limit=candidate_inactive_limit,
            shared=shared, quantized_query=quantized_query,
        )
    if tiles is None:
        tiles = list(_indexer_score_tiles(
            query, head_weights, cache=paged_cache, source_layer=source_layer, ratio=ratio,
            visible=first_tentative, valid_mask=valid_mask, fused_tiles=fused,
            candidates=candidates, candidate_active=candidate_active,
            candidate_block_size=candidate_block_size, candidate_flags_cache=shared,
            quantized_query=quantized_query, tile_metadata_cache=shared,
        ))
    offsets = torch.arange(tail_width, dtype=torch.int64, device=query.device)
    tentative = score_indexer_entries(
        query, head_weights, cache=cache, source_layer=source_layer, ratio=ratio,
        positions=positions, valid_mask=valid_mask, entries=first_tentative[:, None] + offsets,
        quantized_query=quantized_query, score_reducer=reducer,
    )
    if fused:
        # Committed tile writes already set every column at and beyond the
        # first tentative key to -inf. Only the journal's short tail can
        # change these scores. Preserve dense columns and TopK tie order.
        kernels = _compact_kernels()
        allowed = None
        if candidates is not None:
            capacity = sum(tile.shape[1] for tile, _, _ in tiles)
            allowed = _cached_candidate_flags(candidates, capacity, candidate_block_size, shared)
        start = 0
        for logits, _, _ in tiles:
            width = logits.shape[1]
            kernels.overlay_verify_tail[(query.shape[0],)](
                logits, tentative, first_tentative, allowed, candidate_active,
                start, width, tail_width, *tentative.stride(), candidate_block_size,
                0 if allowed is None else allowed.shape[1],
                kernels.triton.next_power_of_2(tail_width), num_warps=1,
            )
            start += width
    return tiles, first_tentative, tentative


def verify_indexer_scores(query, head_weights, *, cache, source_layer, ratio, positions, valid_mask):
    """Use paged scoring for committed keys and the journal for tentative keys."""
    fused = query.is_cuda and os.environ.get(_VERIFY_FUSED_INDEXER_ENV, "1") == "1"
    tiles, first_tentative, tentative = _verify_score_tiles(
        query, head_weights, cache=cache, source_layer=source_layer, ratio=ratio,
        positions=positions, valid_mask=valid_mask, fused=fused,
    )
    committed = _join_tiles([logits for logits, _, _ in tiles])
    if fused:
        return committed
    tail_width = tentative.shape[1]
    columns = torch.arange(committed.shape[1], dtype=torch.int64, device=query.device)
    relative = columns - first_tentative[:, None]
    tail = tentative.gather(1, relative.clamp(0, tail_width - 1))
    tail = tail.masked_fill(relative >= tail_width, float("-inf"))
    return torch.where(relative >= 0, tail, committed)


def select_verify_indices(
    query, head_weights, *, cache, source_layer, ratio, positions, valid_mask, topk,
    candidate_block_size=0, candidate_topk_blocks=0, candidates=None,
    candidate_active=None, shared, candidate_inactive_limit=None,
):
    """Select native TopK from the journal-aware Verify score domain."""
    if (
        not query.is_cuda or topk not in (512, 1024, 2048)
        or ratio not in (1, 2)
        or os.environ.get(_VERIFY_FUSED_INDEXER_ENV, "1") != "1"
        or os.environ.get(_VERIFY_FUSED_TOPK_ENV, "1") != "1"
    ):
        return None
    if candidate_topk_blocks == 0 and os.environ.get(_VERIFY_COMPACT_TOPK_ENV, "1") != "0":
        selected = _select_verify_sparse_indices(
            query, head_weights, cache=cache, source_layer=source_layer, ratio=ratio,
            positions=positions, valid_mask=valid_mask, topk=topk,
            candidate_block_size=candidate_block_size, candidates=candidates,
            candidate_active=candidate_active, candidate_inactive_limit=candidate_inactive_limit,
            shared=shared,
        )
        if selected is not None:
            return selected, None, shared[("verify_indexer_visible", id(positions), ratio)]
    tiles, _, _ = _verify_score_tiles(
        query, head_weights, cache=cache, source_layer=source_layer, ratio=ratio,
        positions=positions, valid_mask=valid_mask, fused=True,
        candidates=candidates, candidate_active=candidate_active,
        candidate_block_size=candidate_block_size, shared=shared,
        candidate_inactive_limit=candidate_inactive_limit, topk=topk,
    )
    visible_key = ("verify_indexer_visible", id(positions), ratio)
    if visible_key not in shared:
        shared[visible_key] = (positions + 1) // ratio
    visible = shared[visible_key]
    published = None
    if candidate_topk_blocks > 0:
        # Keep the legacy candidate torch.topk order on equal block scores.
        candidate_tiles = [(logits, lengths, False) for logits, lengths, _ in tiles]
        published = _candidate_tiles(
            candidate_tiles, visible, candidate_block_size, candidate_topk_blocks,
        )
    # shared is required here, so _verify_score_tiles used the explicit
    # BF16-rounding Verify reducer for every journal score.
    selected = _select_verify_topk(
        tiles, visible, valid_mask, topk, bf16_tail_provenance=shared is not None,
    )
    return selected, published, visible


def select_topk(scores, visible, valid_mask, topk) -> torch.Tensor:
    return select_tokens(scores, visible, contract="graph", valid_mask=valid_mask, topk=topk)


def _select_dense_topk(scores, visible, valid_mask, topk) -> torch.Tensor:
    """Select request-local IDs, using ordinal order for short rows."""
    lengths = torch.where(valid_mask, visible.clamp_min(0), 0).to(torch.int32)
    columns = torch.arange(topk, dtype=torch.int32, device=scores.device)[None, :]
    ordered = torch.where(columns < lengths[:, None], columns, -1)
    width = scores.shape[1]
    if scores.is_cuda and topk in (512, 1024, 2048):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        # Graph-local workspace avoids cross-stream reuse between captured
        # batch shapes. The native op initializes it before each launch.
        workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=scores.device)
        tile_scores, tile_indices = [], []
        for start in range(0, width, _TOPK_TILE):
            logits = scores[:, start : start + _TOPK_TILE].contiguous()
            local_lengths = (lengths - start).clamp(0, logits.shape[1])
            selected = torch.empty_like(ordered)
            rtp_llm_ops.dsv4_persistent_topk(
                logits, local_lengths, selected, workspace, topk, logits.shape[1]
            )
            values = logits.gather(1, selected.clamp(0, logits.shape[1] - 1).long())
            keep = (selected >= 0) & (selected < local_lengths[:, None])
            tile_scores.append(values.masked_fill_(~keep, float("-inf")))
            tile_indices.append(selected + start)
        if len(tile_scores) == 1:
            values, selected = tile_scores[0], tile_indices[0]
        else:
            values, indices = torch.cat(tile_scores, dim=1).topk(topk, dim=1)
            selected = torch.cat(tile_indices, dim=1).gather(1, indices)
    else:
        domain = torch.arange(width, device=scores.device)[None, :] < lengths[:, None]
        values, selected = scores.masked_fill(~domain, float("-inf")).topk(
            min(width, topk), dim=1
        )

    selected = torch.where(torch.isfinite(values), selected.to(torch.int32), -1)
    if selected.shape[1] < topk:
        selected = torch.nn.functional.pad(
            selected, (0, topk - selected.shape[1]), value=-1
        )
    return torch.where((lengths <= topk)[:, None], ordered, selected)

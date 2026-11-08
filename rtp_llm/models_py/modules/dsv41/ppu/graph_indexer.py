"""Visible-length Indexer scoring for PPU Decode Graphs."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant

_INDEXER_POOL = "indexer_kv"
_MAX_SPARSE_BLOCKS = 4096
_TOPK_TILE = 65536


def indexer_scores(
    query, head_weights, *, cache, source_layer, ratio, positions, valid_mask
) -> torch.Tensor:
    """Score only the visible prefix, with a fixed-width result for candidates.

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
    visible = ((positions + 1) // ratio).clamp(0, capacity).to(torch.int32)
    visible = torch.where(valid_mask, visible, 0)
    payload, scales = _quant.quantize_indexer_q(query)
    packed_query = (payload.unsqueeze(1), scales.unsqueeze(1))
    weights = head_weights.to(torch.bfloat16).contiguous()
    entry_bytes = _quant.INDEXER_K_ENTRY_BYTES
    packed_pool = pool.as_strided(
        (pool.shape[0], physical_page, 1, entry_bytes),
        (pool.stride(0), entry_bytes, entry_bytes, 1),
    )
    rows = torch.arange(query.shape[0], dtype=torch.int32, device=query.device)
    logical_blocks = logical_page // tile
    physical_blocks = physical_page // tile
    chunk_pages = _MAX_SPARSE_BLOCKS // logical_blocks
    if chunk_pages == 0:
        raise ValueError("PPU Graph Indexer page exceeds the sparse scorer capacity")

    # Metadata, scores and all temporary shapes are fixed at capture. The
    # number of blocks scored per row comes from lengths on every replay.
    chunks = []
    chunk_width = chunk_pages * logical_page
    for start in range(0, capacity, chunk_width):
        width = min(chunk_width, capacity - start)
        first_page = start // logical_page
        last_page = first_page + (width + logical_page - 1) // logical_page
        pages = table[:, first_page:last_page]
        valid_pages = (pages > 0) & (pages < pool.shape[0])
        safe_pages = torch.where(valid_pages, pages, 0).contiguous()
        lengths = (visible - start).clamp(0, width)
        blocks = ((width + tile * 4 - 1) // (tile * 4)) * 4
        block_ids = torch.arange(blocks, dtype=torch.int32, device=query.device)
        block_ids = (
            block_ids // logical_blocks
        ) * physical_blocks + block_ids % logical_blocks
        block_ids = block_ids.unsqueeze(0).expand(query.shape[0], -1).contiguous()
        metadata = get_paged_sparse_mqa_logits_metadata(
            lengths, safe_pages, rows, physical_page, block_ids, torch.int8, tile
        )
        scores = fp8_fp4_paged_sparse_mqa_logits(
            packed_query, packed_pool, weights, metadata, blocks, tile
        )[:, :width].float()
        columns = torch.arange(width, device=query.device)
        page_indices = (columns // logical_page).expand(query.shape[0], -1)
        valid = (columns[None, :] < lengths[:, None]) & valid_pages.gather(
            1, page_indices
        )
        chunks.append(scores.masked_fill_(~valid, float("-inf")))
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)


def verify_indexer_scores(query, head_weights, *, cache, source_layer, ratio, positions, valid_mask):
    """Use paged scoring for committed keys and the journal for tentative keys."""
    from rtp_llm.models_py.modules.dsv41.ppu.graph_attention import score_indexer_entries

    native = cache.native_cache
    first_tentative = cache.prefixes[cache.requests] // ratio
    verify_width = query.shape[0] // cache.prefixes.numel()
    tail_width = (verify_width + ratio - 1) // ratio
    paged_cache = SimpleNamespace(
        pool=native.pool,
        tables={_INDEXER_POOL: native.tables[_INDEXER_POOL][cache.requests]},
        entries=native.entries,
        spans=native.spans,
        max_seq_len=native.max_seq_len,
    )
    committed = indexer_scores(
        query, head_weights, cache=paged_cache, source_layer=source_layer, ratio=ratio,
        positions=first_tentative * ratio - 1, valid_mask=valid_mask,
    )
    offsets = torch.arange(tail_width, dtype=torch.int64, device=query.device)
    tentative = score_indexer_entries(
        query, head_weights, cache=cache, source_layer=source_layer, ratio=ratio,
        positions=positions, valid_mask=valid_mask, entries=first_tentative[:, None] + offsets,
    )
    columns = torch.arange(committed.shape[1], dtype=torch.int64, device=query.device)
    relative = columns - first_tentative[:, None]
    tail = tentative.gather(1, relative.clamp(0, tail_width - 1))
    tail = tail.masked_fill(relative >= tail_width, float("-inf"))
    return torch.where(relative >= 0, tail, committed)


def select_topk(scores, visible, valid_mask, topk) -> torch.Tensor:
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

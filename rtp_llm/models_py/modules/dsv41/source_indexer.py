"""Shared FP4 source tiles for CP prefill's contiguous index-history scan."""

import os
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion, layer_sources
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.compact_writer import encode_compact
from rtp_llm.models_py.modules.dsv41.indexer import (
    CANDIDATE_BLOCKS,
    SPARSE_BLOCK,
    IndexScoreTile,
    _integer,
)

# Includes merging the previous source tile's token/block selections.
SOURCE_QUERY_TILE = 128
_SOURCE_ROWS = CANDIDATE_BLOCKS * SPARSE_BLOCK


@dataclass(frozen=True)
class PreparedIndexSource:
    payload: torch.Tensor
    scales: torch.Tensor
    first_invalid_position: torch.Tensor
    capacity: int

    @property
    def packed_kv_bytes(self):
        return self.payload.numel() + self.scales.numel() * 4


def prepare_index_source(
    pages: CompactPages, page_table: torch.Tensor, *, capacity: int
) -> PreparedIndexSource:
    """Copy one restored source prefix's encoded bytes once before query tiling.

    The caller holds its existing CP page lease through this copy on the current
    stream. Returned payload/scales own their storage and contain no cache views.
    """
    pages.validate(pages.data.device)
    if pages.region != CacheRegion.INDEX_K or pages.entries_per_page % SPARSE_BLOCK:
        raise ValueError("source scoring requires 68-byte INDEX_K pages")
    if page_table.ndim != 2 or page_table.shape[0] != 1 or page_table.shape[1] < 1:
        raise ValueError("a restored source tile must have one request page table")
    _integer(page_table, page_table.shape, pages.data.device)
    if not 0 < capacity <= min(
        _SOURCE_ROWS, page_table.shape[1] * pages.entries_per_page
    ):
        raise ValueError("source capacity must fit one mapped 16K index tile")
    positions = torch.arange(_SOURCE_ROWS, dtype=torch.int32, device=pages.data.device)
    logical = positions.long() // pages.entries_per_page
    ids = page_table[0].index_select(0, logical.clamp_max(page_table.shape[1] - 1))
    invalid = (positions < capacity) & ((ids <= 0) | (ids >= pages.data.shape[0]))
    first_invalid = torch.where(invalid, positions, _SOURCE_ROWS).amin()
    rows = pages.data[:, : pages.entries_per_page * 68].view(
        pages.data.shape[0], pages.entries_per_page, 68
    )
    raw = rows[
        ids.clamp(0, pages.data.shape[0] - 1).long(),
        positions.long() % pages.entries_per_page,
    ]
    raw.masked_fill_((invalid | (positions >= capacity))[:, None], 0)
    return PreparedIndexSource(
        raw[:, :64].contiguous().view(torch.int8),
        raw[:, 64:].contiguous().view(torch.int32).flatten(),
        first_invalid,
        capacity,
    )


def score_index_source(
    query: torch.Tensor,
    weights: torch.Tensor,
    source: PreparedIndexSource,
    visible_lengths: torch.Tensor,
    *,
    layer: int,
    position_offset: int = 0,
) -> IndexScoreTile:
    """Score a contiguous source prefix with the pinned dense FP4 native API.

    This eager CP-prefill interface leaves the selected-block and decode Graph
    path unchanged. FP4 bytes/scales and BF16 head reduction match that path.
    """
    if not layer_sources(layer).scores_queries or layer > 20:
        raise ValueError("shared source scoring is only for source query owners")
    if os.environ.get("DSV41_SPARSE_INDEXER") != "1":
        raise RuntimeError("set DSV41_SPARSE_INDEXER=1 for source scoring")
    if not query.is_cuda or torch.cuda.get_device_capability(query.device)[0] != 10:
        raise RuntimeError("source scoring requires Blackwell")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("shared source scoring is an eager CP-prefill interface")
    if (
        query.ndim != 3
        or query.shape[1:] != (32, 128)
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or not 0 < query.shape[0] <= SOURCE_QUERY_TILE
    ):
        raise ValueError("source Q must be contiguous BF16 [1..128,32,128]")
    count = query.shape[0]
    if (
        weights.shape != (count, 32)
        or weights.dtype != torch.bfloat16
        or weights.device != query.device
        or not weights.is_contiguous()
    ):
        raise ValueError("source weights must be contiguous BF16 [queries,32]")
    if source.payload.device != query.device:
        raise ValueError("source and queries must share the CUDA device")
    if (
        position_offset % SPARSE_BLOCK
        or not 0 <= position_offset <= 1048576 - source.capacity
    ):
        raise ValueError("source position offset must be an aligned model-context prefix")
    _integer(visible_lengths, (count,), query.device)
    import deep_gemm

    encoded = encode_compact(query.view(-1, 128), CacheRegion.INDEX_K)
    payload = encoded.output[:, :64].contiguous().view(torch.int8).view(count, 32, 64)
    scales = encoded.output[:, 64:].contiguous().view(torch.int32).view(count, 32)
    invalid = encoded.status.view(count, 32).ne(0).any(-1)
    invalid |= ~torch.isfinite(weights).all(-1)
    invalid |= (visible_lengths < 0) | (visible_lengths > source.capacity)
    invalid |= visible_lengths > source.first_invalid_position
    lengths = torch.where(invalid, 0, visible_lengths)
    starts = torch.zeros_like(lengths)
    metadata = deep_gemm.get_mqa_logits_metadata(starts, lengths, _SOURCE_ROWS, 32)
    logits = deep_gemm.fp8_fp4_mqa_logits(
        (payload, scales),
        (source.payload, source.scales),
        torch.where(invalid[:, None], 0, weights),
        starts,
        lengths,
        clean_logits=True,
        logits_dtype=torch.bfloat16,
        schedule_meta=metadata,
    )
    del encoded, payload, scales, metadata
    positions = torch.arange(_SOURCE_ROWS, dtype=torch.int32, device=query.device)[
        None, :
    ]
    masked = positions >= lengths[:, None]
    numeric_error = (~masked & ~torch.isfinite(logits)).any(-1)
    masked |= numeric_error[:, None]
    status = (invalid | numeric_error).to(torch.int32)
    positions = torch.where(masked, -1, positions + position_offset)
    logits.masked_fill_(masked, -torch.inf)
    blocks = torch.arange(CANDIDATE_BLOCKS, dtype=torch.int32, device=query.device)[
        None, :
    ]
    blocks = torch.where(
        blocks * SPARSE_BLOCK < lengths[:, None],
        blocks + position_offset // SPARSE_BLOCK,
        -1,
    )
    return IndexScoreTile(
        logits, positions, blocks, lengths, status, source.packed_kv_bytes
    )

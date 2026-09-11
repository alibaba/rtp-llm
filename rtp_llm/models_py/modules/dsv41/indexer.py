"""Bounded-workspace V4.1 sparse index scoring on the pinned DeepGEMM API.

Q is the projected, post-RoPE BF16 index query; weights already include the
official head/dimension scaling. Only query owners may call this component.
CP gather/ready and forward/tail lifetimes belong to the explicit caller.
This is an opt-in component, not a release-qualified backend selection.
The selected-block planar scratch uses at most 64 MiB for a 32-query tile;
DeepGEMM metadata, stream workspace and Torch selection buffers are additional.
"""

import os
from dataclasses import dataclass, replace
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv41._indexer_triton import repack_index_blocks
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion, layer_sources
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.compact_writer import encode_compact

QUERY_TILE = 32
CANDIDATE_BLOCKS = 2048
SPARSE_BLOCK = 8
INDEX_TOPK = 512


@dataclass(frozen=True)
class RankedPositions:
    scores: torch.Tensor
    positions: torch.Tensor

    def ordered_positions(self) -> torch.Tensor:
        sentinel = torch.iinfo(torch.int32).max
        ids = torch.where(self.scores > -torch.inf, self.positions, sentinel)
        ids = ids.sort(dim=-1).values
        return torch.where(ids == sentinel, -1, ids).to(torch.int32)


def _retain(
    scores: torch.Tensor,
    positions: torch.Tensor,
    capacity: int,
    previous: Optional[RankedPositions] = None,
) -> RankedPositions:
    if previous is not None:
        scores = torch.cat((previous.scores, scores), dim=-1)
        positions = torch.cat((previous.positions, positions), dim=-1)
    if os.environ.get("DSV41_DEEPSELECT") == "1":
        from rtp_llm.models_py.modules.dsv41.deepselect import topk

        selected = topk(scores, min(capacity, scores.shape[-1]))
        torch._assert_async(
            (selected.status == 0).all(), "invalid DeepSelect index selection"
        )
        values, offsets = selected.values, selected.indices.clamp_min(0).long()
    else:
        values, offsets = scores.topk(min(capacity, scores.shape[-1]), dim=-1)
    return RankedPositions(values, positions.gather(-1, offsets))


@dataclass(frozen=True)
class IndexScoreTile:
    logits: torch.Tensor
    positions: torch.Tensor
    candidate_blocks: torch.Tensor
    visible_lengths: torch.Tensor
    status: torch.Tensor
    packed_kv_bytes: int = 0

    def topk(self, previous: Optional[RankedPositions] = None) -> RankedPositions:
        scores = (
            self.logits
            if os.environ.get("DSV41_DEEPSELECT") == "1"
            else self.logits.float()
        )
        return _retain(scores, self.positions, INDEX_TOPK, previous)

    def block_topk(self, previous: Optional[RankedPositions] = None) -> RankedPositions:
        scores = self.logits.float().unflatten(-1, (-1, SPARSE_BLOCK)).amax(-1)
        # The official selector pins the newest reachable block, including a
        # partial block whose future slots are still masked by the scorer.
        newest = (self.visible_lengths[:, None] - 1) // SPARSE_BLOCK
        scores = torch.where(
            (self.candidate_blocks == newest)
            & (self.visible_lengths[:, None] > 0)
            & (self.status[:, None] == 0),
            torch.inf,
            scores,
        )
        return _retain(scores, self.candidate_blocks, CANDIDATE_BLOCKS, previous)


@dataclass(frozen=True)
class IndexSelection:
    topk: torch.Tensor
    candidate_blocks: Optional[torch.Tensor]
    status: torch.Tensor
    query_owner: int
    key_owner: int
    scorer_calls: int
    max_logits_elements: int
    max_packed_kv_bytes: int

    def check(self) -> None:
        with torch.cuda.device(self.status.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("check index status only after Graph execution")
        if torch.any(self.status.detach().cpu() != 0):
            raise RuntimeError(
                "indexer rejected numeric, query, candidate or page metadata"
            )


def _integer(tensor, shape, device):
    if (
        tensor.dtype != torch.int32
        or tensor.device != device
        or tuple(tensor.shape) != tuple(shape)
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "index metadata must be contiguous CUDA int32 of the declared shape"
        )


def _validate(query, weights, pages, page_table, request_ids, visible_lengths, layer):
    source = layer_sources(layer)
    if not source.scores_queries:
        raise ValueError("only the eight V4.1 index query owners may score queries")
    if os.environ.get("DSV41_SPARSE_INDEXER") != "1":
        raise RuntimeError("set DSV41_SPARSE_INDEXER=1 for this component")
    if not query.is_cuda or torch.cuda.get_device_capability(query.device)[0] != 10:
        raise RuntimeError("V4.1 sparse scoring requires Blackwell")
    if (
        query.ndim != 3
        or query.shape[1:] != (32, 128)
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
    ):
        raise ValueError("index Q must be contiguous BF16 [queries,32,128]")
    rows = query.shape[0]
    if rows > QUERY_TILE:
        raise ValueError("tile index queries into at most 32 rows")
    if (
        weights.shape != (rows, 32)
        or weights.dtype != torch.bfloat16
        or weights.device != query.device
        or not weights.is_contiguous()
    ):
        raise ValueError("scaled index weights must be contiguous BF16 [queries,32]")
    pages.validate(query.device)
    if pages.region != CacheRegion.INDEX_K or pages.entries_per_page % SPARSE_BLOCK:
        raise ValueError(
            "index scorer requires native 68-byte index pages divisible by 8"
        )
    if pages.data.stride(0) > torch.iinfo(torch.int32).max:
        raise ValueError("index physical page stride exceeds the DeepGEMM ABI")
    if page_table.ndim != 2 or min(page_table.shape) <= 0:
        raise ValueError("index page table must contain request rows and logical pages")
    _integer(page_table, page_table.shape, query.device)
    _integer(request_ids, (rows,), query.device)
    _integer(visible_lengths, (rows,), query.device)
    return source


def score_candidate_tile(
    query: torch.Tensor,
    weights: torch.Tensor,
    pages: CompactPages,
    page_table: torch.Tensor,
    request_ids: torch.Tensor,
    visible_lengths: torch.Tensor,
    candidate_blocks: torch.Tensor,
    *,
    layer: int,
) -> IndexScoreTile:
    """Score only selected logical blocks; refresh metadata on every replay.

    Request IDs index page_table and must be nondecreasing so queries of one
    request remain consecutive. DSpark Bx6 is flattened into query rows with
    separate visible lengths; DeepGEMM always receives next_n=1. Invalid rows
    are sanitized before the unchecked vendor kernel and return nonzero status.
    """
    _validate(query, weights, pages, page_table, request_ids, visible_lengths, layer)
    rows = query.shape[0]
    _integer(candidate_blocks, (rows, CANDIDATE_BLOCKS), query.device)
    if rows == 0:
        return IndexScoreTile(
            torch.empty(
                (0, CANDIDATE_BLOCKS * SPARSE_BLOCK),
                dtype=torch.bfloat16,
                device=query.device,
            ),
            torch.empty(
                (0, CANDIDATE_BLOCKS * SPARSE_BLOCK),
                dtype=torch.int32,
                device=query.device,
            ),
            candidate_blocks,
            visible_lengths,
            torch.empty_like(visible_lengths),
        )
    import deep_gemm

    encoded = encode_compact(query.view(-1, 128), CacheRegion.INDEX_K)
    # This pinned DeepGEMM ABI uses int8 as the carrier for two E2M1 values.
    packed_q = encoded.output[:, :64].contiguous().view(torch.int8)
    packed_sf = encoded.output[:, 64:].contiguous().view(torch.int32)
    invalid = encoded.status.view(rows, 32).ne(0).any(-1) | ~torch.isfinite(
        weights
    ).all(-1)
    invalid |= (request_ids < 0) | (request_ids >= page_table.shape[0])
    invalid |= (visible_lengths < 0) | (
        visible_lengths > min(1048576, page_table.shape[1] * pages.entries_per_page)
    )
    # A schedule may pair adjacent queries only when their request is identical.
    wrong_order = (request_ids[1:] < request_ids[:-1]).any()
    invalid |= wrong_order
    valid = candidate_blocks >= 0
    upper = (visible_lengths[:, None] + SPARSE_BLOCK - 1) // SPARSE_BLOCK
    invalid |= ((candidate_blocks < -1) | (valid & (candidate_blocks >= upper))).any(-1)
    # The vendor metadata infers this count from its KV range, not -1 padding.
    invalid |= valid.sum(-1) != upper[:, 0].clamp(0, CANDIDATE_BLOCKS)
    invalid |= (
        valid[:, 1:]
        & ((candidate_blocks[:, 1:] <= candidate_blocks[:, :-1]) | ~valid[:, :-1])
    ).any(-1)
    table = page_table.index_select(
        0, request_ids.clamp(0, page_table.shape[0] - 1).long()
    )
    logical_pages = (
        candidate_blocks.clamp_min(0).long() * SPARSE_BLOCK // pages.entries_per_page
    )
    selected_pages = table.gather(1, logical_pages.clamp_max(page_table.shape[1] - 1))
    invalid |= (
        valid & ((selected_pages <= 0) | (selected_pages >= pages.data.shape[0]))
    ).any(-1)
    status = invalid.to(torch.int32)
    candidates = torch.where(invalid[:, None], -1, candidate_blocks)
    lengths = torch.where(invalid, 0, visible_lengths)
    # Native storage is row-interleaved. DeepGEMM instead reads all payloads
    # then all scales within a page. Copy selected 8-row blocks without quantizing.
    packed_cache = torch.empty(
        (rows * CANDIDATE_BLOCKS, 1024), dtype=torch.uint8, device=query.device
    )
    repack_index_blocks[(rows, CANDIDATE_BLOCKS)](
        pages.data,
        selected_pages,
        candidates,
        lengths,
        packed_cache,
        CANDIDATES=CANDIDATE_BLOCKS,
        PAGE_STRIDE=pages.data.stride(0),
        ENTRIES=pages.entries_per_page,
        num_warps=4,
    )
    packed_count = (candidates >= 0).sum(-1, dtype=torch.int32)
    packed_slots = torch.arange(
        CANDIDATE_BLOCKS, dtype=torch.int32, device=query.device
    )[None, :].expand(rows, -1)
    packed_candidates = torch.where(
        packed_slots < packed_count[:, None], packed_slots, -1
    ).contiguous()
    packed_table = torch.arange(
        rows * CANDIDATE_BLOCKS, dtype=torch.int32, device=query.device
    ).view(rows, CANDIDATE_BLOCKS)
    # Each query owns a separate packed context; adjacent queries cannot share
    # the vendor's paired-query page table. Original request/position IDs stay above.
    metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
        packed_count * SPARSE_BLOCK,
        packed_table,
        torch.arange(rows, dtype=torch.int32, device=query.device),
        SPARSE_BLOCK,
        packed_candidates,
        packed_q.dtype,
        SPARSE_BLOCK,
    )
    cache = packed_cache.as_strided(
        (rows * CANDIDATE_BLOCKS, SPARSE_BLOCK, 1, 68),
        (1024, 68, 68, 1),
    )
    logits = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        (packed_q.view(rows, 1, 32, 64), packed_sf.view(rows, 1, 32)),
        cache,
        torch.where(invalid[:, None], 0, weights),
        metadata,
        CANDIDATE_BLOCKS,
        SPARSE_BLOCK,
    )
    offsets = torch.arange(SPARSE_BLOCK, dtype=torch.int32, device=query.device)
    positions = (candidates[:, :, None] * SPARSE_BLOCK + offsets).flatten(1)
    valid_positions = (
        (candidates[:, :, None] >= 0).expand(-1, -1, SPARSE_BLOCK).flatten(1)
    )
    valid_positions &= positions < lengths[:, None]
    numeric_error = (valid_positions & ~torch.isfinite(logits)).any(-1)
    status = torch.maximum(status, numeric_error.to(torch.int32))
    valid_positions &= ~numeric_error[:, None]
    positions = torch.where(valid_positions, positions, -1)
    logits = torch.where(valid_positions, logits, -torch.inf)
    return IndexScoreTile(
        logits, positions, candidates, lengths, status, packed_cache.numel()
    )


def select_index_positions(
    query: torch.Tensor,
    weights: torch.Tensor,
    pages: CompactPages,
    page_table: torch.Tensor,
    request_ids: torch.Tensor,
    visible_lengths: torch.Tensor,
    *,
    layer: int,
    max_visible_length: int,
    candidate_blocks: Optional[torch.Tensor] = None,
) -> IndexSelection:
    """Source scan or candidate-only reindex with O(queries*16384) score storage.

    max_visible_length is the scheduler's fixed capacity for this invocation or
    Graph bucket. It bounds scans, never the recorded real request length.
    Every source tile is consumed before the next; no history-sized logits or
    decoded K tensor is retained. L20 also returns sorted 2048 block candidates.
    """
    source = _validate(
        query, weights, pages, page_table, request_ids, visible_lengths, layer
    )
    if (
        not 0
        < max_visible_length
        <= min(1048576, page_table.shape[1] * pages.entries_per_page)
    ):
        raise ValueError("max_visible_length must fit the mapped model context")
    if (layer > 20) != (candidate_blocks is not None):
        raise ValueError(
            "reindex owners require L20 candidates; source owners scan their own keys"
        )
    rows = query.shape[0]
    top = blocks = None
    max_packed_kv_bytes = 0
    status = ((visible_lengths > max_visible_length) | (visible_lengths < 0)).to(
        torch.int32
    )
    count = (
        1
        if candidate_blocks is not None
        else (max_visible_length + CANDIDATE_BLOCKS * SPARSE_BLOCK - 1)
        // (CANDIDATE_BLOCKS * SPARSE_BLOCK)
    )
    for tile in range(count):
        tile_first = 0
        tile_table = page_table
        tile_lengths = visible_lengths
        if candidate_blocks is None:
            tile_first = tile * CANDIDATE_BLOCKS * SPARSE_BLOCK
            tile_end = min(
                tile_first + CANDIDATE_BLOCKS * SPARSE_BLOCK, max_visible_length
            )
            if tile_first % pages.entries_per_page:
                raise ValueError("source score tiles must align with index pages")
            tile_table = page_table[
                :,
                tile_first
                // pages.entries_per_page : (tile_end + pages.entries_per_page - 1)
                // pages.entries_per_page,
            ].contiguous()
            tile_lengths = (visible_lengths - tile_first).clamp(
                0, tile_end - tile_first
            )
            ids = torch.arange(CANDIDATE_BLOCKS, dtype=torch.int32, device=query.device)
            ids = ids[None, :].expand(rows, -1)
            ids = torch.where(
                ids * SPARSE_BLOCK < tile_lengths[:, None], ids, -1
            ).contiguous()
        else:
            ids = candidate_blocks
        scores = score_candidate_tile(
            query,
            weights,
            pages,
            tile_table,
            request_ids,
            tile_lengths,
            ids,
            layer=layer,
        )
        if candidate_blocks is None:
            scores = replace(
                scores,
                positions=torch.where(
                    scores.positions >= 0, scores.positions + tile_first, -1
                ),
                candidate_blocks=torch.where(
                    scores.candidate_blocks >= 0,
                    scores.candidate_blocks + tile_first // SPARSE_BLOCK,
                    -1,
                ),
                visible_lengths=visible_lengths,
            )
        top = scores.topk(top)
        if layer == 20:
            blocks = scores.block_topk(blocks)
        status = torch.maximum(status, scores.status)
        max_packed_kv_bytes = max(max_packed_kv_bytes, scores.packed_kv_bytes)
    return IndexSelection(
        top.ordered_positions(),
        blocks.ordered_positions() if blocks is not None else None,
        status,
        layer,
        source.index_k_owner,
        count if rows else 0,
        rows * CANDIDATE_BLOCKS * SPARSE_BLOCK,
        max_packed_kv_bytes,
    )

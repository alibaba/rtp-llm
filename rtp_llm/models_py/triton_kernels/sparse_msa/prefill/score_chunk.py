# Copyright 2025. All rights reserved.
"""Shared query-row chunking and workspace helpers for M3 prefill scoring."""

import logging
import os
from typing import NamedTuple, Optional, Sequence, Tuple

import torch


M3_INDEX_SCORE_CHUNK_ROWS_ENV = "M3_MSA_INDEX_SCORE_CHUNK_ROWS"
_DEFAULT_INDEX_SCORE_CHUNK_ROWS = 0
_WORKSPACE_ALIGNMENT = 256


class PrefillScoreHostMetadata(NamedTuple):
    query_lens: Tuple[int, ...]
    seq_lens: Tuple[int, ...]
    prefix_lens: Tuple[int, ...]
    slot_ids: Tuple[int, ...]


class PrefillScoreChunk(NamedTuple):
    q_start: int
    q_end: int
    cu_seqlens: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    slot_ids: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    kv_indices: Optional[torch.Tensor]
    host_metadata: PrefillScoreHostMetadata


# One flat allocation per device is shared by index score chunks and sparse
# attention step3. Both phases execute serially on the current CUDA stream.
M3_PREFILL_WORKSPACE_CACHE: dict = {}


def m3_index_score_chunk_rows() -> int:
    raw = os.environ.get(
        M3_INDEX_SCORE_CHUNK_ROWS_ENV, str(_DEFAULT_INDEX_SCORE_CHUNK_ROWS)
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logging.warning(
            "[M3 index score] invalid %s=%r; using default=%d",
            M3_INDEX_SCORE_CHUNK_ROWS_ENV,
            raw,
            _DEFAULT_INDEX_SCORE_CHUNK_ROWS,
        )
        value = _DEFAULT_INDEX_SCORE_CHUNK_ROWS
    return max(value, 0)


def m3_index_score_chunk_enabled(total_q: int, block_size_q: int = 1) -> bool:
    rows = m3_index_score_chunk_rows()
    return block_size_q == 1 and rows > 0 and total_q > rows


def get_or_create_m3_prefill_workspace(
    nbytes: int, device: torch.device
) -> torch.Tensor:
    workspace = M3_PREFILL_WORKSPACE_CACHE.get(device)
    if workspace is None or workspace.numel() < nbytes:
        workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
        M3_PREFILL_WORKSPACE_CACHE[device] = workspace
    return workspace


def _align_up(value: int, alignment: int = _WORKSPACE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def get_float32_workspace_views(
    device: torch.device, shapes: Sequence[Tuple[int, ...]]
) -> Tuple[torch.Tensor, ...]:
    offsets = []
    offset = 0
    for shape in shapes:
        offset = _align_up(offset)
        offsets.append(offset)
        numel = 1
        for dim in shape:
            numel *= int(dim)
        offset += numel * torch.float32.itemsize

    workspace = get_or_create_m3_prefill_workspace(offset, device)
    views = []
    for byte_offset, shape in zip(offsets, shapes):
        numel = 1
        for dim in shape:
            numel *= int(dim)
        view = workspace[byte_offset : byte_offset + numel * 4].view(torch.float32)
        views.append(view.view(shape))
    return tuple(views)


def _pack_segments_into_chunks(
    query_lens: Sequence[int], chunk_rows: int
) -> list[list[tuple[int, int, int]]]:
    groups = []
    current = []
    current_rows = 0
    for batch_idx, query_len in enumerate(query_lens):
        query_start = 0
        while query_start < query_len:
            take = min(chunk_rows - current_rows, query_len - query_start)
            current.append((batch_idx, query_start, query_start + take))
            current_rows += take
            query_start += take
            if current_rows == chunk_rows:
                groups.append(current)
                current = []
                current_rows = 0
    if current:
        groups.append(current)
    return groups


def resolve_prefill_score_host_metadata(
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    slot_ids: Optional[torch.Tensor] = None,
    host_metadata: Optional[PrefillScoreHostMetadata] = None,
) -> PrefillScoreHostMetadata:
    """Return validated host geometry, falling back to device reads if needed."""
    if host_metadata is None:
        query_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
        seq_lens_host = seq_lens.cpu().tolist()
        prefix_lens_host = prefix_lens.cpu().tolist()
        slot_ids_host = (
            slot_ids.cpu().tolist()
            if slot_ids is not None
            else range(len(query_lens))
        )
        host_metadata = PrefillScoreHostMetadata(
            query_lens=tuple(int(value) for value in query_lens),
            seq_lens=tuple(int(value) for value in seq_lens_host),
            prefix_lens=tuple(int(value) for value in prefix_lens_host),
            slot_ids=tuple(int(value) for value in slot_ids_host),
        )
    segment_count = len(host_metadata.query_lens)
    if not (
        len(host_metadata.seq_lens)
        == len(host_metadata.prefix_lens)
        == len(host_metadata.slot_ids)
        == segment_count
    ):
        raise ValueError(
            "prefill score host metadata length mismatch: "
            f"query={segment_count}, seq={len(host_metadata.seq_lens)}, "
            f"prefix={len(host_metadata.prefix_lens)}, "
            f"slots={len(host_metadata.slot_ids)}"
        )
    return host_metadata


def build_prefill_score_chunks(
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    slot_ids: Optional[torch.Tensor],
    chunk_rows: int,
    block_size_k: int,
    kv_indices: Optional[torch.Tensor] = None,
    *,
    host_metadata: Optional[PrefillScoreHostMetadata] = None,
) -> list[PrefillScoreChunk]:
    """Build packed varlen metadata for query-row score chunks.

    Each chunk covers all K blocks for its query rows. A split segment advances
    its prefix by the segment-local row offset, preserving causal positions.
    """
    device = cu_seqlens.device
    host_metadata = resolve_prefill_score_host_metadata(
        cu_seqlens, seq_lens, prefix_lens, slot_ids, host_metadata
    )
    query_lens = host_metadata.query_lens
    slot_dtype = slot_ids.dtype if slot_ids is not None else torch.int64

    page_runs = []
    page_offset = 0
    if kv_indices is not None:
        for seq_len in host_metadata.seq_lens:
            page_count = (seq_len + block_size_k - 1) // block_size_k
            page_runs.append((page_offset, page_count))
            page_offset += page_count

    query_offsets = [0]
    for query_len in query_lens:
        query_offsets.append(query_offsets[-1] + int(query_len))

    chunks = []
    for group in _pack_segments_into_chunks(query_lens, chunk_rows):
        query_start = query_offsets[group[0][0]] + group[0][1]
        query_end = query_start + sum(end - start for _, start, end in group)
        local_query_lens = [end - start for _, start, end in group]
        local_cu_seqlens = [0]
        for query_len in local_query_lens:
            local_cu_seqlens.append(local_cu_seqlens[-1] + query_len)

        local_seq_lens = [
            host_metadata.seq_lens[batch_idx] for batch_idx, _, _ in group
        ]
        local_prefix_lens = [
            host_metadata.prefix_lens[batch_idx] + start
            for batch_idx, start, _ in group
        ]
        local_slot_ids = [
            host_metadata.slot_ids[batch_idx] for batch_idx, _, _ in group
        ]

        local_kv_indices = None
        if kv_indices is not None:
            page_slices = []
            for batch_idx, _, _ in group:
                offset, count = page_runs[batch_idx]
                page_slices.append(kv_indices[offset : offset + count])
            local_kv_indices = torch.cat(page_slices) if page_slices else kv_indices[:0]

        chunks.append(
            PrefillScoreChunk(
                q_start=query_start,
                q_end=query_end,
                cu_seqlens=torch.tensor(
                    local_cu_seqlens, dtype=torch.int32, device=device
                ),
                seq_lens=torch.tensor(local_seq_lens, dtype=torch.int32, device=device),
                prefix_lens=torch.tensor(
                    local_prefix_lens, dtype=torch.int32, device=device
                ),
                slot_ids=torch.tensor(
                    local_slot_ids, dtype=slot_dtype, device=device
                ),
                max_seqlen_q=max(local_query_lens),
                max_seqlen_k=max(local_seq_lens),
                kv_indices=local_kv_indices,
                host_metadata=PrefillScoreHostMetadata(
                    query_lens=tuple(local_query_lens),
                    seq_lens=tuple(local_seq_lens),
                    prefix_lens=tuple(local_prefix_lens),
                    slot_ids=tuple(local_slot_ids),
                ),
            )
        )
    return chunks

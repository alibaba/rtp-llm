"""Mean-pool compression for the qwen4 QSA indexer's KV + state pools.

The DSv4 path uses ``CompressorFP8`` (1448 lines, weight-based: ``{ape,wkv,wgate,norm}``).
This module is the parameter-free replacement the qwen4 checkpoint ships: raw keys are
averaged in fp32 over each ``compress_ratio``-sized block, then normed, partial-roped,
cast to bf16, and written to the ``indexer_kv`` pool as a 256-byte entry for the
released 128-wide indexer head. This is the same format consumed by
``qsa_paged_indexer_score``; there is no inline scale in the production ABI.

The ``indexer_state`` pool retains the raw keys of the incomplete block so the next
block's mean can be computed (the checkpoint does not keep a running sum; we keep fp32
raw keys via the state ring to reproduce the reference's bit-exact summation order).

The functions here are **production-shaped but pool-agnostic**: they accept a flat buffer
(view of the ``indexer_kv`` / ``indexer_state`` pool tensors) and return metadata about
where entries were written, so the caller can set up the actual pool views through
``PoolBackedModule.set_pool_context`` and call these functions with the real tensors.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import apply_partial_rope

InvalidBlockPolicy = Literal["raise", "skip"]


@dataclass(frozen=True)
class IndexerCacheUndo:
    """Original values for the unique side-cache slots touched by one write."""

    state_pool: torch.Tensor
    state_slots: torch.Tensor
    original_state: torch.Tensor
    kv_pool: torch.Tensor
    kv_slots: torch.Tensor
    original_kv: torch.Tensor
    padded_slots: bool = False


def _capture_pool_slots(
    pool: torch.Tensor, destinations: list[tuple[int, int, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    entries_per_block = int(pool.shape[1])
    # A long prefill can wrap the state ring and touch one destination more than
    # once. Preserve the value that existed before the first write only.
    unique_slots = list(
        dict.fromkeys(
            block_id * entries_per_block + offset
            for block_id, offset, _ in destinations
        )
    )
    slots = torch.tensor(unique_slots, dtype=torch.long, device=pool.device)
    original = pool.flatten(0, 1).index_select(0, slots)
    return slots, original


def restore_indexer_cache(undo: IndexerCacheUndo) -> None:
    """Restore a captured side-cache write on the caller's current stream."""
    if undo.padded_slots:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill import (
            restore_padded_indexer_cache,
        )

        restore_padded_indexer_cache(undo)
        return
    undo.state_pool.flatten(0, 1).index_copy_(0, undo.state_slots, undo.original_state)
    undo.kv_pool.flatten(0, 1).index_copy_(0, undo.kv_slots, undo.original_kv)


def _physical_block(
    block_table: torch.Tensor,
    request_idx: int,
    logical_block: int,
    *,
    num_physical_blocks: int,
    tag: str,
    invalid_block_policy: InvalidBlockPolicy,
) -> Optional[int]:
    """Resolve one request-local block-table entry without flattening batches."""
    table_width = int(block_table.shape[1])
    if logical_block < 0 or logical_block >= table_width:
        if invalid_block_policy == "skip":
            return None
        raise ValueError(
            f"{tag} logical block {logical_block} is outside block_table width "
            f"{table_width} for request {request_idx}"
        )
    block_id = int(block_table[request_idx, logical_block].item())
    if block_id <= 0:
        if invalid_block_policy == "skip":
            return None
        raise ValueError(
            f"{tag} block_id must be > 0 for request {request_idx}, logical "
            f"block {logical_block}; got {block_id}"
        )
    if block_id >= num_physical_blocks:
        raise IndexError(
            f"{tag} physical block {block_id} is outside pool size "
            f"{num_physical_blocks}"
        )
    return block_id


def _last_write_per_slot(
    slots: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate destinations while preserving sequential last-write semantics."""
    if not int(slots.numel()):
        return slots, values
    order = torch.argsort(slots, stable=True)
    sorted_slots = slots.index_select(0, order)
    is_last = torch.ones_like(sorted_slots, dtype=torch.bool)
    is_last[:-1] = sorted_slots[:-1] != sorted_slots[1:]
    last_order = order[is_last]
    return slots.index_select(0, last_order), values.index_select(0, last_order)


def _commit_indexer_cache(
    state_pool: torch.Tensor,
    state_slots: torch.Tensor,
    state_values: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_slots: torch.Tensor,
    kv_values: torch.Tensor,
    *,
    capture_undo: bool,
) -> IndexerCacheUndo | None:
    flat_state, flat_kv = state_pool.flatten(0, 1), kv_pool.flatten(0, 1)
    undo = None
    if capture_undo:
        # The write slots are already unique. index_select owns the captured
        # values, so subsequent writes cannot change either snapshot.
        undo = IndexerCacheUndo(
            state_pool=state_pool,
            state_slots=state_slots,
            original_state=flat_state.index_select(0, state_slots),
            kv_pool=kv_pool,
            kv_slots=kv_slots,
            original_kv=flat_kv.index_select(0, kv_slots),
        )
    try:
        flat_state.index_copy_(0, state_slots, state_values)
        flat_kv.index_copy_(0, kv_slots, kv_values)
    except BaseException:
        if undo is not None:
            restore_indexer_cache(undo)
            if state_pool.is_cuda:
                torch.cuda.current_stream(state_pool.device).synchronize()
        raise
    return undo


@dataclass(frozen=True)
class AlignedPrefillWriteLayout:
    # Page tensors encode (request, logical page). Write tensors encode
    # (index within the page tensor, slot offset, packed token row).
    state_pages: torch.Tensor
    kv_pages: torch.Tensor
    state_writes: torch.Tensor
    kv_writes: torch.Tensor


@dataclass(frozen=True)
class AlignedPrefillWritePlan:
    layout: AlignedPrefillWriteLayout
    state_blocks: torch.Tensor
    kv_blocks: torch.Tensor


def _packed_indices(
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = counts.cumsum() - counts
    requests = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    rows = np.arange(requests.size, dtype=np.int64) - starts[requests]
    return requests, rows, starts


def _build_aligned_prefill_write_layout(
    lengths: Sequence[int],
    prefixes: Sequence[int],
    *,
    page_size: int,
    ratio: int,
    ring_entries: int,
    device: torch.device,
) -> AlignedPrefillWriteLayout:
    """Build fixed-size write indices from a validated, page-aligned partition."""
    length_array = np.asarray(lengths, dtype=np.int64)
    prefix_pages = np.asarray(prefixes, dtype=np.int64) // page_size
    packed_starts = length_array.cumsum() - length_array
    full_pages, tail = np.divmod(length_array, page_size)
    kv_entries = page_size // ratio
    groups = length_array // ratio
    state_requests, state_columns, state_page_starts = _packed_indices(
        full_pages + (tail > 0)
    )
    kv_requests, kv_columns, kv_page_starts = _packed_indices(
        (groups + kv_entries - 1) // kv_entries
    )
    requests, rows, _ = _packed_indices(
        full_pages * ring_entries + np.minimum(tail, ring_entries)
    )
    pages, offsets = np.divmod(rows, ring_entries)
    page_start = pages * page_size
    page_length = np.minimum(length_array[requests] - page_start, page_size)
    # Keep the final writes to the ring of each full or partial page.
    rows = np.maximum(page_length - ring_entries, 0) + offsets
    state_writes = (
        state_page_starts[requests] + pages,
        rows % ring_entries,
        packed_starts[requests] + page_start + rows,
    )
    requests, group, _ = _packed_indices(groups)
    pages, offsets = np.divmod(group, kv_entries)
    kv_writes = (
        kv_page_starts[requests] + pages,
        offsets,
        packed_starts[requests] + (group + 1) * ratio - 1,
    )
    parts = (
        (state_requests, prefix_pages[state_requests] + state_columns),
        (kv_requests, prefix_pages[kv_requests] + kv_columns),
        state_writes,
        kv_writes,
    )
    sizes = [len(part) * part[0].size for part in parts]
    packed = torch.empty(
        sum(sizes), dtype=torch.long, device="cpu", pin_memory=device.type == "cuda"
    )
    host = packed.numpy()
    start = 0
    for part in parts:
        for field in part:
            end = start + field.size
            host[start:end] = field
            start = end
    state_pages, kv_pages, state_writes, kv_writes = packed.to(
        device=device, non_blocking=True
    ).split(sizes)
    return AlignedPrefillWriteLayout(
        state_pages.reshape(2, -1),
        kv_pages.reshape(2, -1),
        state_writes.reshape(3, -1),
        kv_writes.reshape(3, -1),
    )


@lru_cache(maxsize=16)
def _cached_aligned_prefill_write_layout(
    lengths: tuple[int, ...],
    prefixes: tuple[int, ...],
    page_size: int,
    ratio: int,
    ring_entries: int,
    stream: torch.cuda.Stream,
) -> AlignedPrefillWriteLayout:
    # Allocate, fill and consume the immutable index tensors on the same stream.
    with torch.cuda.stream(stream):
        return _build_aligned_prefill_write_layout(
            lengths,
            prefixes,
            page_size=page_size,
            ratio=ratio,
            ring_entries=ring_entries,
            device=stream.device,
        )


def build_aligned_prefill_write_layout(
    lengths: list[int],
    prefixes: list[int],
    *,
    page_size: int,
    ratio: int,
    ring_entries: int,
    device: torch.device,
) -> AlignedPrefillWriteLayout:
    # Bound the cache to 4 MiB of indices. Layouts contain logical coordinates
    # only, so page remapping never invalidates an entry.
    entries = 0
    kv_entries = page_size // ratio
    for length in lengths:
        pages, tail = divmod(length, page_size)
        groups = length // ratio
        entries += 2 * (pages + (tail > 0) + (groups + kv_entries - 1) // kv_entries)
        entries += 3 * (pages * ring_entries + min(tail, ring_entries) + groups)
    if device.type == "cuda" and entries * 8 <= 256 * 1024:
        return _cached_aligned_prefill_write_layout(
            tuple(lengths),
            tuple(prefixes),
            page_size,
            ratio,
            ring_entries,
            torch.cuda.current_stream(device),
        )
    return _build_aligned_prefill_write_layout(
        lengths,
        prefixes,
        page_size=page_size,
        ratio=ratio,
        ring_entries=ring_entries,
        device=device,
    )


def write_aligned_prefill_cache(
    raw_keys: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_norm_gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    state_pool: torch.Tensor,
    plan: AlignedPrefillWritePlan,
    *,
    ratio: int,
    norm_eps: float,
) -> IndexerCacheUndo | None:
    """Write the unique slots established by this layer's preflight."""
    layout = plan.layout
    state_pages, state_offsets, state_rows = layout.state_writes.unbind(0)
    state_slots = (
        plan.state_blocks.to(torch.long).index_select(0, state_pages)
        * state_pool.shape[1]
        + state_offsets
    )
    state_values = raw_keys.index_select(0, state_rows).float()
    kv_pages, kv_offsets, group_ends = layout.kv_writes.unbind(0)
    kv_slots = (
        plan.kv_blocks.to(torch.long).index_select(0, kv_pages) * kv_pool.shape[1]
        + kv_offsets
    )
    kv_values = raw_keys.new_empty((0, raw_keys.shape[1]), dtype=torch.bfloat16)
    if group_ends.numel():
        source_offsets = torch.arange(
            1 - ratio, 1, dtype=torch.long, device=raw_keys.device
        )
        source_rows = group_ends[:, None] + source_offsets
        gathered = raw_keys.index_select(0, source_rows.flatten()).reshape(
            -1, ratio, raw_keys.shape[1]
        )
        pooled = gathered.float().mean(dim=1).to(raw_keys.dtype)
        pooled = exact_head_rms_norm(pooled, k_norm_gamma, norm_eps)
        kv_values = apply_partial_rope(
            pooled,
            rope_cos.index_select(0, group_ends),
            rope_sin.index_select(0, group_ends),
        ).to(torch.bfloat16)
    return _commit_indexer_cache(
        state_pool,
        state_slots,
        state_values,
        kv_pool,
        kv_slots,
        kv_values,
        capture_undo=True,
    )


def _write_indexer_cache_vectorized(
    raw_keys: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_norm_gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_block_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_block_table: torch.Tensor,
    *,
    ratio: int,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    norm_eps: float,
    invalid_block_policy: InvalidBlockPolicy,
    capture_undo: bool,
    rope_is_token_aligned: bool,
    is_s_padded: bool = False,
    collect_stats: bool = True,
) -> dict:
    """Device-vectorized writer used by the production CUDA path.

    Metadata validation has a constant number of host synchronizations. All
    token-to-page resolution, state writes, completed-group pooling and KV
    writes stay on the device; there is no per-token host scalar read or copy.
    """
    device = raw_keys.device
    token_count = int(raw_keys.shape[0])
    batch_size = int(cu_seqlens.numel()) - 1
    head_dim = int(raw_keys.shape[1])
    state_ring_entries = int(state_pool.shape[1])
    kv_entries_per_block = kv_tokens_per_block // ratio

    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must use int32 or int64")
    if start_positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("start_positions must use int32 or int64")
    device_tensors = (
        cu_seqlens,
        start_positions,
        rope_cos,
        rope_sin,
        k_norm_gamma,
        kv_pool,
        kv_block_table,
        state_pool,
        state_block_table,
    )
    if any(tensor.device != device for tensor in device_tensors):
        raise ValueError("vectorized indexer writer requires device-local tensors")
    valid_rope_dims = (2,) if rope_is_token_aligned else (2, 3)
    if rope_cos.shape != rope_sin.shape or rope_cos.dim() not in valid_rope_dims:
        raise ValueError(
            "rope_cos/rope_sin must have equal token-aligned [N, rotary_dim], "
            "shared [max_pos, rotary_dim], or "
            f"[B, max_pos, rotary_dim] shapes, got {rope_cos.shape} and "
            f"{rope_sin.shape}"
        )
    if rope_is_token_aligned and int(rope_cos.shape[0]) != token_count:
        raise ValueError(
            f"token-aligned RoPE must have N={token_count} rows, got "
            f"{rope_cos.shape[0]}"
        )
    if (
        not rope_is_token_aligned
        and rope_cos.dim() == 3
        and int(rope_cos.shape[0]) != batch_size
    ):
        raise ValueError(
            f"request-specific RoPE table must have B={batch_size}, got "
            f"{rope_cos.shape[0]}"
        )
    rotary_dim = int(rope_cos.shape[-1])
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= {head_dim}; got "
            f"{rotary_dim}"
        )

    cu = cu_seqlens.to(torch.long)
    starts = start_positions.to(torch.long)
    lengths = cu[1:] - cu[:-1]
    invalid_cu = (
        (cu[0] != 0)
        | (cu[-1] != token_count)
        | torch.any(lengths < 0)
        | torch.any(starts < 0)
    )
    # Only the first completed group of an unaligned request can need
    # committed keys. Carry this flag in the existing CU validation readback.
    start_in_group = starts.remainder(ratio)
    needs_history = ((start_in_group != 0) & (lengths >= ratio - start_in_group)).any()
    verdict = int((invalid_cu.to(torch.int32) + 2 * needs_history).item())
    if verdict & 1:
        raise ValueError(
            "cu_seqlens must be monotonic from 0 to raw_keys.shape[0], and "
            "start_positions must be non-negative"
        )
    if not rope_is_token_aligned:
        max_position = int((starts + lengths).max().item()) if batch_size else 0
        if max_position > int(rope_cos.shape[-2]):
            raise ValueError(
                f"RoPE tables cover {rope_cos.shape[-2]} positions, "
                f"need {max_position}"
            )

    flat_indices = torch.arange(token_count, dtype=torch.long, device=device)
    request_ids = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long, device=device),
        lengths,
        output_size=token_count,
    )
    local_indices = flat_indices - cu.index_select(0, request_ids)
    positions = starts.index_select(0, request_ids) + local_indices
    active = (
        state_block_table.ne(0).any(dim=1) | kv_block_table.ne(0).any(dim=1)
        if is_s_padded
        else None
    )

    def _resolve(
        table: torch.Tensor,
        requests: torch.Tensor,
        logical_blocks: torch.Tensor,
        *,
        pool_blocks: int,
        tag: str,
        required: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if active is not None:
            required = required & active[requests]
        width = int(table.shape[1])
        in_bounds = (logical_blocks >= 0) & (logical_blocks < width)
        safe_columns = logical_blocks.clamp(0, max(width - 1, 0))
        blocks = table[requests, safe_columns].to(torch.long)
        out_of_pool = required & in_bounds & (blocks >= pool_blocks)
        if bool(torch.any(out_of_pool).item()):
            raise IndexError(f"{tag} physical block is outside pool size {pool_blocks}")
        allocated = in_bounds & (blocks > 0) & (blocks < pool_blocks)
        if active is not None:
            allocated = allocated & active[requests]
        missing = required & ~allocated
        if invalid_block_policy == "raise" and bool(torch.any(missing).item()):
            raise ValueError(f"{tag} resolves to an unallocated logical block")
        return blocks, allocated

    state_logical = positions // state_tokens_per_block
    state_blocks, state_allocated = _resolve(
        state_block_table,
        request_ids,
        state_logical,
        pool_blocks=int(state_pool.shape[0]),
        tag="indexer_state",
        required=torch.ones(token_count, dtype=torch.bool, device=device),
    )
    state_slots = torch.where(
        state_allocated,
        state_blocks * state_ring_entries + positions % state_ring_entries,
        -1,
    )

    completed = (positions + 1).remainder(ratio) == 0
    if active is not None:
        completed = completed & active[request_ids]
    completed_indices = torch.nonzero(completed, as_tuple=False).flatten()
    kv_slots = torch.full_like(flat_indices, -1)
    kv_write_slots = torch.empty(0, dtype=torch.long, device=device)
    kv_write_values = raw_keys.new_empty((0, head_dim), dtype=torch.bfloat16)

    if int(completed_indices.numel()):
        completed_requests = request_ids.index_select(0, completed_indices)
        completed_positions = positions.index_select(0, completed_indices)
        block_starts = completed_positions - ratio + 1
        source_offsets = torch.arange(1 - ratio, 1, dtype=torch.long, device=device)
        raw_source_indices = completed_indices[:, None] + source_offsets
        if verdict & 2:
            raw_source_indices = raw_source_indices.clamp_min(0)
        gathered = (
            raw_keys.index_select(0, raw_source_indices.flatten())
            .view(-1, ratio, head_dim)
            .float()
        )
        if verdict & 2:
            source_positions = completed_positions[:, None] + source_offsets
            row_starts = starts.index_select(0, completed_requests)
            source_in_current = source_positions >= row_starts[:, None]
            source_blocks, source_allocated = _resolve(
                state_block_table,
                completed_requests[:, None].expand(-1, ratio),
                source_positions // state_tokens_per_block,
                pool_blocks=int(state_pool.shape[0]),
                tag="indexer_state source",
                required=~source_in_current,
            )
            group_sources_valid = (source_in_current | source_allocated).all(dim=1)
            state_source_slots = source_blocks.clamp(
                0, int(state_pool.shape[0]) - 1
            ) * state_ring_entries + source_positions.remainder(state_ring_entries)
            gathered_state = (
                state_pool.flatten(0, 1)
                .index_select(0, state_source_slots.flatten())
                .view(-1, ratio, head_dim)
            )
            gathered = torch.where(
                source_in_current.unsqueeze(-1), gathered, gathered_state
            )
        else:
            group_sources_valid = torch.ones_like(completed_indices, dtype=torch.bool)

        kv_logical = block_starts // kv_tokens_per_block
        kv_blocks, kv_allocated = _resolve(
            kv_block_table,
            completed_requests,
            kv_logical,
            pool_blocks=int(kv_pool.shape[0]),
            tag="indexer_kv",
            required=group_sources_valid,
        )
        kv_write_valid = group_sources_valid & kv_allocated
        compressed_indices = block_starts // ratio
        completed_kv_slots = (
            kv_blocks * kv_entries_per_block
            + compressed_indices.remainder(kv_entries_per_block)
        )
        kv_slots[completed_indices[kv_write_valid]] = completed_kv_slots[kv_write_valid]

        pooled = gathered.mean(dim=1).to(raw_keys.dtype)
        pooled = exact_head_rms_norm(pooled, k_norm_gamma, norm_eps)
        if rope_is_token_aligned:
            cos = rope_cos.index_select(0, completed_indices)
            sin = rope_sin.index_select(0, completed_indices)
        elif rope_cos.dim() == 2:
            cos = rope_cos.index_select(0, block_starts)
            sin = rope_sin.index_select(0, block_starts)
        else:
            cos = rope_cos[completed_requests, block_starts]
            sin = rope_sin[completed_requests, block_starts]
        pooled = apply_partial_rope(pooled, cos, sin).to(torch.bfloat16)
        kv_write_slots = completed_kv_slots[kv_write_valid]
        kv_write_values = pooled[kv_write_valid]

    state_write_mask = state_slots >= 0
    state_write_slots, state_write_values = _last_write_per_slot(
        state_slots[state_write_mask], raw_keys[state_write_mask].float()
    )
    kv_write_slots, kv_write_values = _last_write_per_slot(
        kv_write_slots, kv_write_values
    )

    undo = _commit_indexer_cache(
        state_pool,
        state_write_slots,
        state_write_values,
        kv_pool,
        kv_write_slots,
        kv_write_values,
        capture_undo=capture_undo,
    )

    result = {
        "state_slots": state_slots,
        "kv_slots": kv_slots,
        "completed": completed,
    }
    if collect_stats:
        result["num_state_writes"] = int(state_write_mask.sum().item())
        result["num_kv_writes"] = int((kv_slots >= 0).sum().item())
    if capture_undo:
        result["undo"] = undo
    return result


def write_indexer_cache(
    raw_keys: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_norm_gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_block_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_block_table: torch.Tensor,
    *,
    ratio: int,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    norm_eps: float = 1e-6,
    invalid_block_policy: InvalidBlockPolicy = "raise",
    capture_undo: bool = False,
    collect_stats: bool = True,
    rope_is_token_aligned: bool = False,
    is_s_padded: bool = False,
) -> dict:
    """Write ragged raw indexer keys to the state ring and completed means to KV.

    ``raw_keys`` is the packed projection output for this invocation. Request
    ``b`` occupies ``raw_keys[cu_seqlens[b]:cu_seqlens[b+1]]`` and starts at
    absolute token position ``start_positions[b]``.  This explicit metadata is
    sufficient for both ragged prefill and one-token decode, including a block
    completed using raw keys retained by an earlier invocation.

    Pool views are the typed production layouts:

    * ``kv_pool[num_blocks, kv_tokens_per_block / ratio, head_dim]`` bf16
    * ``state_pool[num_blocks, ring_entries, head_dim]`` fp32

    Each pool uses its *own* ``[B, max_blocks]`` block table. Block id zero is
    reserved by the framework. ``invalid_block_policy='raise'`` fails before
    mutating either pool; ``'skip'`` records ``-1`` for that write. Physical ids
    beyond the pool extent always raise because skipping them could hide memory
    corruption. Both tables use absolute logical-block columns. Only the payload
    offset inside an ``indexer_state`` block is a ring.

    With ``is_s_padded=True``, requests whose two side block-table rows are
    entirely zero are engine fake/warmup rows and never touch either pool.
    Every other row retains the chosen invalid-block policy.

    By default, ``rope_cos`` / ``rope_sin`` cover absolute positions and may be
    shared ``[max_position, rotary_dim]`` or request-specific
    ``[B, max_position, rotary_dim]``. With ``rope_is_token_aligned=True``, they
    instead contain one ``[rotary_dim]`` block-start row for each packed input
    token; only rows that complete a compression group are consumed.

    Set ``collect_stats=False`` when the host write counts are unused.
    """
    if invalid_block_policy not in ("raise", "skip"):
        raise ValueError(
            "invalid_block_policy must be 'raise' or 'skip', got "
            f"{invalid_block_policy!r}"
        )
    if raw_keys.dim() != 2:
        raise ValueError(f"raw_keys must be [N, head_dim], got {raw_keys.shape}")
    if raw_keys.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"raw_keys must be floating point, got {raw_keys.dtype}")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if kv_tokens_per_block <= 0 or kv_tokens_per_block % ratio:
        raise ValueError(
            "kv_tokens_per_block must be positive and divisible by ratio, got "
            f"{kv_tokens_per_block} and {ratio}"
        )
    if state_tokens_per_block <= 0:
        raise ValueError(
            f"state_tokens_per_block must be positive, got {state_tokens_per_block}"
        )

    head_dim = int(raw_keys.shape[1])
    if k_norm_gamma.shape != (head_dim,):
        raise ValueError(f"k_norm_gamma must be [{head_dim}], got {k_norm_gamma.shape}")
    kv_entries_per_block = kv_tokens_per_block // ratio
    if (
        kv_pool.dim() != 3
        or kv_pool.dtype != torch.bfloat16
        or tuple(kv_pool.shape[1:]) != (kv_entries_per_block, head_dim)
    ):
        raise ValueError(
            "kv_pool must be bf16 [num_blocks, "
            f"{kv_entries_per_block}, {head_dim}], got {kv_pool.shape} "
            f"{kv_pool.dtype}"
        )
    if (
        state_pool.dim() != 3
        or state_pool.dtype != torch.float32
        or int(state_pool.shape[2]) != head_dim
    ):
        raise ValueError(
            "state_pool must be fp32 [num_blocks, ring_entries, "
            f"{head_dim}], got {state_pool.shape} {state_pool.dtype}"
        )
    state_ring_entries = int(state_pool.shape[1])
    if state_ring_entries < ratio:
        raise ValueError(
            f"state ring needs at least ratio={ratio} entries, got "
            f"{state_ring_entries}"
        )

    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a [B + 1] vector")
    batch_size = int(cu_seqlens.numel()) - 1
    if start_positions.shape != (batch_size,):
        raise ValueError(
            f"start_positions must be [{batch_size}], got {start_positions.shape}"
        )
    for name, table in (
        ("kv_block_table", kv_block_table),
        ("state_block_table", state_block_table),
    ):
        if table.dim() != 2 or int(table.shape[0]) != batch_size:
            raise ValueError(f"{name} must be [B, max_blocks], got {table.shape}")
        if table.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {table.dtype}")
        if int(table.shape[1]) == 0:
            raise ValueError(f"{name} must contain at least one block column")

    if raw_keys.is_cuda and all(
        tensor.device == raw_keys.device
        for tensor in (
            cu_seqlens,
            start_positions,
            rope_cos,
            rope_sin,
            k_norm_gamma,
            kv_pool,
            kv_block_table,
            state_pool,
            state_block_table,
        )
    ):
        return _write_indexer_cache_vectorized(
            raw_keys,
            cu_seqlens,
            start_positions,
            rope_cos,
            rope_sin,
            k_norm_gamma,
            kv_pool,
            kv_block_table,
            state_pool,
            state_block_table,
            ratio=ratio,
            kv_tokens_per_block=kv_tokens_per_block,
            state_tokens_per_block=state_tokens_per_block,
            norm_eps=norm_eps,
            invalid_block_policy=invalid_block_policy,
            capture_undo=capture_undo,
            collect_stats=collect_stats,
            rope_is_token_aligned=rope_is_token_aligned,
            is_s_padded=is_s_padded,
        )

    offsets = [int(v) for v in cu_seqlens.tolist()]
    if offsets[0] != 0 or offsets[-1] != int(raw_keys.shape[0]):
        raise ValueError(
            f"cu_seqlens must start at 0 and end at N={raw_keys.shape[0]}, got "
            f"{offsets[0]} and {offsets[-1]}"
        )
    if any(lo > hi for lo, hi in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")
    starts = [int(v) for v in start_positions.tolist()]
    if any(pos < 0 for pos in starts):
        raise ValueError(f"start_positions must be non-negative, got {starts}")

    valid_rope_dims = (2,) if rope_is_token_aligned else (2, 3)
    if rope_cos.shape != rope_sin.shape or rope_cos.dim() not in valid_rope_dims:
        raise ValueError(
            "rope_cos/rope_sin must have equal token-aligned [N, rotary_dim], "
            "shared [max_pos, rotary_dim], or "
            f"[B, max_pos, rotary_dim] shapes, got {rope_cos.shape} and "
            f"{rope_sin.shape}"
        )
    if rope_is_token_aligned and int(rope_cos.shape[0]) != int(raw_keys.shape[0]):
        raise ValueError(
            f"token-aligned RoPE must have N={raw_keys.shape[0]} rows, got "
            f"{rope_cos.shape[0]}"
        )
    if (
        not rope_is_token_aligned
        and rope_cos.dim() == 3
        and int(rope_cos.shape[0]) != batch_size
    ):
        raise ValueError(
            f"request-specific RoPE table must have B={batch_size}, got "
            f"{rope_cos.shape[0]}"
        )
    rotary_dim = int(rope_cos.shape[-1])
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= {head_dim}; got "
            f"{rotary_dim}"
        )
    if not rope_is_token_aligned:
        rope_positions = int(rope_cos.shape[-2])
        max_position = max(
            (starts[b] + offsets[b + 1] - offsets[b] for b in range(batch_size)),
            default=0,
        )
        if max_position > rope_positions:
            raise ValueError(
                f"RoPE tables cover {rope_positions} positions, need {max_position}"
            )

    # Stage every value and destination first. Strict-mode validation therefore
    # cannot leave one pool half-written if a later request has bad metadata.
    state_writes = []
    kv_writes = []
    state_slots = torch.full(
        (raw_keys.shape[0],), -1, dtype=torch.long, device=raw_keys.device
    )
    kv_slots = torch.full_like(state_slots, -1)
    completed = torch.zeros_like(state_slots, dtype=torch.bool)

    for request_idx in range(batch_size):
        if (
            is_s_padded
            and not bool(state_block_table[request_idx].ne(0).any())
            and not bool(kv_block_table[request_idx].ne(0).any())
        ):
            continue
        flat_lo, flat_hi = offsets[request_idx], offsets[request_idx + 1]
        seq_start = starts[request_idx]
        seq_len = flat_hi - flat_lo

        for local_idx in range(seq_len):
            flat_idx = flat_lo + local_idx
            position = seq_start + local_idx
            state_logical_block = position // state_tokens_per_block
            state_block = _physical_block(
                state_block_table,
                request_idx,
                state_logical_block,
                num_physical_blocks=int(state_pool.shape[0]),
                tag="indexer_state",
                invalid_block_policy=invalid_block_policy,
            )
            if state_block is not None:
                state_offset = position % state_ring_entries
                state_slots[flat_idx] = state_block * state_ring_entries + state_offset
                state_writes.append(
                    (state_block, state_offset, raw_keys[flat_idx].float())
                )

            if (position + 1) % ratio:
                continue
            completed[flat_idx] = True
            block_start = position - ratio + 1
            gathered = []
            source_missing = False
            for source_position in range(block_start, position + 1):
                source_local = source_position - seq_start
                if 0 <= source_local < seq_len:
                    gathered.append(raw_keys[flat_lo + source_local].float())
                    continue
                source_logical_block = source_position // state_tokens_per_block
                source_block = _physical_block(
                    state_block_table,
                    request_idx,
                    source_logical_block,
                    num_physical_blocks=int(state_pool.shape[0]),
                    tag="indexer_state source",
                    invalid_block_policy=invalid_block_policy,
                )
                if source_block is None:
                    source_missing = True
                    break
                gathered.append(
                    state_pool[source_block, source_position % state_ring_entries]
                )
            if source_missing:
                continue

            kv_logical_block = block_start // kv_tokens_per_block
            kv_block = _physical_block(
                kv_block_table,
                request_idx,
                kv_logical_block,
                num_physical_blocks=int(kv_pool.shape[0]),
                tag="indexer_kv",
                invalid_block_policy=invalid_block_policy,
            )
            if kv_block is None:
                continue
            compressed_idx = block_start // ratio
            kv_offset = compressed_idx % kv_entries_per_block
            pooled = torch.stack(gathered).float().mean(dim=0).to(raw_keys.dtype)
            pooled = exact_head_rms_norm(pooled, k_norm_gamma, norm_eps)
            if rope_is_token_aligned:
                cos = rope_cos[flat_idx]
                sin = rope_sin[flat_idx]
            elif rope_cos.dim() == 2:
                cos = rope_cos[block_start]
                sin = rope_sin[block_start]
            else:
                cos = rope_cos[request_idx, block_start]
                sin = rope_sin[request_idx, block_start]
            pooled = apply_partial_rope(pooled, cos, sin).to(torch.bfloat16)
            kv_slots[flat_idx] = kv_block * kv_entries_per_block + kv_offset
            kv_writes.append((kv_block, kv_offset, pooled))

    undo = None
    if capture_undo:
        # Capture only after every value and destination has been validated and
        # staged, immediately before the first persistent copy.
        state_undo_slots, original_state = _capture_pool_slots(state_pool, state_writes)
        kv_undo_slots, original_kv = _capture_pool_slots(kv_pool, kv_writes)
        undo = IndexerCacheUndo(
            state_pool=state_pool,
            state_slots=state_undo_slots,
            original_state=original_state,
            kv_pool=kv_pool,
            kv_slots=kv_undo_slots,
            original_kv=original_kv,
        )

    try:
        for block_id, offset, value in state_writes:
            state_pool[block_id, offset].copy_(value)
        for block_id, offset, value in kv_writes:
            kv_pool[block_id, offset].copy_(value)
    except BaseException:
        if undo is not None:
            restore_indexer_cache(undo)
            if state_pool.is_cuda:
                torch.cuda.current_stream(state_pool.device).synchronize()
        raise

    result = {
        "state_slots": state_slots,
        "kv_slots": kv_slots,
        "completed": completed,
    }
    if collect_stats:
        result["num_state_writes"] = len(state_writes)
        result["num_kv_writes"] = len(kv_writes)
    if capture_undo:
        result["undo"] = undo
    return result

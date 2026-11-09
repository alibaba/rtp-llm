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

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
    qsa_assert_valid,
    qsa_strict_validation_enabled,
)
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
    original = pool.flatten(0, 1).index_select(0, slots).clone()
    return slots, original


def restore_indexer_cache(undo: IndexerCacheUndo) -> None:
    """Restore a captured side-cache write on the caller's current stream."""
    if undo.padded_slots:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill_triton import (
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


def _capture_tensor_slots(
    pool: torch.Tensor, slots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    unique_slots = torch.unique(slots)
    original = pool.flatten(0, 1).index_select(0, unique_slots).clone()
    return unique_slots, original


def _resolve_blocks(
    table: torch.Tensor,
    requests: torch.Tensor,
    logical_blocks: torch.Tensor,
    *,
    pool_blocks: int,
    tag: str,
    required: torch.Tensor,
    invalid_block_policy: InvalidBlockPolicy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve token/page block-table lookups without flattening batches.

    ``requests`` / ``logical_blocks`` may be [N] or [N, M] grade; the returned
    ``blocks`` keep that shape.  ``required`` marks entries whose logical page
    must resolve to an allocated physical block (async device assert on CUDA,
    exact legacy host error in strict mode).
    """
    width = int(table.shape[1])
    in_bounds = (logical_blocks >= 0) & (logical_blocks < width)
    safe_columns = logical_blocks.clamp(0, max(width - 1, 0))
    blocks = table[requests, safe_columns].to(torch.long)
    out_of_pool = required & in_bounds & (blocks >= pool_blocks)
    if qsa_strict_validation_enabled():
        if bool(torch.any(out_of_pool).item()):
            raise IndexError(f"{tag} physical block is outside pool size {pool_blocks}")
    else:
        qsa_assert_valid(
            ~torch.any(out_of_pool),
            f"{tag} physical block is outside pool size {pool_blocks}",
        )
    # The allocated mask stays materialized: the write plan below needs it
    # to mask out unresolved slots.
    allocated = in_bounds & (blocks > 0) & (blocks < pool_blocks)
    missing = required & ~allocated
    if invalid_block_policy == "raise":
        if qsa_strict_validation_enabled():
            if bool(torch.any(missing).item()):
                raise ValueError(f"{tag} resolves to an unallocated logical block")
        else:
            qsa_assert_valid(
                ~torch.any(missing),
                f"{tag} resolves to an unallocated logical block",
            )
    return blocks, allocated


@dataclass(frozen=True)
class IndexerWriterGeometry:
    """Layer-independent geometry for one step's vectorized indexer write.

    Everything here depends only on the step's ``cu_seqlens`` /
    ``start_positions`` / block tables / pool shapes -- never on the layer's
    ``raw_keys`` or pool *contents*.  A step plan freezes it once and every
    layer's writer consumes the same tensors, replacing the per-call eager
    chain (``arange``/``repeat_interleave``/resolve/assert launches) that the
    draft tail window paid per call.

    ``kv_slots`` is the writer's result row (``-1`` where nothing is written)
    and ``kv_write_slots``/``state_slots_safe`` are the clamped destinations
    the full-shape masked-scatter write path uses (invalid rows write back
    their gathered originals, so the pool is untouched).  ``state_slots_unique``
    is a host fact: no request spans more than ``state_ring_entries`` new
    tokens, so two tokens of one call can never map to the same state slot
    (positions of one request are consecutive; two positions collide only when
    they differ by a multiple of the ring size).
    """

    # Config the bundle was validated against (host checks on reuse).
    ratio: int
    kv_tokens_per_block: int
    state_tokens_per_block: int
    state_ring_entries: int
    kv_entries_per_block: int
    token_count: int
    batch_size: int
    invalid_block_policy: InvalidBlockPolicy
    device: torch.device
    kv_pool_blocks: int
    state_pool_blocks: int
    # Identity of the tensors this geometry was derived from; a rebound or
    # in-place edited source falls the writer back to the inline build.
    cu_seqlens: torch.Tensor
    start_positions: torch.Tensor
    kv_block_table: torch.Tensor
    state_block_table: torch.Tensor
    source_fingerprints: dict[str, tuple[int, int] | None]
    # Host facts.
    lengths: tuple[int, ...]
    state_slots_unique: bool
    # Device geometry tensors ([B] / [T] / [T, ratio] grade).
    batch_lengths: torch.Tensor
    request_ids: torch.Tensor
    positions: torch.Tensor
    row_starts: torch.Tensor
    row_lengths: torch.Tensor
    state_slots: torch.Tensor
    state_allocated: torch.Tensor
    state_slots_safe: torch.Tensor
    completed: torch.Tensor
    block_starts: torch.Tensor
    kv_blocks: torch.Tensor
    kv_allocated: torch.Tensor
    kv_entry: torch.Tensor
    kv_write_slots: torch.Tensor
    kv_slots: torch.Tensor
    source_rows: torch.Tensor
    source_slots: torch.Tensor
    source_in_current: torch.Tensor
    source_valid: torch.Tensor
    group_sources_valid: torch.Tensor
    kv_write_valid: torch.Tensor


def _writer_geometry_fingerprint(tensor: torch.Tensor) -> tuple[int, int] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    if torch.is_inference(tensor):
        return (int(tensor.data_ptr()), -1)
    return (int(tensor.data_ptr()), int(tensor._version))


def writer_geometry_applies(
    geometry: IndexerWriterGeometry | None,
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    kv_block_table: torch.Tensor,
    state_block_table: torch.Tensor,
    kv_pool: torch.Tensor,
    state_pool: torch.Tensor,
    *,
    token_count: int,
    ratio: int,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    invalid_block_policy: InvalidBlockPolicy,
) -> bool:
    """Whether the frozen geometry still describes this writer invocation.

    Host-only checks (identity, fingerprints, shapes, config ints): a mismatch
    makes the caller fall back to the legacy inline build instead of trusting a
    stale bundle.  Block tables are compared by tensor identity *and*
    fingerprint so a rebound or in-place edited table between layers can never
    feed layer 1's resolved blocks to layer 2.
    """
    if geometry is None:
        return False
    if (
        geometry.ratio != ratio
        or geometry.kv_tokens_per_block != kv_tokens_per_block
        or geometry.state_tokens_per_block != state_tokens_per_block
        or geometry.state_ring_entries != int(state_pool.shape[1])
        or geometry.kv_entries_per_block != kv_tokens_per_block // ratio
        or geometry.token_count != token_count
        or geometry.batch_size != int(cu_seqlens.numel()) - 1
    ):
        return False
    if geometry.kv_pool_blocks != int(kv_pool.shape[0]):
        return False
    if geometry.state_pool_blocks != int(state_pool.shape[0]):
        return False
    if geometry.invalid_block_policy != invalid_block_policy:
        return False
    if geometry.device != cu_seqlens.device:
        return False
    sources = (
        ("cu_seqlens", cu_seqlens),
        ("start_positions", start_positions),
        ("kv_block_table", kv_block_table),
        ("state_block_table", state_block_table),
    )
    for name, tensor in sources:
        if getattr(geometry, name) is not tensor:
            return False
        if geometry.source_fingerprints.get(name) != _writer_geometry_fingerprint(
            tensor
        ):
            return False
    return True


def build_indexer_writer_geometry(
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    kv_block_table: torch.Tensor,
    state_block_table: torch.Tensor,
    *,
    token_count: int,
    ratio: int,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    state_ring_entries: int,
    kv_pool_blocks: int,
    state_pool_blocks: int,
    invalid_block_policy: InvalidBlockPolicy,
    host_lengths: Optional[Sequence[int]] = None,
    host_prefixes: Optional[Sequence[int]] = None,
    device_positions: Optional[torch.Tensor] = None,
    device_block_starts: Optional[torch.Tensor] = None,
    trusted_tables: bool = False,
) -> IndexerWriterGeometry:
    """Freeze one step's writer geometry (token/page resolution included).

    Mirrors the legacy per-call leading chain of
    ``_write_indexer_cache_vectorized`` -- same resolution semantics and (by
    default) the same assert messages -- but on full ``[T]`` / ``[T, ratio]``
    shapes with no compression, so nothing in here synchronizes with the host.

    ``host_lengths`` / ``host_prefixes`` (the step plan's pinned mirrors) turn
    every ``[T]``/``[T, ratio]`` integer index tensor into pure host arithmetic
    riding two pinned uploads, replacing the device
    ``arange``/``repeat_interleave``/``remainder``/compare chains.  Without
    them the device chain of the legacy path computes the same values.

    ``trusted_tables=True`` (the incremental step plan, whose build already
    proved ``cu_seqlens`` equals the expected partition and validated every
    required table column through the required-blocks verdicts) skips the
    redundant cu/resolve asserts: every lookup below stays inside the
    verdict-checked column window, so the coverage is identical.
    """
    device = cu_seqlens.device
    batch_size = int(cu_seqlens.numel()) - 1
    kv_entries_per_block = kv_tokens_per_block // ratio
    state_width = int(state_block_table.shape[1])
    kv_width = int(kv_block_table.shape[1])

    host_lengths_known = (
        host_lengths is not None
        and len(host_lengths) == batch_size
        and sum(host_lengths) == token_count
    )
    host_prefixes_known = (
        host_prefixes is not None and len(host_prefixes) == batch_size
    )

    # Guarded mirrors: ``lengths_values`` is read below even when only one of
    # the two host facts is available, and the device chain then fills every
    # index tensor without host arithmetic.
    lengths_values: list[int] = []
    prefixes_values: list[int] = []
    host_arithmetic = host_lengths_known and host_prefixes_known
    if host_arithmetic:
        lengths_values = [int(value) for value in host_lengths]
        prefixes_values = [int(value) for value in host_prefixes]
        if not trusted_tables:
            _assert_host_mirrors_match_sources(
                cu_seqlens,
                start_positions,
                lengths_values,
                prefixes_values,
                token_count,
            )
        # One host pass builds every integer index tensor; two pinned uploads
        # replace the device staging chains.
        req: list[int] = []
        pos: list[int] = []
        row_start: list[int] = []
        row_len: list[int] = []
        req_off: list[int] = []
        done: list[bool] = []
        bstart: list[int] = []
        state_col: list[int] = []
        state_good: list[bool] = []
        state_off: list[int] = []
        kv_col: list[int] = []
        kv_good: list[bool] = []
        kv_ent: list[int] = []
        src_req: list[int] = []
        src_col: list[int] = []
        src_good: list[bool] = []
        src_cur: list[bool] = []
        src_off: list[int] = []
        src_row: list[int] = []
        cu_running = 0
        top_row = max(token_count - 1, 0)
        for request_idx, (prefix, length) in enumerate(
            zip(prefixes_values, lengths_values)
        ):
            for row in range(length):
                position = prefix + row
                group_start = position - position % ratio
                req.append(request_idx)
                pos.append(position)
                row_start.append(prefix)
                row_len.append(length)
                req_off.append(cu_running)
                done.append((position + 1) % ratio == 0)
                bstart.append(group_start)
                column = position // state_tokens_per_block
                state_col.append(min(max(column, 0), max(state_width - 1, 0)))
                state_good.append(0 <= column < state_width)
                state_off.append(position % state_ring_entries)
                column = group_start // kv_tokens_per_block
                kv_col.append(min(max(column, 0), max(kv_width - 1, 0)))
                kv_good.append(0 <= column < kv_width)
                kv_ent.append((group_start // ratio) % kv_entries_per_block)
                for offset in range(ratio):
                    source_position = group_start + offset
                    source_local = source_position - prefix
                    in_current = 0 <= source_local < length
                    column = source_position // state_tokens_per_block
                    src_req.append(request_idx)
                    src_col.append(
                        min(max(column, 0), max(state_width - 1, 0))
                    )
                    src_good.append(0 <= column < state_width)
                    src_cur.append(in_current)
                    src_off.append(source_position % state_ring_entries)
                    src_row.append(
                        min(max(cu_running + source_local, 0), top_row)
                    )
            cu_running += length

        host_int = torch.tensor(
            req + pos + row_start + row_len + req_off + bstart
            + state_col + state_off + kv_col + kv_ent
            + src_req + src_col + src_off + src_row
            + lengths_values,
            dtype=torch.long,
        )
        host_bool = torch.tensor(
            done + state_good + kv_good + src_good + src_cur, dtype=torch.bool
        )
        if device.type == "cuda":
            device_int = host_int.pin_memory().to(device=device, non_blocking=True)
            device_bool = host_bool.pin_memory().to(
                device=device, non_blocking=True
            )
        else:
            device_int = host_int.to(device)
            device_bool = host_bool.to(device)

        def _int_view(count):
            nonlocal device_int
            view = device_int[:count]
            device_int = device_int[count:]
            return view

        def _bool_view(count):
            nonlocal device_bool
            view = device_bool[:count]
            device_bool = device_bool[count:]
            return view

        request_ids = _int_view(token_count)
        positions = _int_view(token_count)
        row_starts = _int_view(token_count)
        row_lengths = _int_view(token_count)
        request_offsets = _int_view(token_count)
        block_starts = _int_view(token_count)
        state_columns = _int_view(token_count)
        state_offsets = _int_view(token_count)
        kv_columns = _int_view(token_count)
        kv_entry = _int_view(token_count)
        completed = _bool_view(token_count)
        state_in_bounds = _bool_view(token_count)
        kv_in_bounds = _bool_view(token_count)
        source_count = token_count * ratio
        source_requests = _int_view(source_count).view(token_count, ratio)
        source_columns = _int_view(source_count).view(token_count, ratio)
        source_offsets = _int_view(source_count).view(token_count, ratio)
        source_rows = _int_view(source_count).view(token_count, ratio)
        source_in_bounds = _bool_view(source_count).view(token_count, ratio)
        source_in_current = _bool_view(source_count).view(token_count, ratio)
        # The batch length mirror rides the same pinned upload (a separate
        # pageable host->device staging would violate the step-build
        # contract that every host-built upload is pinned).
        batch_lengths = _int_view(batch_size)
    else:
        # Device chain (the legacy staging): identical values, no host facts.
        cu = cu_seqlens.to(torch.long)
        starts = start_positions.to(torch.long)
        lengths = cu[1:] - cu[:-1]
        if not trusted_tables:
            invalid_cu = (
                (cu[0] != 0)
                | (cu[-1] != token_count)
                | torch.any(lengths < 0)
                | torch.any(starts < 0)
            )
            if qsa_strict_validation_enabled():
                if bool(invalid_cu.item()):
                    raise ValueError(
                        "cu_seqlens must be monotonic from 0 to "
                        "raw_keys.shape[0], and start_positions must be "
                        "non-negative"
                    )
            else:
                qsa_assert_valid(
                    ~invalid_cu,
                    "cu_seqlens must be monotonic from 0 to raw_keys.shape[0],"
                    " and start_positions must be non-negative",
                )
        flat_indices = torch.arange(token_count, dtype=torch.long, device=device)
        request_ids = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long, device=device),
            lengths,
            output_size=token_count,
        )
        if device_positions is not None and int(device_positions.shape[0]) == token_count:
            positions = device_positions
        else:
            local_indices = flat_indices - cu.index_select(0, request_ids)
            positions = starts.index_select(0, request_ids) + local_indices
        row_starts = starts.index_select(0, request_ids)
        row_lengths = lengths.index_select(0, request_ids)
        request_offsets = cu.index_select(0, request_ids)
        completed = (positions + 1).remainder(ratio) == 0
        if device_block_starts is not None and int(
            device_block_starts.shape[0]
        ) == token_count:
            block_starts = device_block_starts
        else:
            block_starts = positions - positions.remainder(ratio)
        state_logical = positions // state_tokens_per_block
        state_in_bounds = (state_logical >= 0) & (state_logical < state_width)
        state_columns = state_logical.clamp(0, max(state_width - 1, 0))
        state_offsets = positions % state_ring_entries
        kv_logical = block_starts // kv_tokens_per_block
        kv_in_bounds = (kv_logical >= 0) & (kv_logical < kv_width)
        kv_columns = kv_logical.clamp(0, max(kv_width - 1, 0))
        kv_entry = (block_starts // ratio).remainder(kv_entries_per_block)
        source_offsets = torch.arange(ratio, dtype=torch.long, device=device)
        source_positions = block_starts[:, None] + source_offsets[None, :]
        source_local = source_positions - row_starts[:, None]
        source_in_current = (source_local >= 0) & (
            source_local < row_lengths[:, None]
        )
        source_requests = request_ids[:, None].expand(-1, ratio)
        source_logical = source_positions // state_tokens_per_block
        source_in_bounds = (source_logical >= 0) & (source_logical < state_width)
        source_columns = source_logical.clamp(0, max(state_width - 1, 0))
        source_offsets = source_positions.remainder(state_ring_entries)
        source_rows = (request_offsets[:, None] + source_local).clamp(
            0, max(token_count - 1, 0)
        )

    # Table lookups: three gathers, nothing else touches the tables.
    state_blocks = state_block_table[request_ids, state_columns].to(torch.long)
    source_blocks = state_block_table[source_requests, source_columns].to(
        torch.long
    )
    kv_blocks = kv_block_table[request_ids, kv_columns].to(torch.long)

    if not trusted_tables:
        _assert_resolved_blocks(
            tag="indexer_state",
            required=torch.ones(token_count, dtype=torch.bool, device=device),
            in_bounds=state_in_bounds,
            blocks=state_blocks,
            pool_blocks=state_pool_blocks,
            invalid_block_policy=invalid_block_policy,
        )
        _assert_resolved_blocks(
            tag="indexer_state source",
            required=(~source_in_current) & completed[:, None],
            in_bounds=source_in_bounds,
            blocks=source_blocks,
            pool_blocks=state_pool_blocks,
            invalid_block_policy=invalid_block_policy,
        )
        group_pre_valid = torch.all(source_in_current | (
            source_in_bounds & (source_blocks > 0) & (source_blocks < state_pool_blocks)
        ), dim=1)
        _assert_resolved_blocks(
            tag="indexer_kv",
            required=group_pre_valid & completed,
            in_bounds=kv_in_bounds,
            blocks=kv_blocks,
            pool_blocks=kv_pool_blocks,
            invalid_block_policy=invalid_block_policy,
        )

    state_allocated = state_in_bounds & (state_blocks > 0) & (
        state_blocks < state_pool_blocks
    )
    source_allocated = source_in_bounds & (source_blocks > 0) & (
        source_blocks < state_pool_blocks
    )
    kv_allocated = kv_in_bounds & (kv_blocks > 0) & (kv_blocks < kv_pool_blocks)
    state_slots = torch.where(
        state_allocated, state_blocks * state_ring_entries + state_offsets, -1
    )
    state_slots_safe = state_slots.clamp(min=0)
    source_valid = source_in_current | source_allocated
    group_sources_valid = torch.all(source_valid, dim=1)
    source_slots = (
        source_blocks.clamp(0, state_pool_blocks - 1) * state_ring_entries
        + source_offsets
    )

    kv_write_valid = completed & group_sources_valid & kv_allocated
    # Slot zero is the framework-reserved block: valid destinations always sit
    # above it (``allocated`` requires block > 0), so invalid rows redirect
    # there and write back its own original in the full-shape scatter.
    kv_write_slots = torch.where(
        kv_write_valid,
        kv_blocks.clamp(0, kv_pool_blocks - 1) * kv_entries_per_block + kv_entry,
        0,
    )
    kv_slots = torch.where(kv_write_valid, kv_write_slots, -1)

    # Host facts only: with the step's length mirror the uniqueness verdict is
    # pure host arithmetic; without it the conservative False keeps the legacy
    # write path (no device readback here, ever).
    host_length_values = lengths_values if host_lengths_known else []
    state_slots_unique = bool(
        host_length_values and max(host_length_values) <= state_ring_entries
    )
    if state_slots_unique:
        # The full-shape masked-scatter write is exact only when no two rows
        # resolve to the same destination slot.  Within one request that holds
        # whenever the request spans at most ``state_ring_entries`` new tokens
        # (consecutive positions); across requests it is the allocator's
        # per-request block disjointness.  Enforce both once per step with an
        # async device assert instead of a host sync: a violation is a broken
        # engine/allocator invariant and traps fail-stop, matching
        # ``qsa_assert_valid``'s contract.
        sorted_state = torch.sort(state_slots_safe).values
        qsa_assert_valid(
            ~torch.any(
                (sorted_state[1:] == sorted_state[:-1]) & (sorted_state[1:] > 0)
            ),
            "indexer writer state destinations must be unique; the state block "
            "table shares a physical block across requests",
        )
        sorted_kv = torch.sort(kv_write_slots).values
        qsa_assert_valid(
            ~torch.any((sorted_kv[1:] == sorted_kv[:-1]) & (sorted_kv[1:] > 0)),
            "indexer writer KV destinations must be unique; the KV block table "
            "shares a physical block across requests",
        )

    if not host_arithmetic:
        batch_lengths = (cu[1:] - cu[:-1]).to(torch.long)
    return IndexerWriterGeometry(
        ratio=ratio,
        kv_tokens_per_block=kv_tokens_per_block,
        state_tokens_per_block=state_tokens_per_block,
        state_ring_entries=state_ring_entries,
        kv_entries_per_block=kv_entries_per_block,
        token_count=token_count,
        batch_size=batch_size,
        invalid_block_policy=invalid_block_policy,
        device=device,
        kv_pool_blocks=kv_pool_blocks,
        state_pool_blocks=state_pool_blocks,
        cu_seqlens=cu_seqlens,
        start_positions=start_positions,
        kv_block_table=kv_block_table,
        state_block_table=state_block_table,
        source_fingerprints={
            name: _writer_geometry_fingerprint(tensor)
            for name, tensor in (
                ("cu_seqlens", cu_seqlens),
                ("start_positions", start_positions),
                ("kv_block_table", kv_block_table),
                ("state_block_table", state_block_table),
            )
        },
        lengths=tuple(host_length_values),
        state_slots_unique=state_slots_unique,
        batch_lengths=batch_lengths,
        request_ids=request_ids,
        positions=positions,
        row_starts=row_starts,
        row_lengths=row_lengths,
        state_slots=state_slots,
        state_allocated=state_allocated,
        state_slots_safe=state_slots_safe,
        completed=completed,
        block_starts=block_starts,
        kv_blocks=kv_blocks,
        kv_allocated=kv_allocated,
        kv_entry=kv_entry,
        kv_write_slots=kv_write_slots,
        kv_slots=kv_slots,
        source_rows=source_rows,
        source_slots=source_slots,
        source_in_current=source_in_current,
        source_valid=source_valid,
        group_sources_valid=group_sources_valid,
        kv_write_valid=kv_write_valid,
    )


def _assert_host_mirrors_match_sources(
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    lengths: list[int],
    prefixes: list[int],
    token_count: int,
) -> None:
    """Host-derived geometry is only valid if it describes the real sources.

    The host path builds every index tensor from the mirrors, so the device
    ``cu_seqlens`` / ``start_positions`` must agree with them.  One async
    verdict per source (no host sync); strict mode keeps the legacy
    fail-fast host comparison with the legacy message.
    """
    cu_values = [0]
    for length in lengths:
        if length < 0:
            raise ValueError(
                "cu_seqlens must be monotonic from 0 to raw_keys.shape[0], and "
                "start_positions must be non-negative"
            )
        cu_values.append(cu_values[-1] + length)
    if cu_values[-1] != token_count or any(
        prefix < 0 for prefix in prefixes
    ):
        raise ValueError(
            "cu_seqlens must be monotonic from 0 to raw_keys.shape[0], and "
            "start_positions must be non-negative"
        )
    expected_cu = torch.tensor(
        cu_values, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    expected_starts = torch.tensor(
        prefixes, dtype=start_positions.dtype, device=start_positions.device
    )
    message = (
        "cu_seqlens must be monotonic from 0 to raw_keys.shape[0], and "
        "start_positions must be non-negative"
    )
    cu_verdict = torch.equal if qsa_strict_validation_enabled() else (
        lambda a, b: (a == b).all()
    )
    cu_ok = cu_verdict(cu_seqlens, expected_cu)
    starts_ok = cu_verdict(start_positions, expected_starts)
    if qsa_strict_validation_enabled():
        if not bool(cu_ok) or not bool(starts_ok):
            raise ValueError(message)
    else:
        qsa_assert_valid(cu_ok & starts_ok, message)


def _assert_resolved_blocks(
    *,
    tag: str,
    required: torch.Tensor,
    in_bounds: torch.Tensor,
    blocks: torch.Tensor,
    pool_blocks: int,
    invalid_block_policy: InvalidBlockPolicy,
) -> None:
    """The resolve asserts of the legacy chain over pre-gathered blocks."""
    out_of_pool = required & in_bounds & (blocks >= pool_blocks)
    if qsa_strict_validation_enabled():
        if bool(torch.any(out_of_pool).item()):
            raise IndexError(
                f"{tag} physical block is outside pool size {pool_blocks}"
            )
    else:
        qsa_assert_valid(
            ~torch.any(out_of_pool),
            f"{tag} physical block is outside pool size {pool_blocks}",
        )
    missing = required & ~(in_bounds & (blocks > 0) & (blocks < pool_blocks))
    if invalid_block_policy == "raise":
        if qsa_strict_validation_enabled():
            if bool(torch.any(missing).item()):
                raise ValueError(
                    f"{tag} resolves to an unallocated logical block"
                )
        else:
            qsa_assert_valid(
                ~torch.any(missing),
                f"{tag} resolves to an unallocated logical block",
            )


def _write_indexer_cache_full_shape(
    raw_keys: torch.Tensor,
    geometry: IndexerWriterGeometry,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_norm_gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    state_pool: torch.Tensor,
    *,
    norm_eps: float,
    capture_undo: bool,
    rope_is_token_aligned: bool,
) -> dict:
    """Compression-free writer for a step whose geometry is frozen.

    Every destination stays full-shape: no ``torch.nonzero`` compression, no
    masked-select gathers, no ``argsort``/``unique`` chains and therefore no
    device-to-host synchronization.  Writes are masked scatters where invalid
    rows write back their own gathered original, so untouched pool entries keep
    their bytes; the uniqueness of the written slots is a host fact of the
    frozen geometry (``state_slots_unique``) plus the geometry build's async
    cross-request-block-disjointness assert.

    Byte-for-byte this produces the same pool contents, ``state_slots`` /
    ``kv_slots`` rows and write counts as the legacy compressed path.
    """
    device = raw_keys.device
    token_count = geometry.token_count
    head_dim = int(raw_keys.shape[1])
    ratio = geometry.ratio
    kv_flat = kv_pool.flatten(0, 1)
    state_flat = state_pool.flatten(0, 1)

    if not rope_is_token_aligned:
        # Same coverage contract as the legacy path (rope tables are per-call
        # arguments, so the check cannot ride the frozen geometry): the [B]-grade
        # start/length pair sum, exactly as the legacy writer computes it.
        # Strict validation never reaches this path: the dispatch keeps the
        # legacy writer with its exact host diagnostics.
        qsa_assert_valid(
            torch.all(
                (geometry.start_positions.to(torch.long) + geometry.batch_lengths)
                <= int(rope_cos.shape[-2])
            ),
            f"RoPE tables cover {rope_cos.shape[-2]} positions, need more",
        )

    # Capture every destination's original before the first mutation; the
    # gathered rows double as the write-back source for invalid rows and as the
    # undo payload, so undo capture adds no launches.
    original_kv = kv_flat.index_select(0, geometry.kv_write_slots)
    original_state = state_flat.index_select(0, geometry.state_slots_safe)

    gathered_raw = raw_keys.index_select(0, geometry.source_rows.reshape(-1)).view(
        token_count, ratio, head_dim
    )
    gathered_state = state_flat.index_select(
        0, geometry.source_slots.reshape(-1)
    ).view(token_count, ratio, head_dim)
    gathered = torch.where(
        geometry.source_in_current.unsqueeze(-1),
        gathered_raw.float(),
        gathered_state,
    )
    pooled = gathered.mean(dim=1).to(raw_keys.dtype)
    pooled = exact_head_rms_norm(pooled, k_norm_gamma, norm_eps)
    if rope_is_token_aligned:
        cos = rope_cos
        sin = rope_sin
    elif rope_cos.dim() == 2:
        cos = rope_cos.index_select(0, geometry.block_starts)
        sin = rope_sin.index_select(0, geometry.block_starts)
    else:
        cos = rope_cos[geometry.request_ids, geometry.block_starts]
        sin = rope_sin[geometry.request_ids, geometry.block_starts]
    pooled = apply_partial_rope(pooled, cos, sin).to(torch.bfloat16)

    kv_values = torch.where(
        geometry.kv_write_valid.unsqueeze(-1), pooled, original_kv
    )
    kv_flat.scatter_(
        0, geometry.kv_write_slots.unsqueeze(-1).expand(-1, head_dim), kv_values
    )
    state_values = torch.where(
        geometry.state_allocated.unsqueeze(-1), raw_keys.float(), original_state
    )
    state_flat.scatter_(
        0, geometry.state_slots_safe.unsqueeze(-1).expand(-1, head_dim), state_values
    )

    result = {
        "state_slots": geometry.state_slots,
        "kv_slots": geometry.kv_slots,
        "completed": geometry.completed,
        # 0-dim device tensors: no production consumer reads these back,
        # so they stay lazy instead of two host-synchronizing sums.
        "num_state_writes": geometry.state_allocated.sum(),
        "num_kv_writes": geometry.kv_write_valid.sum(),
    }
    if capture_undo:
        result["undo"] = IndexerCacheUndo(
            state_pool=state_pool,
            state_slots=geometry.state_slots_safe,
            original_state=original_state,
            kv_pool=kv_pool,
            kv_slots=geometry.kv_write_slots,
            original_kv=original_kv,
        )
    return result


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
    geometry: Optional[IndexerWriterGeometry] = None,
) -> dict:
    """Device-vectorized writer used by the production CUDA path.

    Metadata validation has a constant number of host synchronizations. All
    token-to-page resolution, state writes, completed-group pooling and KV
    writes stay on the device; there is no per-token host scalar read or copy.

    ``geometry`` is the step-frozen :class:`IndexerWriterGeometry` (see
    :func:`build_indexer_writer_geometry`).  When it still describes this
    invocation and its slots are provably unique, the compression-free
    full-shape write path replaces the per-call ``nonzero``/``argsort``/
    ``unique`` chains -- byte-identical pool contents, no device-to-host
    synchronization.  A stale or inapplicable bundle (and strict validation,
    which keeps the legacy per-call diagnostics) falls back to the legacy body
    below.
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

    if qsa_strict_validation_enabled() or not writer_geometry_applies(
        geometry,
        cu_seqlens,
        start_positions,
        kv_block_table,
        state_block_table,
        kv_pool,
        state_pool,
        token_count=token_count,
        ratio=ratio,
        kv_tokens_per_block=kv_tokens_per_block,
        state_tokens_per_block=state_tokens_per_block,
        invalid_block_policy=invalid_block_policy,
    ):
        geometry = None
    if (
        geometry is not None
        and geometry.state_slots_unique
        and int(kv_pool.shape[0]) > 0
        and int(state_pool.shape[0]) > 0
    ):
        # The frozen bundle was validated against these exact tensors (identity
        # + fingerprint checks above), so the leading cu/rope asserts below
        # already ran once per step at build time.
        return _write_indexer_cache_full_shape(
            raw_keys,
            geometry,
            rope_cos,
            rope_sin,
            k_norm_gamma,
            kv_pool,
            state_pool,
            norm_eps=norm_eps,
            capture_undo=capture_undo,
            rope_is_token_aligned=rope_is_token_aligned,
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
    if qsa_strict_validation_enabled():
        if bool(invalid_cu.item()):
            raise ValueError(
                "cu_seqlens must be monotonic from 0 to raw_keys.shape[0], and "
                "start_positions must be non-negative"
            )
    else:
        qsa_assert_valid(
            ~invalid_cu,
            "cu_seqlens must be monotonic from 0 to raw_keys.shape[0], and "
            "start_positions must be non-negative",
        )
    if not rope_is_token_aligned:
        # Cold path (the production QSA writers pass token-aligned tables).
        if qsa_strict_validation_enabled():
            max_position = int((starts + lengths).max().item()) if batch_size else 0
            if max_position > int(rope_cos.shape[-2]):
                raise ValueError(
                    f"RoPE tables cover {rope_cos.shape[-2]} positions, "
                    f"need {max_position}"
                )
        elif batch_size:
            qsa_assert_valid(
                torch.all((starts + lengths) <= int(rope_cos.shape[-2])),
                f"RoPE tables cover {rope_cos.shape[-2]} positions, "
                "need more",
            )

    flat_indices = torch.arange(token_count, dtype=torch.long, device=device)
    request_ids = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long, device=device),
        lengths,
        output_size=token_count,
    )
    local_indices = flat_indices - cu.index_select(0, request_ids)
    positions = starts.index_select(0, request_ids) + local_indices

    def _resolve(
        table: torch.Tensor,
        requests: torch.Tensor,
        logical_blocks: torch.Tensor,
        *,
        pool_blocks: int,
        tag: str,
        required: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = int(table.shape[1])
        in_bounds = (logical_blocks >= 0) & (logical_blocks < width)
        safe_columns = logical_blocks.clamp(0, max(width - 1, 0))
        blocks = table[requests, safe_columns].to(torch.long)
        out_of_pool = required & in_bounds & (blocks >= pool_blocks)
        if qsa_strict_validation_enabled():
            if bool(torch.any(out_of_pool).item()):
                raise IndexError(
                    f"{tag} physical block is outside pool size {pool_blocks}"
                )
        else:
            qsa_assert_valid(
                ~torch.any(out_of_pool),
                f"{tag} physical block is outside pool size {pool_blocks}",
            )
        # The allocated mask stays materialized: the write plan below needs it
        # to mask out unresolved slots.
        allocated = in_bounds & (blocks > 0) & (blocks < pool_blocks)
        missing = required & ~allocated
        if invalid_block_policy == "raise":
            if qsa_strict_validation_enabled():
                if bool(torch.any(missing).item()):
                    raise ValueError(
                        f"{tag} resolves to an unallocated logical block"
                    )
            else:
                qsa_assert_valid(
                    ~torch.any(missing),
                    f"{tag} resolves to an unallocated logical block",
                )
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
    completed_indices = torch.nonzero(completed, as_tuple=False).flatten()
    kv_slots = torch.full_like(flat_indices, -1)
    kv_write_slots = torch.empty(0, dtype=torch.long, device=device)
    kv_write_values = raw_keys.new_empty((0, head_dim), dtype=torch.bfloat16)

    if int(completed_indices.numel()):
        completed_requests = request_ids.index_select(0, completed_indices)
        completed_positions = positions.index_select(0, completed_indices)
        block_starts = completed_positions - ratio + 1
        source_positions = block_starts[:, None] + torch.arange(
            ratio, dtype=torch.long, device=device
        )
        row_starts = starts.index_select(0, completed_requests)
        row_lengths = lengths.index_select(0, completed_requests)
        source_local = source_positions - row_starts[:, None]
        source_in_current = (source_local >= 0) & (source_local < row_lengths[:, None])

        source_requests = completed_requests[:, None].expand(-1, ratio)
        source_logical = source_positions // state_tokens_per_block
        source_blocks, source_allocated = _resolve(
            state_block_table,
            source_requests,
            source_logical,
            pool_blocks=int(state_pool.shape[0]),
            tag="indexer_state source",
            required=~source_in_current,
        )
        source_valid = source_in_current | source_allocated
        group_sources_valid = torch.all(source_valid, dim=1)

        raw_source_indices = (
            cu.index_select(0, source_requests.reshape(-1)).view_as(source_requests)
            + source_local
        ).clamp(0, max(token_count - 1, 0))
        state_source_slots = source_blocks.clamp(
            0, int(state_pool.shape[0]) - 1
        ) * state_ring_entries + source_positions.remainder(state_ring_entries)
        gathered_raw = raw_keys.index_select(0, raw_source_indices.reshape(-1)).view(
            -1, ratio, head_dim
        )
        gathered_state = (
            state_pool.flatten(0, 1)
            .index_select(0, state_source_slots.reshape(-1))
            .view(-1, ratio, head_dim)
        )
        gathered = torch.where(
            source_in_current.unsqueeze(-1), gathered_raw.float(), gathered_state
        )

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

    undo = None
    if capture_undo:
        state_undo_slots, original_state = _capture_tensor_slots(
            state_pool, state_write_slots
        )
        kv_undo_slots, original_kv = _capture_tensor_slots(kv_pool, kv_write_slots)
        undo = IndexerCacheUndo(
            state_pool=state_pool,
            state_slots=state_undo_slots,
            original_state=original_state,
            kv_pool=kv_pool,
            kv_slots=kv_undo_slots,
            original_kv=original_kv,
        )

    try:
        state_pool.flatten(0, 1).index_copy_(0, state_write_slots, state_write_values)
        kv_pool.flatten(0, 1).index_copy_(0, kv_write_slots, kv_write_values)
    except BaseException:
        if undo is not None:
            restore_indexer_cache(undo)
            if state_pool.is_cuda:
                torch.cuda.current_stream(device).synchronize()
        raise

    result = {
        "state_slots": state_slots,
        "kv_slots": kv_slots,
        "completed": completed,
        # 0-dim device int64 tensors: no production consumer reads these back,
        # so they stay lazy instead of two host-synchronizing sums. Tests only
        # compare them for equality (bool() of the 0-dim tensor is free there).
        "num_state_writes": state_write_mask.sum(),
        "num_kv_writes": (kv_slots >= 0).sum(),
    }
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
    rope_is_token_aligned: bool = False,
    geometry: Optional[IndexerWriterGeometry] = None,
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

    By default, ``rope_cos`` / ``rope_sin`` cover absolute positions and may be
    shared ``[max_position, rotary_dim]`` or request-specific
    ``[B, max_position, rotary_dim]``. With ``rope_is_token_aligned=True``, they
    instead contain one ``[rotary_dim]`` block-start row for each packed input
    token; only rows that complete a compression group are consumed.
    """
    if invalid_block_policy not in ("raise", "skip"):
        raise ValueError(
            "invalid_block_policy must be 'raise' or 'skip', got "
            f"{invalid_block_policy!r}"
        )
    if geometry is not None and not isinstance(geometry, IndexerWriterGeometry):
        raise ValueError("geometry must be an IndexerWriterGeometry or None")
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
            rope_is_token_aligned=rope_is_token_aligned,
            geometry=geometry,
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
        "num_state_writes": len(state_writes),
        "num_kv_writes": len(kv_writes),
    }
    if capture_undo:
        result["undo"] = undo
    return result


def compress_prefill(
    hidden_states: torch.Tensor,  # [B, T, D]
    qk_proj: torch.Tensor,  # [(n_heads+1)*head_dim, D]
    k_norm_gamma: torch.Tensor,  # [head_dim]
    cos: torch.Tensor,  # [B, T, rotary_dim] (full-range)
    sin: torch.Tensor,  # [B, T, rotary_dim]
    *,
    ratio: int,
    head_dim: int,
    norm_eps: float = 1e-6,
) -> dict:
    """Full prefill compression returning the compressed pool entries + raw keys.

    This is the **flat-buffer form**: instead of writing to framework-managed paged
    pools, it returns tensors that let the caller (or a test) verify the compressed
    content by placing it in a pool buffer.

    Returns a dict::

        raw_keys      [B, T, D] fp32
        pooled_k      [B, nb, D] bf16          (one per **complete** block)
        block_starts  [B, nb] int64            first token index of each block
        num_blocks    [B] int64                how many complete blocks per batch

    The caller can copy ``pooled_k[i, b]`` directly into the bf16 view of the
    ``indexer_kv`` pool at slot ``i * num_blocks + b``.
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must be [B, T, hidden], got {tuple(hidden_states.shape)}"
        )
    if qk_proj.dim() != 2 or qk_proj.shape[1] != hidden_states.shape[-1]:
        raise ValueError(
            "qk_proj must be [(n_heads + 1) * head_dim, hidden], got "
            f"{tuple(qk_proj.shape)} for hidden={hidden_states.shape[-1]}"
        )
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if head_dim <= 0 or qk_proj.shape[0] % head_dim:
        raise ValueError(
            f"qk_proj output {qk_proj.shape[0]} is not divisible by head_dim={head_dim}"
        )
    n_heads = (qk_proj.shape[0] // head_dim) - 1
    if n_heads <= 0:
        raise ValueError("qk_proj must contain at least one Q head and one K head")
    if k_norm_gamma.shape != (head_dim,):
        raise ValueError(
            f"k_norm_gamma must be [{head_dim}], got {tuple(k_norm_gamma.shape)}"
        )
    B, T, _ = hidden_states.shape
    device = hidden_states.device
    if cos.shape[:2] != (B, T) or sin.shape != cos.shape:
        raise ValueError(
            f"cos/sin must have matching [B, T, rotary_dim] shapes, got "
            f"{tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    rotary_dim = cos.shape[-1]
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= head_dim; got {rotary_dim}"
        )

    qk = hidden_states @ qk_proj.T
    _, raw_k = qk.split([n_heads * head_dim, head_dim], dim=-1)
    raw_keys = raw_k.float()  # [B, T, D] fp32

    num_blocks_full = T // ratio
    # Truncate raw_keys to the token positions that form complete blocks.
    grouped = raw_keys[:, : num_blocks_full * ratio]  # [B, nb*ratio, D]
    grouped = grouped.unflatten(1, (num_blocks_full, ratio))  # [B, nb, ratio, D]
    # Mean in fp32 (upstream orders the summation the same way).
    pooled = grouped.mean(dim=2).to(raw_k.dtype)  # [B, nb, D]

    # Apply k_layernorm (with +1) and partial RoPE at each block's first position.
    pooled = exact_head_rms_norm(pooled, k_norm_gamma[None, None, :], norm_eps)
    block_starts = torch.arange(num_blocks_full, device=device) * ratio  # [nb]
    pooled_rotated = apply_partial_rope(
        pooled,
        cos[:, block_starts],
        sin[:, block_starts],
    )
    pooled_k = pooled_rotated.to(torch.bfloat16).contiguous()

    return {
        "raw_keys": raw_keys,
        "pooled_k": pooled_k,
        "block_starts": block_starts.expand(B, -1),
        "num_blocks": torch.full(
            (B,), num_blocks_full, dtype=torch.long, device=device
        ),
    }

"""DSV4 SWA prefill ops — vLLM-aligned Triton kernels.

Two kernels vendored verbatim (signature + math) from vLLM:

* ``compute_prefill_gather_lens`` — vLLM
  ``vllm/v1/attention/backends/mla/sparse_swa.py:_compute_prefill_metadata_kernel``.
  Per request: ``gather_len = query_len + min(prefix_len, window_size - 1)``.
  Drives ``dequantize_and_gather_k_cache``'s gather amount so every prefill
  query has its full SWA window in the BF16 workspace.

* ``combine_topk_swa_indices`` — vLLM ``cache_utils.py:445``. Per query
  token, lays out a ``[combined_topk]`` int32 row of:
    - first ``topk_len`` entries: compressed-attn indices, offset by
      ``M * batch_idx`` so they land in this batch's slice of the gathered
      workspace
    - next ``swa_len`` entries: SWA window indices into the same workspace
      (shifted by ``N`` to skip the compressed segment)
  + a parallel ``[num_tokens]`` int32 ``combined_lens`` so
  ``flash_mla_sparse_fwd``'s ``topk_length`` masks the right tail.

The paged SWA write helper in this module builds flat slot ids for the current
ring layout; SWA reads consume those flat slots in ``_swa_dequant_triton.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

# Mirror vLLM's combined-topk row alignment (kernel tile width).
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128

# Production caps request batching at MAX_BATCH_SIZE=1024.  Pinning the
# gather-lens tile to one bucket keeps batch-shape changes from creating new
# Triton keys after startup warmup.
_GATHER_LENS_FIXED_BLOCK_SIZE = 1024


# ---------------------------------------------------------------------------
# 1) Per-request gather_len = query_len + min(prefix_len, window_size - 1)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_prefills", "num_decodes", "window_size"])
def _compute_prefill_gather_lens_kernel(
    # Outputs
    prefill_gather_lens_ptr,
    # Inputs
    seq_lens_ptr,
    query_start_loc_ptr,
    num_prefills,
    num_decodes,
    window_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass compute of per-prefill ``gather_len``.

    For prefill request ``i`` (offset by ``num_decodes`` in the merged
    decode+prefill batch view):
        query_len  = qsl[num_decodes+i+1] - qsl[num_decodes+i]
        prefix_len = seq_len - query_len             # already-cached tokens
        gather_len = query_len + min(prefix_len, window_size - 1)

    The ``window_size - 1`` cap is the SWA invariant: the earliest prefill
    query at sequence position p reuses at most win-1 prefix tokens
    (positions p-win+1 .. p-1), so gathering more than that is wasted.
    """
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < num_prefills

    seq_len = tl.load(seq_lens_ptr + num_decodes + offset, mask=mask)
    qsl_start = tl.load(query_start_loc_ptr + num_decodes + offset, mask=mask)
    qsl_end = tl.load(query_start_loc_ptr + num_decodes + offset + 1, mask=mask)

    query_len = qsl_end - qsl_start
    prefix_len = seq_len - query_len
    gather_len = query_len + tl.minimum(prefix_len, window_size - 1)

    tl.store(prefill_gather_lens_ptr + offset, gather_len, mask=mask)


def compute_prefill_gather_lens(
    seq_lens: torch.Tensor,  # [num_decodes + num_prefills] int32 — total seq lens
    query_start_loc: torch.Tensor,  # [num_decodes + num_prefills + 1] int32
    num_prefills: int,
    num_decodes: int,
    window_size: int,
) -> torch.Tensor:
    """Return ``[num_prefills]`` int32 ``gather_lens``.

    Args:
      seq_lens: total token count per request (decodes + prefills, in that order).
      query_start_loc: cumulative new-token offsets; length = batch_size + 1.
      num_prefills:  count of prefill requests in the merged batch.
      num_decodes:   count of decode requests preceding the prefills.
      window_size:   SWA window (≥1).
    """
    assert seq_lens.dtype == torch.int32 and query_start_loc.dtype == torch.int32, (
        f"seq_lens and query_start_loc must be int32, got {seq_lens.dtype} / "
        f"{query_start_loc.dtype}"
    )
    assert num_prefills >= 0 and num_decodes >= 0 and window_size >= 1
    out = torch.empty(num_prefills, dtype=torch.int32, device=seq_lens.device)
    if num_prefills == 0:
        return out
    if num_prefills > _GATHER_LENS_FIXED_BLOCK_SIZE:
        raise ValueError(
            f"num_prefills={num_prefills} exceeds fixed BLOCK_SIZE="
            f"{_GATHER_LENS_FIXED_BLOCK_SIZE}; bump _GATHER_LENS_FIXED_BLOCK_SIZE."
        )
    block_size = _GATHER_LENS_FIXED_BLOCK_SIZE
    _compute_prefill_gather_lens_kernel[(1,)](
        out,
        seq_lens,
        query_start_loc,
        num_prefills,
        num_decodes,
        window_size,
        BLOCK_SIZE=block_size,
    )
    return out


# ---------------------------------------------------------------------------
# 1b) Per-token flat SWA window indices + topk length
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_tokens"])
def _compute_window_topk_and_length_varlen_kernel(
    topk_idxs_ptr,  # [num_tokens, window_size] int32
    topk_length_ptr,  # [num_tokens] int32
    cu_seqlens_ptr,  # [B+1] int32
    position_ids_ptr,  # [num_tokens] int32/int64
    prefix_lengths_ptr,  # [B] int32
    req_id_per_token_ptr,  # [num_tokens] int32
    num_tokens,
    window_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Build varlen SWA cold-path topk metadata in one launch.

    For token ``t`` in request ``b``:

      local_pos = position_ids[t] - prefix_lengths[b]
      topk_length[t] = min(local_pos + 1, window_size)
      topk_idxs[t, j] = cu_seqlens[b] + max(0, local_pos-window+1) + j
                       or -1 when j exceeds the causal window.

    This is the fused equivalent of ``_get_window_topk_idxs_varlen`` plus the
    ``topk_length_kv_full`` math in ``_build_swa_prefill_meta_varlen``.
    """
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, BLOCK_W)
    row_mask = rows < num_tokens
    col_mask = cols < window_size

    req = tl.load(req_id_per_token_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    pos = tl.load(position_ids_ptr + rows, mask=row_mask, other=0).to(tl.int32)
    prefix = tl.load(prefix_lengths_ptr + req, mask=row_mask, other=0).to(tl.int32)
    req_start = tl.load(cu_seqlens_ptr + req, mask=row_mask, other=0).to(tl.int32)

    local_pos = pos - prefix
    win_start = tl.maximum(local_pos - window_size + 1, 0)
    topk_len = tl.minimum(local_pos + 1, window_size)
    tl.store(topk_length_ptr + rows, topk_len, mask=row_mask)

    idx = req_start[:, None] + win_start[:, None] + cols[None, :]
    valid = (win_start[:, None] + cols[None, :]) <= local_pos[:, None]
    out = tl.where(valid, idx, -1)
    tl.store(
        topk_idxs_ptr + rows[:, None] * window_size + cols[None, :],
        out,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def compute_window_topk_and_length_varlen(
    window_size: int,
    cu_seqlens: torch.Tensor,
    position_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    req_id_per_token: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(topk_idxs, topk_length)`` for varlen SWA prefill.

    ``topk_idxs`` matches ``_get_window_topk_idxs_varlen``'s int32
    ``[num_tokens, window_size]`` flat-KV indices. ``topk_length`` matches the
    per-token ``min(local_pos + 1, window_size)`` vector consumed by
    ``flash_mla_sparse_fwd`` on the SWA ``kv_full`` fallback path.

    CPU / unsupported inputs fall back to the same torch expression shape so
    unit tests and dry paths can run without Triton.
    """
    window_size = int(window_size)
    cu_seqlens = cu_seqlens.reshape(-1)
    position_ids = position_ids.reshape(-1)
    prefix_lengths = prefix_lengths.reshape(-1)
    req_id_per_token = req_id_per_token.reshape(-1)
    assert window_size >= 1
    assert position_ids.numel() == req_id_per_token.numel()

    device = position_ids.device
    num_tokens = int(position_ids.numel())
    topk_idxs = torch.empty((num_tokens, window_size), dtype=torch.int32, device=device)
    topk_length = torch.empty(num_tokens, dtype=torch.int32, device=device)
    if num_tokens == 0:
        return topk_idxs, topk_length

    if not position_ids.is_cuda:
        req_id_idx = req_id_per_token.to(device=device, dtype=torch.long)
        cu_i32 = cu_seqlens.to(device=device, dtype=torch.int32)
        prefix_i32 = prefix_lengths.to(device=device, dtype=torch.int32)
        pos_i32 = position_ids.to(device=device, dtype=torch.int32)
        prefix_per_token = prefix_i32.gather(0, req_id_idx)
        req_start_in_flat = cu_i32.gather(0, req_id_idx)
        local_query_pos = pos_i32 - prefix_per_token
        offsets = torch.arange(window_size, device=device, dtype=torch.int32)
        win_start = (local_query_pos.unsqueeze(1) - window_size + 1).clamp_min(0)
        local_idx = win_start + offsets
        topk_idxs.copy_(
            torch.where(
                local_idx > local_query_pos.unsqueeze(1),
                torch.full_like(local_idx, -1),
                req_start_in_flat.unsqueeze(1) + local_idx,
            ).contiguous()
        )
        topk_length.copy_(torch.clamp(local_query_pos + 1, max=window_size))
        return topk_idxs, topk_length

    if cu_seqlens.dtype != torch.int32 or cu_seqlens.device != device:
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
    if prefix_lengths.dtype != torch.int32 or prefix_lengths.device != device:
        prefix_lengths = prefix_lengths.to(
            device=device, dtype=torch.int32
        ).contiguous()
    if req_id_per_token.dtype != torch.int32 or req_id_per_token.device != device:
        req_id_per_token = req_id_per_token.to(
            device=device, dtype=torch.int32
        ).contiguous()
    if not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()
    if not position_ids.is_contiguous():
        position_ids = position_ids.contiguous()
    if not prefix_lengths.is_contiguous():
        prefix_lengths = prefix_lengths.contiguous()
    if not req_id_per_token.is_contiguous():
        req_id_per_token = req_id_per_token.contiguous()

    block_t = 16
    block_w = max(1, triton.next_power_of_2(window_size))
    _compute_window_topk_and_length_varlen_kernel[
        ((num_tokens + block_t - 1) // block_t,)
    ](
        topk_idxs,
        topk_length,
        cu_seqlens,
        position_ids,
        prefix_lengths,
        req_id_per_token,
        num_tokens,
        window_size=window_size,
        BLOCK_T=block_t,
        BLOCK_W=block_w,
    )
    return topk_idxs, topk_length


# ---------------------------------------------------------------------------
# 2) Per-token SWA paged slot_mapping (write side)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_reqs", "max_blocks_per_seq"])
def _compute_swa_slot_mapping_kernel(
    # Output
    slot_mapping_ptr,  # [num_tokens] int64
    # Inputs
    block_table_ptr,  # [num_reqs, max_blocks_per_seq] int32
    query_start_loc_ptr,  # [num_reqs+1] int32 — chunk-local cumulative
    seq_lens_ptr,  # [num_reqs] int32 — total seq len per req (sp + S_local)
    # Constants
    num_reqs,
    max_blocks_per_seq,
    pool_entries_per_block: tl.constexpr,
    tokens_per_block_for_block_table: tl.constexpr,
    ring_entries: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tokens per program
):
    """One program per (request, BLOCK_M chunk-of-tokens).

    For request ``b``, chunk-local token index ``i`` (= absolute token
    in flattened ``[T]`` view at ``query_start[b] + i``):

        global_pos    = (seq_len[b] - query_len[b]) + i
        block_in_seq  = global_pos // tokens_per_block_for_block_table
        in_block      = global_pos % ring_entries
        block_id      = block_table[b, block_in_seq]
        slot          = -1                       if block_id <= 0
                        block_id * pool_entries_per_block + in_block otherwise

    SWA paged-write semantics: only the final ``ring_entries`` positions before
    each physical block boundary or request end are writable. Earlier tokens in
    the same physical block map to ``-1`` so concurrent prefill writes do not
    race on the small ring.
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    chunk_idx = tl.program_id(1)

    if batch_idx >= num_reqs:
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    sp = seq_len - query_len  # absolute pos of first new token

    chunk_start = chunk_idx * BLOCK_M
    i = chunk_start + tl.arange(0, BLOCK_M)  # [BLOCK_M] chunk-local
    valid_token = i < query_len
    global_pos = sp + i  # [BLOCK_M] absolute

    block_in_seq = global_pos // tokens_per_block_for_block_table
    in_block = global_pos % ring_entries

    in_capacity = block_in_seq < max_blocks_per_seq
    safe_block_in_seq = tl.where(in_capacity, block_in_seq, 0)

    bt_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
    block_id = tl.load(bt_row_ptr + safe_block_in_seq, mask=valid_token, other=-1)

    block_end = (block_in_seq + 1) * tokens_per_block_for_block_table
    effective_end = tl.minimum(block_end, seq_len)
    tail_write = (global_pos + ring_entries) >= effective_end
    valid = (block_id > 0) & in_capacity & valid_token & tail_write
    slot = tl.where(
        valid,
        block_id.to(tl.int64) * pool_entries_per_block + in_block.to(tl.int64),
        tl.full((BLOCK_M,), -1, dtype=tl.int64),
    )

    out_ptr = slot_mapping_ptr + query_start.to(tl.int64) + i.to(tl.int64)
    tl.store(out_ptr, slot, mask=valid_token)


def compute_swa_slot_mapping(
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq] int32
    query_start_loc: torch.Tensor,  # [num_reqs+1] int32 — chunk-local cumulative
    seq_lens: torch.Tensor,  # [num_reqs] int32 — total seq len = sp + query_len
    num_tokens: int,  # total tokens across all reqs (= query_start_loc[-1])
    *,
    pool_entries_per_block: int,
    tokens_per_block_for_block_table: int,
    ring_entries: int,
) -> torch.Tensor:
    """Build ``[num_tokens]`` int64 slot_mapping for SWA paged FP8 write.

    New layout parameters are intentionally split:
      * ``pool_entries_per_block``: SWA pool tensor second dimension.
      * ``tokens_per_block_for_block_table``: raw-token coverage of a
        block-table row.
      * ``ring_entries``: in-block modulo domain.

    Only the final ``ring_entries`` tokens of each physical block/request tail
    are written. This matches state-ring writes and avoids ring collisions when
    physical blocks are much larger than the SWA ring.
    """
    assert (
        block_table.dtype == torch.int32
    ), f"block_table must be int32, got {block_table.dtype}"
    assert (
        query_start_loc.dtype == torch.int32 and seq_lens.dtype == torch.int32
    ), "query_start_loc and seq_lens must be int32"
    pool_entries_per_block = int(pool_entries_per_block)
    tokens_per_block_for_block_table = int(tokens_per_block_for_block_table)
    ring_entries = int(ring_entries)
    assert pool_entries_per_block >= 1
    assert tokens_per_block_for_block_table >= 1
    assert ring_entries >= 1
    device = block_table.device
    slot_mapping = torch.empty(num_tokens, dtype=torch.long, device=device)
    if num_tokens == 0:
        return slot_mapping

    num_reqs = int(seq_lens.shape[0])
    max_blocks_per_seq = int(block_table.shape[1])
    BLOCK_M = 256
    num_chunks = max(1, (num_tokens + BLOCK_M - 1) // BLOCK_M)
    grid = (num_reqs, num_chunks)
    _compute_swa_slot_mapping_kernel[grid](
        slot_mapping,
        block_table,
        query_start_loc,
        seq_lens,
        num_reqs,
        max_blocks_per_seq=max_blocks_per_seq,
        pool_entries_per_block=pool_entries_per_block,
        tokens_per_block_for_block_table=tokens_per_block_for_block_table,
        ring_entries=ring_entries,
        BLOCK_M=BLOCK_M,
    )
    return slot_mapping


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "max_blocks_per_seq"])
def _compute_swa_slot_mapping_from_positions_kernel(
    slot_mapping_ptr,  # [num_tokens] int64
    block_table_ptr,  # [num_reqs, max_blocks_per_seq] int32
    req_id_per_token_ptr,  # [num_tokens] int32; -1 means padding/hole
    positions_ptr,  # [num_tokens] int32; absolute token positions
    seq_lens_ptr,  # [num_reqs] int32; committed request ends (exclusive)
    num_tokens,
    num_reqs,
    max_blocks_per_seq,
    pool_entries_per_block: tl.constexpr,
    tokens_per_block_for_block_table: tl.constexpr,
    ring_entries: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Map independent token positions to the final writable SWA ring tails."""
    offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    in_token_range = offsets < num_tokens

    req_id = tl.load(req_id_per_token_ptr + offsets, mask=in_token_range, other=-1).to(
        tl.int64
    )
    position = tl.load(positions_ptr + offsets, mask=in_token_range, other=-1).to(
        tl.int64
    )
    valid_req = in_token_range & (req_id >= 0) & (req_id < num_reqs)
    safe_req_id = tl.where(valid_req, req_id, 0)
    seq_len = tl.load(seq_lens_ptr + safe_req_id, mask=valid_req, other=0).to(tl.int64)
    valid_position = valid_req & (position >= 0) & (position < seq_len)

    block_in_seq = position // tokens_per_block_for_block_table
    in_capacity = (
        valid_position & (block_in_seq >= 0) & (block_in_seq < max_blocks_per_seq)
    )
    safe_block_in_seq = tl.where(in_capacity, block_in_seq, 0)
    block_table_offset = safe_req_id * max_blocks_per_seq + safe_block_in_seq
    block_id = tl.load(
        block_table_ptr + block_table_offset, mask=in_capacity, other=-1
    ).to(tl.int64)

    block_end = (block_in_seq + 1) * tokens_per_block_for_block_table
    effective_end = tl.minimum(block_end, seq_len)
    tail_write = (position + ring_entries) >= effective_end
    valid = in_capacity & (block_id > 0) & tail_write
    slot = tl.where(
        valid,
        block_id * pool_entries_per_block + (position % ring_entries),
        tl.full((BLOCK_M,), -1, dtype=tl.int64),
    )
    tl.store(slot_mapping_ptr + offsets, slot, mask=in_token_range)


def compute_swa_slot_mapping_from_positions(
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq] int32
    req_id_per_token: torch.Tensor,  # [num_tokens] int32; -1 means padding/hole
    positions: torch.Tensor,  # [num_tokens] int32; absolute token positions
    seq_lens: torch.Tensor,  # [num_reqs] int32; committed request ends
    num_tokens: int,
    pool_entries_per_block: int,
    tokens_per_block_for_block_table: int,
    ring_entries: int,
) -> torch.Tensor:
    """Build SWA write slots directly from each token's request and position.

    A token is writable only when it belongs to the final ``ring_entries``
    positions before either its physical block boundary or its request's
    committed end. Invalid request ids, invalid positions, unallocated blocks,
    and older generations of the same ring slot map to ``-1``.
    """
    assert block_table.ndim == 2
    assert seq_lens.ndim == 1
    assert block_table.dtype == torch.int32
    assert req_id_per_token.dtype == torch.int32
    assert positions.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert block_table.shape[0] == seq_lens.numel()
    assert 0 <= num_tokens <= min(req_id_per_token.numel(), positions.numel())
    assert (
        block_table.device
        == req_id_per_token.device
        == positions.device
        == seq_lens.device
    )

    block_table = block_table.contiguous()
    req_id_per_token = req_id_per_token.reshape(-1).contiguous()
    positions = positions.reshape(-1).contiguous()
    seq_lens = seq_lens.contiguous()

    pool_entries_per_block = int(pool_entries_per_block)
    tokens_per_block_for_block_table = int(tokens_per_block_for_block_table)
    ring_entries = int(ring_entries)
    assert pool_entries_per_block >= 1
    assert tokens_per_block_for_block_table >= 1
    assert ring_entries >= 1
    assert ring_entries <= pool_entries_per_block

    slot_mapping = torch.empty(num_tokens, dtype=torch.long, device=block_table.device)
    if num_tokens == 0:
        return slot_mapping

    num_reqs = int(seq_lens.numel())
    max_blocks_per_seq = int(block_table.shape[1])
    BLOCK_M = 256
    grid = ((num_tokens + BLOCK_M - 1) // BLOCK_M,)
    _compute_swa_slot_mapping_from_positions_kernel[grid](
        slot_mapping,
        block_table,
        req_id_per_token,
        positions,
        seq_lens,
        num_tokens,
        num_reqs,
        max_blocks_per_seq,
        pool_entries_per_block=pool_entries_per_block,
        tokens_per_block_for_block_table=tokens_per_block_for_block_table,
        ring_entries=ring_entries,
        BLOCK_M=BLOCK_M,
    )
    return slot_mapping


@triton.jit(do_not_specialize=["num_reqs", "max_blocks_per_seq"])
def _compute_swa_cp_sliced_slot_mapping_kernel(
    slot_mapping_ptr,
    block_table_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    num_reqs,
    max_blocks_per_seq,
    tokens_per_block_for_block_table: tl.constexpr,
    local_entries_per_block: tl.constexpr,
    full_entries_per_block: tl.constexpr,
    cp_rank: tl.constexpr,
    cp_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    chunk_idx = tl.program_id(1)

    if batch_idx >= num_reqs:
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    sp = seq_len - query_len

    chunk_start = chunk_idx * BLOCK_M
    i = chunk_start + tl.arange(0, BLOCK_M)
    valid_token = i < query_len
    global_pos = sp + i

    block_in_seq = global_pos // tokens_per_block_for_block_table
    ring_offset = global_pos % full_entries_per_block
    owner_rank = ring_offset // local_entries_per_block
    local_offset = ring_offset - owner_rank * local_entries_per_block
    owned_by_rank = owner_rank == cp_rank

    in_capacity = block_in_seq < max_blocks_per_seq
    safe_block_in_seq = tl.where(in_capacity, block_in_seq, 0)

    bt_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
    block_id = tl.load(bt_row_ptr + safe_block_in_seq, mask=valid_token, other=-1)

    block_end = (block_in_seq + 1) * tokens_per_block_for_block_table
    effective_end = tl.minimum(block_end, seq_len)
    tail_write = (global_pos + full_entries_per_block) >= effective_end
    valid = (block_id > 0) & in_capacity & valid_token & owned_by_rank & tail_write
    slot = tl.where(
        valid,
        block_id.to(tl.int64) * local_entries_per_block + local_offset.to(tl.int64),
        tl.full((BLOCK_M,), -1, dtype=tl.int64),
    )

    out_ptr = slot_mapping_ptr + query_start.to(tl.int64) + i.to(tl.int64)
    tl.store(out_ptr, slot, mask=valid_token)


def compute_swa_cp_sliced_slot_mapping(
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_tokens: int,
    *,
    tokens_per_block_for_block_table: int,
    local_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Build SWA slot mapping for CP-sliced fixed/SWA blocks.

    ``tokens_per_block_for_block_table`` is the logical/cache-key block size.
    ``local_entries_per_block * cp_size`` is the full SWA ring size. These are
    intentionally separate: logical block rows and SWA ring entries need not
    match or divide each other.
    """
    assert (
        block_table.dtype == torch.int32
    ), f"block_table must be int32, got {block_table.dtype}"
    assert (
        query_start_loc.dtype == torch.int32 and seq_lens.dtype == torch.int32
    ), "query_start_loc and seq_lens must be int32"
    tokens_per_block_for_block_table = int(tokens_per_block_for_block_table)
    local_entries_per_block = int(local_entries_per_block)
    cp_rank = int(cp_rank)
    cp_size = int(cp_size)
    assert cp_size > 1 and 0 <= cp_rank < cp_size
    assert tokens_per_block_for_block_table >= 1 and local_entries_per_block >= 1
    full_entries_per_block = local_entries_per_block * cp_size

    device = block_table.device
    slot_mapping = torch.empty(num_tokens, dtype=torch.long, device=device)
    if num_tokens == 0:
        return slot_mapping

    num_reqs = int(seq_lens.shape[0])
    max_blocks_per_seq = int(block_table.shape[1])
    BLOCK_M = 256
    num_chunks = max(1, (num_tokens + BLOCK_M - 1) // BLOCK_M)
    grid = (num_reqs, num_chunks)
    _compute_swa_cp_sliced_slot_mapping_kernel[grid](
        slot_mapping,
        block_table,
        query_start_loc,
        seq_lens,
        num_reqs,
        max_blocks_per_seq=max_blocks_per_seq,
        tokens_per_block_for_block_table=tokens_per_block_for_block_table,
        local_entries_per_block=local_entries_per_block,
        full_entries_per_block=full_entries_per_block,
        cp_rank=cp_rank,
        cp_size=cp_size,
        BLOCK_M=BLOCK_M,
    )
    return slot_mapping


# ---------------------------------------------------------------------------
# 2b) Per-token workspace slot_in_flat metadata
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_tokens", "M", "base_offset"])
def _compute_swa_slot_in_flat_kernel(
    out_ptr,  # [num_tokens] int64
    position_ids_ptr,  # [num_tokens] int64/int32 absolute positions
    req_id_per_token_ptr,  # [num_tokens] int64/int32
    prefix_lengths_ptr,  # [B] int64/int32
    cu_swa_offsets_ptr,  # optional [>=B] int32/int64 ragged SWA bases
    num_tokens,
    M,
    base_offset,
    window_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_CU_SWA_OFFSETS: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < num_tokens
    req = tl.load(req_id_per_token_ptr + rows, mask=mask, other=0).to(tl.int64)
    pos = tl.load(position_ids_ptr + rows, mask=mask, other=0).to(tl.int64)
    prefix = tl.load(prefix_lengths_ptr + req, mask=mask, other=0).to(tl.int64)
    p = tl.minimum(prefix, window_size - 1)
    if HAS_CU_SWA_OFFSETS:
        swa_base = tl.load(cu_swa_offsets_ptr + req, mask=mask, other=0).to(tl.int64)
        slot = swa_base + p + (pos - prefix)
    else:
        slot = req * M + base_offset + p + (pos - prefix)
    tl.store(out_ptr + rows, slot, mask=mask)


def compute_swa_slot_in_flat(
    position_ids: torch.Tensor,
    req_id_per_token: torch.Tensor,
    prefix_lengths: torch.Tensor,
    *,
    M: int = 0,
    window_size: int,
    base_offset: int = 0,
    cu_swa_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build ``swa_base[req] + min(prefix, win-1) + (pos-prefix)``.

    Uses ``cu_swa_offsets[req]`` for ragged 2D workspaces when provided, else
    ``req * M + base_offset`` for padded 3D workspaces.
    """
    position_ids = position_ids.reshape(-1)
    req_id_per_token = req_id_per_token.reshape(-1)
    prefix_lengths = prefix_lengths.reshape(-1)
    assert position_ids.numel() == req_id_per_token.numel()
    num_tokens = int(position_ids.numel())
    device = position_ids.device
    has_cu_swa_offsets = cu_swa_offsets is not None
    if has_cu_swa_offsets:
        cu_swa_offsets = cu_swa_offsets.reshape(-1)
        assert (
            int(cu_swa_offsets.numel()) >= int(prefix_lengths.numel())
            and cu_swa_offsets.device == device
        )
    out = torch.empty(num_tokens, dtype=torch.long, device=device)
    if num_tokens == 0:
        return out

    if not position_ids.is_cuda:
        req = req_id_per_token.to(device=device, dtype=torch.long)
        pos = position_ids.to(device=device, dtype=torch.long)
        prefix = prefix_lengths.to(device=device, dtype=torch.long)
        p = torch.clamp_max(prefix, int(window_size) - 1)
        if has_cu_swa_offsets:
            swa_base = cu_swa_offsets.to(device=device, dtype=torch.long).gather(0, req)
        else:
            swa_base = req * int(M) + int(base_offset)
        out.copy_(
            (swa_base + p.gather(0, req) + (pos - prefix.gather(0, req))).contiguous()
        )
        return out

    if not position_ids.is_contiguous():
        position_ids = position_ids.contiguous()
    if not req_id_per_token.is_contiguous():
        req_id_per_token = req_id_per_token.contiguous()
    if not prefix_lengths.is_contiguous():
        prefix_lengths = prefix_lengths.contiguous()
    if has_cu_swa_offsets and not cu_swa_offsets.is_contiguous():
        cu_swa_offsets = cu_swa_offsets.contiguous()
    block_m = 256
    _compute_swa_slot_in_flat_kernel[(triton.cdiv(num_tokens, block_m),)](
        out,
        position_ids,
        req_id_per_token,
        prefix_lengths,
        cu_swa_offsets,
        num_tokens,
        int(M),
        int(base_offset),
        window_size=int(window_size),
        BLOCK_M=block_m,
        HAS_CU_SWA_OFFSETS=has_cu_swa_offsets,
    )
    return out


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "M",
        "base_offset",
        "window_size",
        "num_reqs",
    ]
)
def _compute_swa_slot_in_flat_from_cu_kernel(
    out_ptr,  # [num_tokens] int64
    cu_seqlens_ptr,  # [B+1] int32/int64 full-view starts
    prefix_lengths_ptr,  # [B] int32/int64
    cu_swa_offsets_ptr,  # optional [>=B] int32/int64 ragged SWA bases
    num_tokens,
    M,
    base_offset,
    window_size,
    num_reqs,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_CU_SWA_OFFSETS: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask = rows < num_tokens
    b = tl.arange(0, BLOCK_B)
    ends = tl.load(
        cu_seqlens_ptr + 1 + b,
        mask=b < num_reqs,
        other=num_tokens,
    ).to(tl.int64)
    req = tl.sum(tl.where(rows[:, None] >= ends[None, :], 1, 0), axis=1).to(tl.int64)
    req = tl.minimum(req, num_reqs - 1)
    start = tl.load(cu_seqlens_ptr + req, mask=mask, other=0).to(tl.int64)
    prefix = tl.load(prefix_lengths_ptr + req, mask=mask, other=0).to(tl.int64)
    p = tl.minimum(prefix, window_size - 1)
    if HAS_CU_SWA_OFFSETS:
        swa_base = tl.load(cu_swa_offsets_ptr + req, mask=mask, other=0).to(tl.int64)
        slot = swa_base + p + (rows - start)
    else:
        slot = req * M + base_offset + p + (rows - start)
    tl.store(out_ptr + rows, slot, mask=mask)


def compute_swa_slot_in_flat_from_cu(
    cu_seqlens: torch.Tensor,
    prefix_lengths: torch.Tensor,
    *,
    num_tokens: int,
    M: int = 0,
    window_size: int,
    base_offset: int = 0,
    cu_swa_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build ``slot_in_flat`` for a full-view CP workspace from cu-seqlens."""
    cu_seqlens = cu_seqlens.reshape(-1)
    prefix_lengths = prefix_lengths.reshape(-1)
    assert cu_seqlens.numel() == prefix_lengths.numel() + 1
    num_tokens = int(num_tokens)
    device = cu_seqlens.device
    has_cu_swa_offsets = cu_swa_offsets is not None
    if has_cu_swa_offsets:
        cu_swa_offsets = cu_swa_offsets.reshape(-1)
        assert (
            int(cu_swa_offsets.numel()) >= int(prefix_lengths.numel())
            and cu_swa_offsets.device == device
        )
    out = torch.empty(num_tokens, dtype=torch.long, device=device)
    if num_tokens == 0:
        return out

    if not cu_seqlens.is_cuda:
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        prefix = prefix_lengths.to(device=device, dtype=torch.long)
        req = torch.bucketize(
            torch.arange(num_tokens, device=device, dtype=torch.long),
            cu[1:],
            right=True,
        ).clamp(max=prefix.numel() - 1)
        p = torch.clamp_max(prefix, int(window_size) - 1)
        local_pos = torch.arange(
            num_tokens, device=device, dtype=torch.long
        ) - cu.gather(0, req)
        if has_cu_swa_offsets:
            swa_base = cu_swa_offsets.to(device=device, dtype=torch.long).gather(0, req)
        else:
            swa_base = req * int(M) + int(base_offset)
        out.copy_((swa_base + p.gather(0, req) + local_pos).contiguous())
        return out

    if not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()
    if not prefix_lengths.is_contiguous():
        prefix_lengths = prefix_lengths.contiguous()
    if has_cu_swa_offsets and not cu_swa_offsets.is_contiguous():
        cu_swa_offsets = cu_swa_offsets.contiguous()
    num_reqs = int(prefix_lengths.numel())
    if num_reqs > 1024:
        raise ValueError(
            f"num_reqs={num_reqs} exceeds compute_swa_slot_in_flat_from_cu limit"
        )
    block_b = max(1, triton.next_power_of_2(num_reqs))
    block_m = 256
    _compute_swa_slot_in_flat_from_cu_kernel[(triton.cdiv(num_tokens, block_m),)](
        out,
        cu_seqlens,
        prefix_lengths,
        cu_swa_offsets,
        num_tokens,
        int(M),
        int(base_offset),
        int(window_size),
        num_reqs,
        BLOCK_B=block_b,
        BLOCK_M=block_m,
        HAS_CU_SWA_OFFSETS=has_cu_swa_offsets,
    )
    return out


# ---------------------------------------------------------------------------
# 3) Combine compressed-attn topk + SWA window indices into one row per query
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "combined_indices_stride",
        "topk_indices_stride",
        "num_tokens",
        "num_reqs",
        "M",
        "N",
        "TOP_K",
        "COMBINED_TOPK",
    ]
)
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    cu_cmp_offsets_ptr,
    cu_swa_offsets_ptr,
    num_tokens,
    num_reqs,
    M,
    N,
    TOP_K,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    COMBINED_TOPK,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SINGLE_REQ: tl.constexpr,
    DENSE_ARANGE_TOPK: tl.constexpr,
    HAS_TOPK: tl.constexpr,
    ALIGNED_IN: tl.constexpr,
    ALIGNED_OUT: tl.constexpr,
    HAS_CU_CMP_OFFSETS: tl.constexpr,
    HAS_CU_SWA_OFFSETS: tl.constexpr,
):
    """Tiled vectorized combine of compressed-attn topk + SWA window indices."""
    block_t = tl.program_id(0)
    rows = block_t * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < num_tokens

    base = tl.load(query_start_loc_ptr).to(tl.int32)
    if SINGLE_REQ:
        q_start = tl.zeros((BLOCK_T,), dtype=tl.int32)
        q_end = tl.full(
            (BLOCK_T,),
            tl.load(query_start_loc_ptr + 1).to(tl.int32) - base,
            tl.int32,
        )
        seq_len = tl.full((BLOCK_T,), tl.load(seq_lens_ptr).to(tl.int32), tl.int32)
        gather_len = tl.full(
            (BLOCK_T,), tl.load(gather_lens_ptr).to(tl.int32), tl.int32
        )
        if HAS_CU_CMP_OFFSETS:
            cmp_base = tl.full(
                (BLOCK_T,), tl.load(cu_cmp_offsets_ptr).to(tl.int32), tl.int32
            )
        else:
            cmp_base = tl.zeros((BLOCK_T,), dtype=tl.int32)
        if HAS_CU_SWA_OFFSETS:
            swa_ws_base = tl.full(
                (BLOCK_T,), tl.load(cu_swa_offsets_ptr).to(tl.int32), tl.int32
            )
        else:
            swa_ws_base = tl.full((BLOCK_T,), N, tl.int32)
        in_range = row_mask & (rows < q_end)
    else:
        lo = tl.zeros((BLOCK_T,), dtype=tl.int32)
        hi = tl.full((BLOCK_T,), num_reqs, dtype=tl.int32)
        while tl.max((lo < hi).to(tl.int32), axis=0) != 0:
            active = lo < hi
            mid = (lo + hi) >> 1
            safe_idx = tl.minimum(mid + 1, num_reqs)
            end_val = tl.load(query_start_loc_ptr + safe_idx).to(tl.int32) - base
            go_left = end_val > rows
            hi = tl.where(active & go_left, mid, hi)
            lo = tl.where(active & ~go_left, mid + 1, lo)
        req = lo
        in_range = row_mask & (req < num_reqs)
        safe_req = tl.where(in_range, req, 0)
        q_start = tl.load(query_start_loc_ptr + safe_req).to(tl.int32) - base
        q_end = tl.load(query_start_loc_ptr + safe_req + 1).to(tl.int32) - base
        seq_len = tl.load(seq_lens_ptr + safe_req).to(tl.int32)
        gather_len = tl.load(gather_lens_ptr + safe_req).to(tl.int32)
        if HAS_CU_CMP_OFFSETS:
            cmp_base = tl.load(cu_cmp_offsets_ptr + safe_req).to(tl.int32)
        else:
            cmp_base = safe_req * M
        if HAS_CU_SWA_OFFSETS:
            swa_ws_base = tl.load(cu_swa_offsets_ptr + safe_req).to(tl.int32)
        else:
            swa_ws_base = safe_req * M + N

    query_len = q_end - q_start
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len
    pos = start_pos + (rows - q_start)

    topk_len = tl.where(in_range, tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K), 0)
    swa_len = tl.where(in_range, tl.minimum(pos + 1, WINDOW_SIZE), 0)
    combined_len = topk_len + swa_len
    swa_base = swa_ws_base + pos - gather_start - swa_len + 1 - topk_len

    tl.store(combined_lens_ptr + rows, combined_len.to(tl.int32), mask=in_range)

    max_topk_len = tl.max(topk_len, axis=0)
    max_combined_len = tl.max(combined_len, axis=0)
    row_i64 = rows[:, None].to(tl.int64)
    c_stride = (
        combined_indices_stride.to(tl.int64) * 4
        if ALIGNED_OUT
        else combined_indices_stride.to(tl.int64)
    )
    in_stride = (
        topk_indices_stride.to(tl.int64) * 4
        if ALIGNED_IN
        else topk_indices_stride.to(tl.int64)
    )
    neg_one = tl.full((BLOCK_T, BLOCK_C), -1, dtype=tl.int32)

    for col_base in range(0, COMBINED_TOPK, BLOCK_C):
        if ALIGNED_OUT:
            cols = tl.max_contiguous(
                tl.multiple_of(col_base + tl.arange(0, BLOCK_C), [4]),
                [BLOCK_C],
            )
            col_mask = (cols // 4) < (COMBINED_TOPK // 4)
            out_off = tl.max_contiguous(
                tl.multiple_of(row_i64 * c_stride + cols[None, :], [1, 4]),
                [1, BLOCK_C],
            )
        else:
            cols = col_base + tl.arange(0, BLOCK_C)
            col_mask = cols < COMBINED_TOPK
            out_off = row_i64 * c_stride + cols[None, :]

        store_mask = row_mask[:, None] & col_mask[None, :]
        if col_base >= max_combined_len:
            tl.store(combined_indices_ptr + out_off, neg_one, mask=store_mask)
        else:
            col = cols[None, :]
            in_topk = col < topk_len[:, None]
            in_swa = (col >= topk_len[:, None]) & (col < combined_len[:, None])
            if DENSE_ARANGE_TOPK:
                topk_val = col.to(tl.int32)
                out_val = tl.where(
                    in_topk,
                    cmp_base[:, None] + topk_val,
                    tl.where(in_swa, swa_base[:, None] + col, -1),
                ).to(tl.int32)
            elif HAS_TOPK and col_base < max_topk_len:
                if ALIGNED_IN:
                    in_cols = tl.max_contiguous(
                        tl.multiple_of(col_base + tl.arange(0, BLOCK_C), [4]),
                        [BLOCK_C],
                    )
                    in_off = tl.max_contiguous(
                        tl.multiple_of(row_i64 * in_stride + in_cols[None, :], [1, 4]),
                        [1, BLOCK_C],
                    )
                    topk_load_mask = in_range[:, None] & ((in_cols // 4) < (TOP_K // 4))[None, :]
                else:
                    in_off = row_i64 * in_stride + col
                    topk_load_mask = in_range[:, None] & (cols < TOP_K)[None, :]
                topk_val = tl.load(
                    topk_indices_ptr + in_off,
                    mask=topk_load_mask,
                    other=0,
                ).to(tl.int32)
                out_val = tl.where(
                    in_topk,
                    cmp_base[:, None] + topk_val,
                    tl.where(in_swa, swa_base[:, None] + col, -1),
                ).to(tl.int32)
            else:
                out_val = tl.where(in_swa, swa_base[:, None] + col, -1).to(tl.int32)
            tl.store(combined_indices_ptr + out_off, out_val, mask=store_mask)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,  # [num_tokens, topk] int32 — compressed-attn indices
    query_start_loc: torch.Tensor,  # [num_reqs + 1] int32 — chunk-local start locs
    seq_lens: torch.Tensor,  # [num_reqs] int32 — total seq len per request
    gather_lens: torch.Tensor,  # [num_reqs] int32 — from compute_prefill_gather_lens
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int = 0,
    N: int = 0,
    *,
    flash_mla_indices: bool = False,
    cu_cmp_offsets: Optional[torch.Tensor] = None,
    cu_swa_offsets: Optional[torch.Tensor] = None,
    dense_arange_topk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``(combined_indices, combined_lens)`` for ``flash_mla_sparse_fwd``.

    Returns:
      combined_indices: ``[num_tokens, combined_topk]`` int32; ``combined_topk
        = align(topk + window_size, 128)``. Sentinel ``-1`` in unused tail.
      combined_lens:   ``[num_tokens]`` int32 — pass to ``flash_mla_sparse_fwd``
        as ``topk_length``.

    Args mirror vLLM signature plus optional ragged offset tensors
    ``cu_cmp_offsets`` and ``cu_swa_offsets`` for flat 2D workspaces.
    """
    assert (
        topk_indices.dtype == torch.int32
    ), f"topk_indices must be int32, got {topk_indices.dtype}"
    assert (
        query_start_loc.dtype == torch.int32 and seq_lens.dtype == torch.int32
    ), "query_start_loc and seq_lens must be int32"
    assert gather_lens.dtype == torch.int32, "gather_lens must be int32"
    assert window_size >= 1 and compress_ratio >= 1
    assert dense_arange_topk or int(topk_indices.shape[-1]) >= int(
        topk
    ), f"topk_indices width {topk_indices.shape[-1]} < topk {topk}"

    num_tokens = int(topk_indices.shape[0])
    num_reqs = int(seq_lens.shape[0])
    if cu_cmp_offsets is not None:
        cu_cmp_offsets = cu_cmp_offsets.reshape(-1)
        assert (
            cu_cmp_offsets.dtype == torch.int32
            and cu_cmp_offsets.device == topk_indices.device
            and int(cu_cmp_offsets.numel()) >= num_reqs
        )
        if not cu_cmp_offsets.is_contiguous():
            cu_cmp_offsets = cu_cmp_offsets.contiguous()
    if cu_swa_offsets is not None:
        cu_swa_offsets = cu_swa_offsets.reshape(-1)
        assert (
            cu_swa_offsets.dtype == torch.int32
            and cu_swa_offsets.device == topk_indices.device
            and int(cu_swa_offsets.numel()) >= num_reqs
        )
        if not cu_swa_offsets.is_contiguous():
            cu_swa_offsets = cu_swa_offsets.contiguous()
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices_shape = (
        (num_tokens, 1, combined_topk)
        if flash_mla_indices
        else (num_tokens, combined_topk)
    )
    combined_indices = torch.empty(
        combined_indices_shape,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    if num_tokens == 0 or num_reqs == 0:
        if num_tokens > 0:
            combined_indices.fill_(-1)
        return combined_indices, combined_lens

    BLOCK_T = 2
    BLOCK_C = 256
    out_stride = int(combined_indices.stride(0))
    aligned_out = (
        (combined_topk % 4 == 0)
        and (out_stride % 4 == 0)
        and ((combined_indices.data_ptr() % 16) == 0)
    )
    out_stride_arg = out_stride // 4 if aligned_out else out_stride

    in_stride = int(topk_indices.stride(0))
    aligned_in = (
        (not dense_arange_topk)
        and (topk > 0)
        and (topk % 4 == 0)
        and (in_stride % 4 == 0)
        and ((topk_indices.data_ptr() % 16) == 0)
    )
    in_stride_arg = in_stride // 4 if aligned_in else in_stride

    dummy_off = query_start_loc
    _combine_topk_swa_indices_kernel[(triton.cdiv(num_tokens, BLOCK_T),)](
        combined_indices,
        out_stride_arg,
        combined_lens,
        topk_indices,
        in_stride_arg,
        query_start_loc,
        seq_lens,
        gather_lens,
        cu_cmp_offsets if cu_cmp_offsets is not None else dummy_off,
        cu_swa_offsets if cu_swa_offsets is not None else dummy_off,
        num_tokens,
        num_reqs,
        int(M),
        int(N),
        TOP_K=int(topk),
        COMPRESS_RATIO=int(compress_ratio),
        WINDOW_SIZE=int(window_size),
        COMBINED_TOPK=int(combined_topk),
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
        SINGLE_REQ=(num_reqs == 1),
        DENSE_ARANGE_TOPK=bool(dense_arange_topk),
        HAS_TOPK=(int(topk) > 0),
        ALIGNED_IN=bool(aligned_in),
        ALIGNED_OUT=bool(aligned_out),
        HAS_CU_CMP_OFFSETS=cu_cmp_offsets is not None,
        HAS_CU_SWA_OFFSETS=cu_swa_offsets is not None,
        num_warps=4,
    )
    return combined_indices, combined_lens


@triton.jit(
    do_not_specialize=[
        "combined_indices_stride",
        "topk_indices_stride",
        "num_tokens",
        "sp_int",
        "M",
        "N",
        "TOP_K",
        "COMBINED_TOPK",
    ]
)
def _combine_topk_swa_indices_cp_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    global_positions_ptr,
    req_id_per_token_ptr,
    prefix_lengths_ptr,
    cu_cmp_offsets_ptr,
    cu_swa_offsets_ptr,
    num_tokens,
    sp_int,
    M,
    N,
    TOP_K,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    COMBINED_TOPK,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DENSE_ARANGE_TOPK: tl.constexpr,
    ALIGNED_IN: tl.constexpr,
    ALIGNED_OUT: tl.constexpr,
    HAS_TOPK: tl.constexpr,
    HAS_CU_CMP_OFFSETS: tl.constexpr,
    HAS_CU_SWA_OFFSETS: tl.constexpr,
):
    """CP-aware fused combine (1D row-loop CTA + 128-bit vectorized I/O)."""
    block_t = tl.program_id(0)
    rows = block_t * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < num_tokens

    gp = tl.load(global_positions_ptr + rows, mask=row_mask, other=0).to(tl.int32)
    if IS_VARLEN:
        req = tl.load(req_id_per_token_ptr + rows, mask=row_mask, other=0).to(
            tl.int32
        )
        prefix = tl.load(prefix_lengths_ptr + req, mask=row_mask, other=0).to(
            tl.int32
        )
        if HAS_CU_CMP_OFFSETS:
            cmp_base = tl.load(cu_cmp_offsets_ptr + req, mask=row_mask, other=0).to(
                tl.int32
            )
        else:
            cmp_base = req * M
        if HAS_CU_SWA_OFFSETS:
            swa_ws_base = tl.load(
                cu_swa_offsets_ptr + req, mask=row_mask, other=0
            ).to(tl.int32)
        else:
            swa_ws_base = req * M + N
    else:
        prefix = tl.full((BLOCK_T,), sp_int, tl.int32)
        if HAS_CU_CMP_OFFSETS:
            cmp_base = tl.full(
                (BLOCK_T,), tl.load(cu_cmp_offsets_ptr).to(tl.int32), tl.int32
            )
        else:
            cmp_base = tl.zeros((BLOCK_T,), tl.int32)
        if HAS_CU_SWA_OFFSETS:
            swa_ws_base = tl.full(
                (BLOCK_T,), tl.load(cu_swa_offsets_ptr).to(tl.int32), tl.int32
            )
        else:
            swa_ws_base = tl.full((BLOCK_T,), N, tl.int32)

    p = tl.minimum(prefix, WINDOW_SIZE - 1)
    gather_start = prefix - p
    topk_len = tl.where(row_mask, tl.minimum((gp + 1) // COMPRESS_RATIO, TOP_K), 0)
    swa_len = tl.where(row_mask, tl.minimum(gp + 1, WINDOW_SIZE), 0)
    combined_len = topk_len + swa_len
    swa_base = swa_ws_base + gp - gather_start - swa_len + 1 - topk_len

    tl.store(combined_lens_ptr + rows, combined_len.to(tl.int32), mask=row_mask)

    max_topk_len = tl.max(topk_len, axis=0)
    max_combined_len = tl.max(combined_len, axis=0)
    row_i64 = rows[:, None].to(tl.int64)
    c_stride = (
        combined_indices_stride.to(tl.int64) * 4
        if ALIGNED_OUT
        else combined_indices_stride.to(tl.int64)
    )
    in_stride = (
        topk_indices_stride.to(tl.int64) * 4
        if ALIGNED_IN
        else topk_indices_stride.to(tl.int64)
    )
    neg_one = tl.full((BLOCK_T, BLOCK_C), -1, dtype=tl.int32)

    for col_base in range(0, COMBINED_TOPK, BLOCK_C):
        if ALIGNED_OUT:
            cols = tl.max_contiguous(
                tl.multiple_of(col_base + tl.arange(0, BLOCK_C), [4]),
                [BLOCK_C],
            )
            col_mask = (cols // 4) < (COMBINED_TOPK // 4)
            out_off = tl.max_contiguous(
                tl.multiple_of(row_i64 * c_stride + cols[None, :], [1, 4]),
                [1, BLOCK_C],
            )
        else:
            cols = col_base + tl.arange(0, BLOCK_C)
            col_mask = cols < COMBINED_TOPK
            out_off = row_i64 * c_stride + cols[None, :]

        store_mask = row_mask[:, None] & col_mask[None, :]
        if col_base >= max_combined_len:
            tl.store(combined_indices_ptr + out_off, neg_one, mask=store_mask)
        else:
            col = cols[None, :]
            in_topk = col < topk_len[:, None]
            in_swa = (col >= topk_len[:, None]) & (col < combined_len[:, None])
            if DENSE_ARANGE_TOPK:
                topk_val = col.to(tl.int32)
                out_val = tl.where(
                    in_topk,
                    cmp_base[:, None] + topk_val,
                    tl.where(in_swa, swa_base[:, None] + col, -1),
                ).to(tl.int32)
            elif HAS_TOPK and col_base < max_topk_len:
                if ALIGNED_IN:
                    in_cols = tl.max_contiguous(
                        tl.multiple_of(col_base + tl.arange(0, BLOCK_C), [4]),
                        [BLOCK_C],
                    )
                    in_off = tl.max_contiguous(
                        tl.multiple_of(row_i64 * in_stride + in_cols[None, :], [1, 4]),
                        [1, BLOCK_C],
                    )
                    topk_load_mask = row_mask[:, None] & (
                        (in_cols // 4) < (TOP_K // 4)
                    )[None, :]
                else:
                    in_off = row_i64 * in_stride + col
                    topk_load_mask = row_mask[:, None] & (cols < TOP_K)[None, :]
                topk_val = tl.load(
                    topk_indices_ptr + in_off,
                    mask=topk_load_mask,
                    other=0,
                ).to(tl.int32)
                out_val = tl.where(
                    in_topk,
                    cmp_base[:, None] + topk_val,
                    tl.where(in_swa, swa_base[:, None] + col, -1),
                ).to(tl.int32)
            else:
                out_val = tl.where(in_swa, swa_base[:, None] + col, -1).to(tl.int32)
            tl.store(combined_indices_ptr + out_off, out_val, mask=store_mask)


def combine_topk_swa_indices_cp(
    topk_indices: torch.Tensor,
    global_positions: torch.Tensor,
    sp_int: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int = 0,
    N: int = 0,
    req_id_per_token: torch.Tensor | None = None,
    prefix_lengths: torch.Tensor | None = None,
    *,
    flash_mla_indices: bool = False,
    dense_arange_topk: bool = False,
    cu_cmp_offsets: torch.Tensor | None = None,
    cu_swa_offsets: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CP-aware fused combine for ``flash_mla_sparse_fwd``.

    This is the kernel equivalent of
    ``cp.combine_topk_swa_indices_cp_varlen`` / ``_b1``. It consumes
    explicit rank-local CP global positions, so it works for zigzag CP
    where contiguous query-row math is invalid.
    """
    assert (
        topk_indices.dtype == torch.int32
    ), f"topk_indices must be int32, got {topk_indices.dtype}"
    assert topk_indices.dim() == 2, f"topk_indices must be 2D, got {topk_indices.shape}"
    assert window_size >= 1 and compress_ratio >= 1

    num_tokens = int(global_positions.numel())
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices_shape = (
        (num_tokens, 1, combined_topk)
        if flash_mla_indices
        else (num_tokens, combined_topk)
    )
    combined_indices = torch.empty(
        combined_indices_shape,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    if num_tokens == 0:
        return combined_indices, combined_lens

    assert (
        topk_indices.shape[0] == num_tokens
    ), f"topk rows {topk_indices.shape[0]} != positions {num_tokens}"
    assert int(topk_indices.shape[1]) >= int(
        topk
    ), f"topk_indices width {topk_indices.shape[1]} < topk {topk}"
    # Note: do NOT force ``topk_indices.contiguous()`` here. HCA passes a
    # [T_total, N] broadcast view (stride 0 on dim 0) of arange(N) as a
    # ~9-32 GiB peak-memory optimization at 1M ctx; the kernel reads via
    # ``ptr + row*stride + col`` and a 0-stride broadcast is bit-equal to
    # the materialized version. A force-contiguous here defeats the
    # optimization and reintroduces the alloc.
    global_positions = global_positions.reshape(-1).contiguous()

    assert (req_id_per_token is None) == (prefix_lengths is None), (
        "req_id_per_token and prefix_lengths must be passed together for "
        "CP varlen combine"
    )
    is_varlen = req_id_per_token is not None
    if is_varlen:
        req_id_per_token = req_id_per_token.reshape(-1).contiguous()
        prefix_lengths = prefix_lengths.reshape(-1).contiguous()
        assert (
            req_id_per_token.numel() == num_tokens
        ), f"req_id_per_token rows {req_id_per_token.numel()} != positions {num_tokens}"
        req_ptr = req_id_per_token
        prefix_ptr = prefix_lengths
    else:
        req_ptr = global_positions
        prefix_ptr = global_positions

    min_offsets = int(prefix_lengths.numel()) if is_varlen else 1
    if cu_cmp_offsets is not None:
        cu_cmp_offsets = cu_cmp_offsets.reshape(-1)
        assert (
            cu_cmp_offsets.dtype == torch.int32
            and cu_cmp_offsets.device == topk_indices.device
            and int(cu_cmp_offsets.numel()) >= min_offsets
        )
        if not cu_cmp_offsets.is_contiguous():
            cu_cmp_offsets = cu_cmp_offsets.contiguous()
    if cu_swa_offsets is not None:
        cu_swa_offsets = cu_swa_offsets.reshape(-1)
        assert (
            cu_swa_offsets.dtype == torch.int32
            and cu_swa_offsets.device == topk_indices.device
            and int(cu_swa_offsets.numel()) >= min_offsets
        )
        if not cu_swa_offsets.is_contiguous():
            cu_swa_offsets = cu_swa_offsets.contiguous()

    BLOCK_T = 2
    BLOCK_C = 256
    out_stride = int(combined_indices.stride(0))
    aligned_out = (
        (combined_topk % 4 == 0)
        and (out_stride % 4 == 0)
        and ((combined_indices.data_ptr() % 16) == 0)
        and (BLOCK_C % 4 == 0)
    )
    out_stride_arg = out_stride // 4 if aligned_out else out_stride

    in_stride = int(topk_indices.stride(0))
    aligned_in = (
        (not dense_arange_topk)
        and (topk > 0)
        and (topk % 4 == 0)
        and (in_stride % 4 == 0)
        and ((topk_indices.data_ptr() % 16) == 0)
        and (BLOCK_C % 4 == 0)
    )
    in_stride_arg = in_stride // 4 if aligned_in else in_stride

    grid = (triton.cdiv(num_tokens, BLOCK_T),)
    _combine_topk_swa_indices_cp_kernel[grid](
        combined_indices,
        out_stride_arg,
        combined_lens,
        topk_indices,
        in_stride_arg,
        global_positions,
        req_ptr,
        prefix_ptr,
        cu_cmp_offsets if cu_cmp_offsets is not None else global_positions,
        cu_swa_offsets if cu_swa_offsets is not None else global_positions,
        num_tokens,
        int(sp_int),
        int(M),
        int(N),
        TOP_K=int(topk),
        COMPRESS_RATIO=int(compress_ratio),
        WINDOW_SIZE=int(window_size),
        COMBINED_TOPK=int(combined_topk),
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
        IS_VARLEN=bool(is_varlen),
        DENSE_ARANGE_TOPK=bool(dense_arange_topk),
        ALIGNED_IN=bool(aligned_in),
        ALIGNED_OUT=bool(aligned_out),
        HAS_TOPK=(int(topk) > 0),
        HAS_CU_CMP_OFFSETS=cu_cmp_offsets is not None,
        HAS_CU_SWA_OFFSETS=cu_swa_offsets is not None,
        num_warps=4,
    )
    return combined_indices, combined_lens

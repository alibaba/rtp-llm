"""Fused Triton kernel for DSpARK draft decode attention metadata.

Collapses the two pure-integer torch chains that
:meth:`DeepSeekV4DSparkModel._forward_dspark_attention` runs per draft
layer — ``_global_pool_slots`` (query pool slots) and
``_build_noncausal_indices`` (packed-left topk rows + lengths) — from ~30
ATen launches (graph-captured nodes included) into a single Triton launch
producing all three consumers:

* ``query_slots`` ``[B*gamma]`` int64 — ``block_id * entries_per_block +
  pos % entries_per_block``; ``-1`` sentinels for negative positions,
  block-in-seq overflow and unallocated (``<= 0``) block ids.  Exactly the
  ``_global_pool_slots`` math for request ids ``[0, B)``.
* ``global_indices`` ``[B*gamma, topk]`` int32 — packed-left
  context+query layout of ``_build_noncausal_indices``: the first
  ``min(prefix, window)`` columns map the latest committed window
  (starting at ``prefix - committed``), the next ``gamma`` columns map the
  whole non-causal query block (``prefix .. prefix+gamma``), the tail is
  ``-1``.  Every ``gamma`` query row of one request carries the identical
  row — the deliberate non-causal part of DSpARK.  ``topk`` is the
  FlashMLA 128-alignment of ``window + gamma``.
* ``topk_length`` ``[B]`` int32 — ``min(prefix, window) + gamma`` for
  active requests, ``0`` otherwise.

CUDA-graph safety: every varying value (prefix lengths, active mask,
query positions, block table) is read from device memory; all shape
constants come from host metadata only, so a captured launch replays with
fresh device-side values.  ``num_reqs`` / ``max_blocks_per_seq`` are
``do_not_specialize`` so batch-size and block-table-width changes never
recompile the kernel.

Correctness contract: bit-exact with the torch reference (pure integer
metadata); the unit test asserts ``torch.equal`` against the production
model methods, including the ``-1`` sentinels and the packed-left layout.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


# FlashMLA tile alignment for the physical topk width; mirrors the
# ``((window + gamma + 127) // 128) * 128`` expression in the model.
_TOPK_ALIGNMENT = 128

# Column tile of the index builder; topk is always a multiple of this.
_BLOCK_K = 128


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["num_reqs", "max_blocks_per_seq"])
    def _dspark_decode_meta_kernel(
        # Outputs
        query_slots_ptr,  # [num_reqs * GAMMA] int64
        global_indices_ptr,  # [num_reqs * GAMMA, TOPK] int32
        topk_length_ptr,  # [num_reqs] int32
        # Inputs
        block_table_ptr,  # [num_reqs, max_blocks_per_seq] int32
        positions_ptr,  # [num_reqs * GAMMA] int32/int64, row-major [B, gamma]
        prefix_lengths_ptr,  # [num_reqs] int32/int64
        active_ptr,  # [num_reqs] bool; non-zero = live request
        # Batch-varying runtime args (kept out of specialization)
        num_reqs,
        max_blocks_per_seq,
        # Model-lifetime constants
        GAMMA: tl.constexpr,
        WINDOW: tl.constexpr,
        ENTRIES_PER_BLOCK: tl.constexpr,
        TOKENS_PER_BLOCK: tl.constexpr,
        TOPK: tl.constexpr,
        GAMMA_BLOCK: tl.constexpr,  # power-of-2 tile covering GAMMA
        BLOCK_K: tl.constexpr,
    ):
        """One program per (request, topk column chunk)."""
        req = tl.program_id(0).to(tl.int64)
        chunk = tl.program_id(1)
        if req >= num_reqs:
            return

        prefix = tl.load(prefix_lengths_ptr + req).to(tl.int64)
        # Normalize to 0/1 so both the row masking and the length math can
        # fold it in arithmetically (scalar broadcast is fine, but this
        # keeps every ``tl.where`` condition a plain lane tensor).
        active = (tl.load(active_ptr + req).to(tl.int64) != 0).to(tl.int64)
        committed = tl.minimum(prefix, WINDOW)
        committed_start = prefix - committed

        rows = tl.arange(0, GAMMA_BLOCK)
        row_mask = rows < GAMMA

        if chunk == 0:
            # ---- query-block pool slots + topk lengths (first chunk) ----
            # Request ids are the row indices of the row-major [B, gamma]
            # position stream, so only the position validity rules of
            # ``_global_pool_slots`` apply here.
            n = req * GAMMA + rows
            q_pos = tl.load(positions_ptr + n, mask=row_mask, other=-1).to(tl.int64)
            q_safe_pos = tl.maximum(q_pos, 0)
            q_block_in_seq = q_safe_pos // TOKENS_PER_BLOCK
            q_in_capacity = q_block_in_seq < max_blocks_per_seq
            q_safe_block = tl.where(q_in_capacity, q_block_in_seq, 0)
            q_block_id = tl.load(
                block_table_ptr + req * max_blocks_per_seq + q_safe_block,
                mask=row_mask & q_in_capacity,
                other=-1,
            ).to(tl.int64)
            q_valid = row_mask & (q_pos >= 0) & q_in_capacity & (q_block_id > 0)
            q_slot = tl.where(
                q_valid,
                q_block_id * ENTRIES_PER_BLOCK + (q_safe_pos % ENTRIES_PER_BLOCK),
                tl.full((GAMMA_BLOCK,), -1, dtype=tl.int64),
            )
            tl.store(query_slots_ptr + n, q_slot, mask=row_mask)
            tl.store(
                topk_length_ptr + req,
                ((committed + GAMMA) * active).to(tl.int32),
            )

        # ---- packed-left non-causal context+query indices ----
        cols = (chunk * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
        col_mask = cols < TOPK
        in_context = cols < committed
        in_query = cols < (committed + GAMMA)
        local = tl.where(
            in_context,
            committed_start + cols,
            tl.where(
                in_query,
                prefix + cols - committed,
                tl.full((BLOCK_K,), -1, dtype=tl.int64),
            ),
        )
        # Inactive requests contribute only -1 sentinels (the reference
        # masks the whole row before the pool-slot translation).
        local = local * active - (1 - active)
        safe_local = tl.maximum(local, 0)
        block_in_seq = safe_local // TOKENS_PER_BLOCK
        in_capacity = block_in_seq < max_blocks_per_seq
        safe_block = tl.where(in_capacity, block_in_seq, 0)
        block_id = tl.load(
            block_table_ptr + req * max_blocks_per_seq + safe_block,
            mask=col_mask & in_capacity,
            other=-1,
        ).to(tl.int64)
        valid = col_mask & (local >= 0) & in_capacity & (block_id > 0)
        slot = tl.where(
            valid,
            block_id * ENTRIES_PER_BLOCK + (safe_local % ENTRIES_PER_BLOCK),
            tl.full((BLOCK_K,), -1, dtype=tl.int64),
        )

        # Every gamma query row of this request sees the identical
        # context+query set; broadcast the computed row on store.
        slot32 = slot.to(tl.int32)
        value = slot32[None, :] + tl.zeros((GAMMA_BLOCK, 1), tl.int32)
        out_offset = (req * GAMMA + rows)[:, None] * TOPK + cols[None, :]
        tl.store(
            global_indices_ptr + out_offset,
            value,
            mask=row_mask[:, None] & col_mask[None, :],
        )


def try_compute_dspark_decode_meta(
    block_table: torch.Tensor,  # [B, max_blocks] int32 CUDA contiguous
    positions: torch.Tensor,  # [B, gamma] int64 CUDA contiguous
    prefix_lengths: torch.Tensor,  # [B] int32/int64 CUDA contiguous
    active_requests: torch.Tensor,  # [B] bool CUDA contiguous
    *,
    gamma: int,
    window: int,
    entries_per_block: int,
    tokens_per_block: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fused DSpARK decode metadata builder (single Triton launch).

    Returns ``(query_slots, global_indices, topk_length)`` with the exact
    dtypes/shapes/values of the ``DeepSeekV4DSparkModel`` torch chains
    (int64 ``[B*gamma]``, int32 ``[B*gamma, topk]``, int32 ``[B]``), or
    ``None`` when the inputs do not match the fast-path contract — the
    caller then falls back to the torch reference implementation.

    ``positions`` is the row-major ``[B, gamma]`` query-position stream (a
    flat ``[B*gamma]`` tensor is accepted as well).  ``topk`` follows the
    model's FlashMLA alignment ``((window + gamma + 127) // 128) * 128``.
    """
    if not _TRITON_AVAILABLE:
        return None
    gamma = int(gamma)
    window = int(window)
    entries_per_block = int(entries_per_block)
    tokens_per_block = int(tokens_per_block)
    if gamma <= 0 or window <= 0 or entries_per_block <= 0 or tokens_per_block <= 0:
        return None

    batch_size = int(prefix_lengths.numel())
    if batch_size <= 0 or prefix_lengths.dim() != 1:
        return None
    if active_requests.dim() != 1 or int(active_requests.numel()) != batch_size:
        return None
    if block_table.dim() != 2 or int(block_table.shape[0]) < batch_size:
        return None

    if positions.dim() == 2:
        if tuple(positions.shape) != (batch_size, gamma):
            return None
        if not positions.is_contiguous():
            return None
        flat_positions = positions.view(-1)
    elif positions.dim() == 1:
        if int(positions.numel()) != batch_size * gamma:
            return None
        flat_positions = positions
    else:
        return None

    if (
        not block_table.is_cuda
        or not flat_positions.is_cuda
        or not prefix_lengths.is_cuda
        or not active_requests.is_cuda
    ):
        return None
    device = block_table.device
    if (
        flat_positions.device != device
        or prefix_lengths.device != device
        or active_requests.device != device
    ):
        return None
    if (
        block_table.dtype != torch.int32
        or flat_positions.dtype not in (torch.int32, torch.int64)
        or prefix_lengths.dtype not in (torch.int32, torch.int64)
        or active_requests.dtype
        not in (torch.bool, torch.uint8, torch.int32, torch.int64)
    ):
        return None
    if not (
        block_table.is_contiguous()
        and flat_positions.is_contiguous()
        and prefix_lengths.is_contiguous()
        and active_requests.is_contiguous()
    ):
        return None

    aligned = (window + gamma + _TOPK_ALIGNMENT - 1) // _TOPK_ALIGNMENT
    topk = aligned * _TOPK_ALIGNMENT
    query_slots = torch.empty(batch_size * gamma, dtype=torch.int64, device=device)
    global_indices = torch.empty(
        (batch_size * gamma, topk), dtype=torch.int32, device=device
    )
    topk_length = torch.empty(batch_size, dtype=torch.int32, device=device)

    max_blocks_per_seq = int(block_table.shape[1])
    gamma_block = max(1, triton.next_power_of_2(gamma))
    grid = (batch_size, triton.cdiv(topk, _BLOCK_K))
    _dspark_decode_meta_kernel[grid](
        query_slots,
        global_indices,
        topk_length,
        block_table,
        flat_positions,
        prefix_lengths,
        active_requests,
        batch_size,
        max_blocks_per_seq,
        GAMMA=gamma,
        WINDOW=window,
        ENTRIES_PER_BLOCK=entries_per_block,
        TOKENS_PER_BLOCK=tokens_per_block,
        TOPK=topk,
        GAMMA_BLOCK=gamma_block,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
    )
    return query_slots, global_indices, topk_length

"""Fused Triton kernel for :meth:`CompressorFP8.prepare_metadata`.

Collapses the 3 pure-integer helpers (~25 aten ops total) called per
decode layer into a single Triton kernel launch:

  * ``_compute_state_slot_mapping``: ``state_bt[b, pos // state_eb] *
    state_eb + pos % state_eb`` with ``-1`` sentinel when the logical
    block is absent or the resolved block id is negative.
  * ``_compute_kv_slot_mapping``: ``kv_bt[b, pos // tokens_per_block] *
    kv_eb + (pos % tokens_per_block) // ratio``, masked to ``-1`` unless
    ``(pos+1) % ratio == 0`` and ``block_id >= 0`` and the block-in-seq
    index fits the block table (plus an optional pool-row overflow guard
    when the caller can supply the pool's flat row count).
  * ``token_to_req``: ``b_idx.to(int32)``.

All three are captured into the outer decode CUDA graph today (the
attention layer calls ``compressor.forward_decode_vectorized`` without
a pre-built meta, so ``prepare_metadata`` runs inside the graph).
Fusing them removes ~25 graph nodes per compressor layer × 41+
compressors = ~1000+ nodes / step, which directly shrinks the
``cudaGraphLaunch`` CPU overhead measured at ~6 ms / step in iter7.

Correctness contract: bit-exact with the Python reference across every
combination of (boundary token, off-boundary token, block_in_seq past
``max_blocks``, unallocated negative block id).  ``pool_rows`` guard is
gated by a non-zero runtime arg and matches the upstream 2184f972 fix.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _compressor_slot_mapping_kernel(
        # inputs
        positions_ptr,  # [N] i64
        b_idx_ptr,  # [N] i64
        state_bt_ptr,  # [B, STATE_MAX_BLOCKS] i32
        kv_bt_ptr,  # [B, KV_MAX_BLOCKS] i32 (ignored when HAS_KV=False)
        # outputs
        state_slots_ptr,  # [N] i64
        kv_slots_ptr,  # [N] i64 (written with -1 when HAS_KV=False)
        token_to_req_ptr,  # [N] i32
        # runtime
        N,
        POOL_ROWS,  # <= 0 means skip overflow check
        # constexpr
        STATE_EB: tl.constexpr,
        STATE_MAX_BLOCKS: tl.constexpr,
        HAS_KV: tl.constexpr,
        KV_EB: tl.constexpr,
        KV_MAX_BLOCKS: tl.constexpr,
        RATIO: tl.constexpr,
        TOKENS_PER_BLOCK: tl.constexpr,  # = KV_EB * RATIO (noop when HAS_KV=False)
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        pos = tl.load(positions_ptr + offs, mask=mask, other=0).to(tl.int64)
        b = tl.load(b_idx_ptr + offs, mask=mask, other=0).to(tl.int64)

        # ---------- State slot ----------
        state_bis_raw = pos // STATE_EB
        state_in_capacity = state_bis_raw < STATE_MAX_BLOCKS
        state_bis = tl.maximum(tl.minimum(state_bis_raw, STATE_MAX_BLOCKS - 1), 0)
        in_blk_s = pos % STATE_EB
        state_bid = tl.load(
            state_bt_ptr + b * STATE_MAX_BLOCKS + state_bis, mask=mask, other=0
        ).to(tl.int64)
        state_valid = state_in_capacity & (state_bid >= 0)
        state_slot = tl.where(state_valid, state_bid * STATE_EB + in_blk_s, -1)
        tl.store(state_slots_ptr + offs, state_slot, mask=mask)

        # ---------- KV slot ----------
        if HAS_KV:
            boundary = ((pos + 1) % RATIO) == 0
            kv_bis_raw = pos // TOKENS_PER_BLOCK
            in_blk_k = (pos % TOKENS_PER_BLOCK) // RATIO
            in_capacity = kv_bis_raw < KV_MAX_BLOCKS
            # Clamp for safe gather; correctness relies on the `valid` mask.
            safe_kv_bis = tl.maximum(tl.minimum(kv_bis_raw, KV_MAX_BLOCKS - 1), 0)
            kv_bid = tl.load(
                kv_bt_ptr + b * KV_MAX_BLOCKS + safe_kv_bis, mask=mask, other=0
            ).to(tl.int64)
            kv_slot = kv_bid * KV_EB + in_blk_k
            kv_valid = boundary & in_capacity & (kv_bid >= 0)
            if POOL_ROWS > 0:
                kv_valid = kv_valid & (kv_slot < POOL_ROWS)
            kv_slot = tl.where(kv_valid, kv_slot, -1)
            tl.store(kv_slots_ptr + offs, kv_slot, mask=mask)
        else:
            tl.store(
                kv_slots_ptr + offs,
                tl.full((BLOCK_SIZE,), -1, tl.int64),
                mask=mask,
            )

        # ---------- token_to_req ----------
        tl.store(token_to_req_ptr + offs, b.to(tl.int32), mask=mask)


def fused_compressor_slot_mapping(
    positions: torch.Tensor,  # [N] int64
    b_idx: torch.Tensor,  # [N] int64
    state_bt: torch.Tensor,  # [B, state_max_blocks] int32
    state_eb: int,
    kv_bt: Optional[torch.Tensor],  # [B, kv_max_blocks] int32 or None
    kv_eb: int,
    ratio: int,
    pool_rows: int = 0,  # > 0 to enable overflow guard
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-kernel equivalent of
    :meth:`CompressorFP8._compute_state_slot_mapping` +
    :meth:`CompressorFP8._compute_kv_slot_mapping` + ``b_idx.to(int32)``.

    Returns ``(state_slots, kv_slots, token_to_req)`` on the same device
    as ``positions``.

    Handles the ``kv_bt is None`` / ``kv_eb <= 0`` sentinel case (SWA-only
    layers) by writing ``kv_slots`` as all ``-1``.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton unavailable")

    assert positions.dim() == 1 and b_idx.dim() == 1
    assert positions.shape == b_idx.shape
    assert positions.dtype == torch.int64
    assert b_idx.dtype == torch.int64

    if __debug__:
        # M08 §10.4 pool-row guard tightening: pool_rows MUST be derived
        # from the KV pool view alone (kv_pool.numel() // last_dim) and
        # MUST stay KV-pool-local even when ``unified_bt`` has more
        # columns than the per-pool table. Caller passes 0 to disable the
        # guard; positive value must be sensible (>= state_bt.shape[1]).
        assert pool_rows >= 0, (
            f"fused_compressor_slot_mapping: pool_rows must be >= 0, "
            f"got {pool_rows} (KV-pool-local row count or 0 to skip)"
        )

    N = positions.shape[0]
    device = positions.device

    state_slots = torch.empty(N, dtype=torch.int64, device=device)
    kv_slots = torch.empty(N, dtype=torch.int64, device=device)
    token_to_req = torch.empty(N, dtype=torch.int32, device=device)

    if N == 0:
        return state_slots, kv_slots, token_to_req

    state_max_blocks = int(state_bt.shape[1])

    has_kv = kv_bt is not None and kv_eb > 0
    if has_kv:
        kv_max_blocks = int(kv_bt.shape[1])
        tokens_per_block = kv_eb * ratio
        kv_bt_arg = kv_bt
    else:
        # Passing state_bt as placeholder; kernel won't read it when HAS_KV=False.
        kv_max_blocks = 1
        tokens_per_block = 1
        kv_bt_arg = state_bt

    BLOCK = 128
    grid = ((N + BLOCK - 1) // BLOCK,)
    _compressor_slot_mapping_kernel[grid](
        positions,
        b_idx,
        state_bt,
        kv_bt_arg,
        state_slots,
        kv_slots,
        token_to_req,
        N,
        pool_rows,
        STATE_EB=state_eb,
        STATE_MAX_BLOCKS=state_max_blocks,
        HAS_KV=has_kv,
        KV_EB=max(1, kv_eb),
        KV_MAX_BLOCKS=kv_max_blocks,
        RATIO=max(1, ratio),
        TOKENS_PER_BLOCK=tokens_per_block,
        BLOCK_SIZE=BLOCK,
    )
    return state_slots, kv_slots, token_to_req

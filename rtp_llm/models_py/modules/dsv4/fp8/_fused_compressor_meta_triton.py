"""Fused Triton kernel for :meth:`CompressorFP8.prepare_metadata`.

Collapses the 3 pure-integer helpers (~25 aten ops total) called per
decode layer into a single Triton kernel launch:

  * ``_compute_state_slot_mapping``: ``state_bt[b, pos // state_eb] *
    state_eb + pos % state_eb`` with ``-1`` sentinel when the logical
    block is absent or the resolved block id is negative.
  * ``_compute_kv_slot_mapping``: ``kv_bt[b, pos // tokens_per_block] *
    kv_eb + (pos % tokens_per_block) // ratio``, masked to ``-1`` unless
    ``(pos+1) % ratio == 0`` and ``block_id > 0`` and the block-in-seq
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

    @triton.jit(
        do_not_specialize=[
            "N",
            "POOL_ROWS",
            "STATE_EB",
            "STATE_TOKENS_PER_BLOCK",
            "STATE_MAX_BLOCKS",
            "KV_EB",
            "KV_MAX_BLOCKS",
            "RATIO",
            "TOKENS_PER_BLOCK",
            "CP_SIZE",
            "CP_RANK",
            "KV_OWNER_TOKENS_PER_BLOCK",
        ]
    )
    def _compressor_slot_mapping_kernel(
        # inputs
        positions_ptr,  # [N] i64
        b_idx_ptr,  # [N] i64
        state_bt_ptr,  # [B, STATE_MAX_BLOCKS] i32
        kv_bt_ptr,  # [B, KV_MAX_BLOCKS] i32 (ignored when HAS_KV=False)
        seq_start_ptr,  # [B] i32/i64
        cu_seq_ptr,  # [B+1] i32/i64
        # outputs
        state_slots_ptr,  # [N] i64
        kv_slots_ptr,  # [N] i64 (written with -1 when HAS_KV=False)
        token_to_req_ptr,  # [N] i32
        # runtime
        N,
        POOL_ROWS,  # <= 0 means skip overflow check
        STATE_EB,
        STATE_TOKENS_PER_BLOCK,
        STATE_MAX_BLOCKS,
        HAS_KV: tl.constexpr,
        KV_EB,
        KV_MAX_BLOCKS,
        RATIO,
        TOKENS_PER_BLOCK,
        CP_SHARDED: tl.constexpr,
        CP_SIZE,
        CP_RANK,
        KV_OWNER_TOKENS_PER_BLOCK,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        pos = tl.load(positions_ptr + offs, mask=mask, other=0).to(tl.int64)
        b = tl.load(b_idx_ptr + offs, mask=mask, other=0).to(tl.int64)

        # ---------- State slot ----------
        # State pools are SWA-type (cyclic block table). Under CP sharding the
        # full logical ring is split into one contiguous slice per rank.
        state_bis_raw = pos // STATE_TOKENS_PER_BLOCK
        state_bis = state_bis_raw % STATE_MAX_BLOCKS
        if CP_SHARDED:
            full_state_eb = STATE_EB * CP_SIZE
            logical_offset = pos % full_state_eb
            state_owner = logical_offset // STATE_EB
            in_blk_s = logical_offset - state_owner * STATE_EB
        else:
            full_state_eb = STATE_EB
            state_owner = tl.full((BLOCK_SIZE,), 0, tl.int64)
            in_blk_s = pos % STATE_EB
        state_bid = tl.load(
            state_bt_ptr + b * STATE_MAX_BLOCKS + state_bis, mask=mask, other=0
        ).to(tl.int64)
        state_valid = state_bid > 0
        if CP_SHARDED:
            state_valid = state_valid & (state_owner == CP_RANK)
        seq_start = tl.load(seq_start_ptr + b, mask=mask, other=0).to(tl.int64)
        seq_begin = tl.load(cu_seq_ptr + b, mask=mask, other=0).to(tl.int64)
        seq_finish = tl.load(cu_seq_ptr + b + 1, mask=mask, other=0).to(tl.int64)
        seq_end = seq_start + seq_finish - seq_begin
        block_end = (state_bis_raw + 1) * STATE_TOKENS_PER_BLOCK
        effective_end = tl.minimum(block_end, seq_end)
        state_valid = state_valid & ((pos + full_state_eb) >= effective_end)
        state_slot = tl.where(state_valid, state_bid * STATE_EB + in_blk_s, -1)
        tl.store(state_slots_ptr + offs, state_slot, mask=mask)

        # ---------- KV slot ----------
        if HAS_KV:
            boundary = ((pos + 1) % RATIO) == 0
            if CP_SHARDED:
                owner_block = pos // KV_OWNER_TOKENS_PER_BLOCK
                kv_owner = owner_block % CP_SIZE
                local_owner_block = owner_block // CP_SIZE
                kernel_blocks_per_owner = KV_OWNER_TOKENS_PER_BLOCK // TOKENS_PER_BLOCK
                kernel_in_owner = (pos % KV_OWNER_TOKENS_PER_BLOCK) // TOKENS_PER_BLOCK
                kv_bis_raw = (
                    local_owner_block * kernel_blocks_per_owner + kernel_in_owner
                )
                owned_by_rank = kv_owner == CP_RANK
            else:
                kv_bis_raw = pos // TOKENS_PER_BLOCK
                owned_by_rank = tl.full((BLOCK_SIZE,), 1, tl.int1)
            in_blk_k = (pos % TOKENS_PER_BLOCK) // RATIO
            in_capacity = kv_bis_raw < KV_MAX_BLOCKS
            # Clamp for safe gather; correctness relies on the `valid` mask.
            safe_kv_bis = tl.maximum(tl.minimum(kv_bis_raw, KV_MAX_BLOCKS - 1), 0)
            kv_bid = tl.load(
                kv_bt_ptr + b * KV_MAX_BLOCKS + safe_kv_bis, mask=mask, other=0
            ).to(tl.int64)
            kv_slot = kv_bid * KV_EB + in_blk_k
            if CP_SHARDED:
                kv_valid = boundary & owned_by_rank & in_capacity & (kv_bid > 0)
            else:
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
    seq_start_per_req: torch.Tensor,  # [B] int32/int64
    cu_seq_per_req: torch.Tensor,  # [B+1] int32/int64
    state_tokens_per_block: int,
    pool_rows: int = 0,  # > 0 to enable overflow guard
    *,
    kv_tokens_per_block: int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
    kv_owner_tokens_per_block: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-kernel equivalent of
    :meth:`CompressorFP8._compute_state_slot_mapping` +
    :meth:`CompressorFP8._compute_kv_slot_mapping` + ``b_idx.to(int32)``.

    Returns ``(state_slots, kv_slots, token_to_req)`` on the same device
    as ``positions``.

    Handles the ``kv_bt is None`` / ``kv_eb <= 0`` sentinel case (SWA-only
    layers) by writing ``kv_slots`` as all ``-1``.

    ``state_tokens_per_block``: block_table indexing stride (DSV4 = 256).
    The ring offset uses ``state_eb`` (= R) for ``pos % R``.

    ``seq_start_per_req`` and ``cu_seq_per_req`` let the kernel derive each
    request's sequence end without launching a separate sub/cast/add chain.
    Under CP sharding, fixed STATE rings are byte-sliced across ranks and FULL
    KV blocks use page-RR ownership; the same kernel applies both mappings.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton unavailable")

    assert positions.dim() == 1 and b_idx.dim() == 1
    assert positions.shape == b_idx.shape
    assert positions.dtype == torch.int64
    assert b_idx.dtype == torch.int64

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
        if kv_bt.shape[0] != state_bt.shape[0]:
            raise RuntimeError(
                "fused_compressor_slot_mapping expects state_bt and kv_bt to "
                f"share batch dim, got state_bt={tuple(state_bt.shape)} and "
                f"kv_bt={tuple(kv_bt.shape)}"
            )
        kv_max_blocks = int(kv_bt.shape[1])
        tokens_per_block = int(kv_tokens_per_block or (kv_eb * ratio))
        kv_bt_arg = kv_bt
    else:
        # Passing state_bt as placeholder; kernel won't read it when HAS_KV=False.
        kv_max_blocks = 1
        tokens_per_block = 1
        kv_bt_arg = state_bt

    assert state_tokens_per_block > 0, (
        f"state_tokens_per_block={state_tokens_per_block} must be > 0; "
        "caller must propagate kernel_seq_size_per_block from CacheConfig"
    )
    assert seq_start_per_req is not None and cu_seq_per_req is not None
    assert seq_start_per_req.dim() == 1 and cu_seq_per_req.dim() == 1
    assert cu_seq_per_req.numel() == seq_start_per_req.numel() + 1
    assert 1 <= int(cp_size) and 0 <= int(cp_rank) < int(cp_size)
    cp_sharded = int(cp_size) > 1
    owner_tokens_per_block = int(kv_owner_tokens_per_block or tokens_per_block)
    if cp_sharded and has_kv:
        assert owner_tokens_per_block > 0
        assert owner_tokens_per_block % tokens_per_block == 0

    BLOCK = 128
    grid = ((N + BLOCK - 1) // BLOCK,)
    _compressor_slot_mapping_kernel[grid](
        positions,
        b_idx,
        state_bt,
        kv_bt_arg,
        seq_start_per_req,
        cu_seq_per_req,
        state_slots,
        kv_slots,
        token_to_req,
        N,
        pool_rows,
        state_eb,
        state_tokens_per_block,
        state_max_blocks,
        HAS_KV=has_kv,
        KV_EB=max(1, kv_eb),
        KV_MAX_BLOCKS=kv_max_blocks,
        RATIO=max(1, ratio),
        TOKENS_PER_BLOCK=tokens_per_block,
        CP_SHARDED=cp_sharded,
        CP_SIZE=int(cp_size),
        CP_RANK=int(cp_rank),
        KV_OWNER_TOKENS_PER_BLOCK=owner_tokens_per_block,
        BLOCK_SIZE=BLOCK,
    )
    return state_slots, kv_slots, token_to_req

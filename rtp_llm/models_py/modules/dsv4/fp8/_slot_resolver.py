"""DSV4 FP8 single-source-of-truth slot resolver.

Foundation module for M06 (Python address helpers). Replaces 5 hand-rolled
``block_id → slot`` arithmetic chains scattered across:

  * ``_fused_prepare_meta_triton.py`` (decode fused phase2b)
  * ``compressor.py`` (prefill non-CP / CP)
  * ``_kv_cache_utils.PoolBackedModule._compute_pool_slots`` (prefill bind/scatter)
  * ``decode/decode_attn_metadata._compute_state_pool_slot_mapping`` (state cyclic)
  * ``decode/pool_slot_mapping.compute_kv_pool_slot_mapping`` (decode KV)

Sentinel policy is encoded by :class:`SentinelStrict` — every call site
states the rule it has always used; no behaviour change. The conservative
default is ``GT_ZERO`` (drop rather than corrupt) per M06 §4.2.

Capacity guard semantics: ``block_in_seq < max_blocks`` is a safety net.
The metadata builder (M09) is the authoritative gatekeeper; well-formed
inputs already satisfy capacity. ``in_capacity`` exists so a buggy
upstream produces ``slot=-1`` instead of an OOB pool gather (per C02 47-2).

Zero-shape safe (T06): ``bsz==0`` / ``T==0`` inputs return empty tensors
without raising — required for CUDA-graph capture and empty decode batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch

from rtp_llm.models_py.modules.dsv4.fp8._pool_handle import PoolHandle


class SentinelStrict(Enum):
    """Sentinel threshold for ``block_id`` interpretation.

    Two-valued (semantically a bool); earlier drafts described a
    three-valued enum but only two thresholds exist on real call sites
    (per M06 §4.1, C01 47-3).

    * ``GE_ZERO`` (``block_id < 0`` rejects, block 0 IS valid):
      SWA prefill UT, prefill ``_compute_pool_slots``, compressor non-CP.
    * ``GT_ZERO`` (``block_id <= 0`` rejects):
      decode fused phase2b, compressor CP, decode meta KV / state cyclic,
      Triton ``block_index_to_global``.
    """

    GE_ZERO = 1
    GT_ZERO = 2


@dataclass(frozen=True)
class CPContext:
    """CP topology — sourced exclusively from ``CPShardConfig`` per M06 §3.4.

    NEVER recompute from ``torch.distributed.get_rank()`` at slot-resolve
    time. NEVER derive from ``PyKVCacheRegionDesc.cp_sharded`` (that bit
    is a STATIC layout flag; topology is a per-step concern).
    """

    size: int
    rank: int


@dataclass(frozen=True)
class SlotMapping:
    """Per-block path output (returned by :func:`build_slot_mapping`).

    * ``valid`` ``[B, T]`` bool — True iff (boundary & in_capacity & owned
      & bid_ok & valid_mask) for that ``(req, t)`` cell.
    * ``safe_slot`` ``[B, T]`` int64 — valid rows hold flat pool slot;
      masked rows hold 0 (safe for ``index_select``).
    * ``slot_neg1`` ``[B*T]`` int64 — masked rows hold ``-1``; suitable
      for ``write_kv_to_pool(..., mask_negative=True)``.
    """

    valid: torch.Tensor
    safe_slot: torch.Tensor
    slot_neg1: torch.Tensor


def resolve_pool_slot(
    handle: PoolHandle,
    abs_pos: torch.Tensor,
    super_block_id_table: torch.Tensor,
    *,
    req_idx: Optional[torch.Tensor] = None,
    cp: Optional[CPContext] = None,
    sentinel_strict: SentinelStrict = SentinelStrict.GT_ZERO,
    cyclic: bool = False,
    overflow_guard: bool = False,
    pool_rows: Optional[int] = None,
    valid_mask: Optional[torch.Tensor] = None,
    eff_pos_already_resolved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(slot[N] int64, in_block[N] int64)``. ``slot == -1`` ⇒ skip.

    Pure tensor; CUDA-graph safe; no D2H, no Python-level branches inside
    the captured region (the ``cp.size > 1`` selector resolves at graph-
    build time — see M06 §9.6 / C01 47-7).

    Covers all 5 paths today:
      * P1 decode fused phase2b (SWA/CSA/IDX/HCA)  ``GT_ZERO``, ``cyclic=False``, ``cp=None``
      * P2 compressor non-CP KV (prefill)          ``GE_ZERO``, ``cp=None``, ``overflow_guard=True``
      * P3 compressor CP KV (prefill)              ``GT_ZERO``, ``cp=...``
      * P4 decode-meta state cyclic                ``GT_ZERO``, ``cyclic=True``, ``handle.is_state=True``
      * P5 SWA prefill UT (block 0 valid)          ``GE_ZERO``
    """
    # ---- Step 0. Zero-shape early-out (T06; per M06 §8 / C02 33-6) ----------
    if abs_pos.numel() == 0:
        empty = abs_pos.new_empty(0, dtype=torch.long)
        return empty, empty

    pos = abs_pos.to(torch.long)
    bt = super_block_id_table.to(torch.long)
    B, max_blocks = bt.shape
    E = int(handle.eb)
    R = int(handle.ratio)

    # tokens-per-block — state pools are 1 entry per token (R=1, TPB=E).
    TPB = E if handle.is_state else E * R

    # Refuse to operate on a degenerate STATE handle in cyclic mode that
    # lacks the canonical modulus (Path B handle). Surfaces the contract
    # violation at the resolver boundary (Panel C CD-4).
    if cyclic:
        assert handle.is_state and handle.max_state_blocks is not None, (
            f"resolve_pool_slot: cyclic=True requires a STATE pool with "
            f"handle.max_state_blocks set (got is_state={handle.is_state}, "
            f"max_state_blocks={handle.max_state_blocks})"
        )
        assert max_blocks == handle.max_state_blocks, (
            f"resolve_pool_slot: STATE-pool cyclic-modulus contract "
            f"violation — super_block_id_table.shape[1]={max_blocks} != "
            f"handle.max_state_blocks={handle.max_state_blocks}. M09 owns "
            f"the per-pool narrowing of "
            f"unified_block_table[:, :max_state_blocks] before invocation "
            f"(Panel C CD-4)."
        )

    # State pools must NOT enter the CP branch (per M06 §3.5; HybridAllocator
    # cpShardThisGroup pins this on the C++ side too).
    if handle.is_state and cp is not None and cp.size > 1:
        raise AssertionError(
            "resolve_pool_slot: STATE pools are not CP-sharded; refusing "
            "to apply CP ownership masking (M06 §3.5)."
        )

    # ---- Step 1. Derive request index --------------------------------------
    if req_idx is None:
        N = pos.shape[0]
        assert (
            N % B == 0
        ), f"resolve_pool_slot: N={N} not divisible by batch B={B}"
        q_len = N // B if B > 0 else 0
        if B == 0:
            req_idx = pos.new_empty(0)
        else:
            req_idx = (
                torch.arange(B, device=pos.device, dtype=torch.long)
                .view(B, 1)
                .expand(B, q_len)
                .reshape(-1)
            )
    else:
        req_idx = req_idx.to(torch.long)

    # ---- Step 2. Entry-space projection ------------------------------------
    if eff_pos_already_resolved or R == 1:
        boundary = torch.ones_like(pos, dtype=torch.bool)
        eff_pos = pos
    else:
        boundary = ((pos + 1) % R) == 0
        # cmpR(pos) = (pos + 1) // R - 1
        eff_pos = torch.where(
            boundary, (pos + 1) // R - 1, torch.zeros_like(pos)
        )

    # ---- Step 3. Super-block-table addressing ------------------------------
    if cp is not None and cp.size > 1 and not handle.is_state:
        # CP path: block_table_local[b, l] = super_block_id of global
        # logical block (cp_rank + l*cp_size).
        # CRITICAL: g_blk uses raw token pos, NOT eff_pos (M06 §9.5).
        g_blk = pos // TPB
        owned = (g_blk % cp.size) == cp.rank
        local_blk = g_blk // cp.size
        in_capacity = local_blk < max_blocks
        bis_safe = local_blk.clamp(0, max(max_blocks - 1, 0))
        in_block = eff_pos % E
    else:
        owned = torch.ones_like(pos, dtype=torch.bool)
        block_in_seq = eff_pos // E
        if cyclic:
            modulus = int(handle.max_state_blocks)
            # ``block_in_seq`` is wrapped for the gather; ``in_block`` is
            # the natural offset within the block (``eff_pos % E``) — the
            # cyclic wrap does NOT change the per-block byte offset.
            block_in_seq_cyc = block_in_seq % modulus
            in_capacity = block_in_seq_cyc < modulus
            bis_safe = block_in_seq_cyc.clamp(0, max(modulus - 1, 0))
            in_block = eff_pos - block_in_seq * E
        else:
            in_capacity = block_in_seq < max_blocks
            bis_safe = block_in_seq.clamp(0, max(max_blocks - 1, 0))
            in_block = eff_pos - block_in_seq * E

    # ---- Step 4. Gather super-block id (bps≡1 ⇒ super_block_id == pool_block_id)
    sblk_id = bt[req_idx, bis_safe]
    block_id = sblk_id * int(handle.bps) if handle.bps != 1 else sblk_id

    # ---- Step 5. Validity composition --------------------------------------
    if sentinel_strict == SentinelStrict.GT_ZERO:
        bid_ok = block_id > 0
    else:  # GE_ZERO
        bid_ok = block_id >= 0

    valid = boundary & in_capacity & owned & bid_ok
    if valid_mask is not None:
        valid = valid & valid_mask.to(torch.bool)

    slot = block_id * E + in_block
    if overflow_guard and pool_rows is not None and pool_rows > 0:
        valid = valid & (slot < int(pool_rows))

    slot = torch.where(valid, slot, slot.new_full(slot.shape, -1))
    return slot, in_block


def build_slot_mapping(
    handle: PoolHandle,
    super_block_id_table: torch.Tensor,
    bsz: int,
    T: int,
    device: torch.device,
    *,
    cp: Optional[CPContext] = None,
    cyclic: bool = False,
    sentinel_strict: SentinelStrict = SentinelStrict.GE_ZERO,
    pool_rows: Optional[int] = None,
    overflow_guard: bool = False,
) -> SlotMapping:
    """Per-block path: drives one ``pos = arange(T)`` per request.

    Returns a :class:`SlotMapping` directly consumable by
    ``pool_view.index_select`` (reads) or
    ``write_kv_to_pool(..., mask_negative=True)`` (writes).

    Default sentinel is ``GE_ZERO`` to preserve bit-equality with today's
    ``_compute_pool_slots`` (which treats block 0 as a valid warmup
    allocation, per ``_kv_cache_utils.py:147``).

    Zero-shape safe (T06): if ``bsz==0`` or ``T==0`` returns empty
    tensors with the expected shape ``(bsz, T)`` / ``(bsz*T,)``.
    """
    if bsz == 0 or T == 0:
        valid = torch.zeros((bsz, T), dtype=torch.bool, device=device)
        safe_slot = torch.zeros((bsz, T), dtype=torch.long, device=device)
        slot_neg1 = torch.zeros((bsz * T,), dtype=torch.long, device=device)
        return SlotMapping(valid=valid, safe_slot=safe_slot, slot_neg1=slot_neg1)

    pos = torch.arange(T, device=device, dtype=torch.long)
    pos_full = pos.unsqueeze(0).expand(bsz, -1).reshape(-1)
    req_idx = (
        torch.arange(bsz, device=device, dtype=torch.long)
        .view(bsz, 1)
        .expand(bsz, T)
        .reshape(-1)
    )

    slot, _ = resolve_pool_slot(
        handle,
        pos_full,
        super_block_id_table[:bsz],
        req_idx=req_idx,
        cp=cp,
        sentinel_strict=sentinel_strict,
        cyclic=cyclic,
        overflow_guard=overflow_guard,
        pool_rows=pool_rows,
    )
    valid = slot >= 0
    safe_slot = torch.where(valid, slot, torch.zeros_like(slot))
    return SlotMapping(
        valid=valid.view(bsz, T),
        safe_slot=safe_slot.view(bsz, T),
        slot_neg1=slot,
    )


def gathered_or_mask(
    values: torch.Tensor,
    valid: torch.Tensor,
    fill=0,
) -> torch.Tensor:
    """Re-mask ``values`` after ``index_select`` on a clamped slot tensor.

    ``pool_view.index_select(0, safe_slot)`` returns garbage for masked
    rows (because ``safe_slot`` was clamped to 0). Callers MUST re-mask
    before use; this helper makes the contract explicit and grep-able
    (per M06 §3.1.1; replaces inlined ``torch.where(valid[..., None], ...)``).
    """
    if valid.dim() < values.dim():
        valid = valid.view(*valid.shape, *([1] * (values.dim() - valid.dim())))
    return torch.where(valid, values, values.new_full((), fill))

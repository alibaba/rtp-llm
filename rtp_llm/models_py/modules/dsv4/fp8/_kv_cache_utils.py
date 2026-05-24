"""FP8-only DSV4 KV-cache utilities.

Lives entirely under ``dsv4/fp8/``; do NOT import from BF16
``dsv4/kv_cache_utils.py`` for these symbols (target branch BF16 path
doesn't need them).  ``build_block_tables`` / ``build_block_tables_batched``
are still imported from the BF16 module — they're shared.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.fp8._pool_handle import (
    PoolHandle,
    _adhoc_kv_handle,
    _adhoc_state_handle,
    make_pool_handle,
)
from rtp_llm.models_py.modules.dsv4.fp8._slot_resolver import (
    SentinelStrict,
    build_slot_mapping,
)

# DSV4 per-layer gid slot count. Must match the alloc size used in
# ``NormalModelInputGatherer`` for ``kv_cache_layer_to_group_dpsk_v4``.
# A CSA layer touches 5 groups (SWA_KV, CSA_KV, INDEXER_KV, INDEXER_STATE,
# CSA_STATE); HCA touches 3; SWA-only touches 1. Unused slots hold -1.
_DSV4_MAX_GROUPS_PER_LAYER = 5

# FP8 KV slot byte layouts. See cpp/cache/DSV4CacheConfig.h for the
# canonical 584B SWA/CSA/HCA fp8_model1_mla layout (448 fp8 NoPE + 64 bf16
# RoPE + 7 UE8M0 scales + 1 pad). The vLLM-vendored fused writer in
# _compressor_vllm_triton.py emits it; _swa_dequant_triton.py reads it.
# Indexer 132B layout (128 fp8 + 4-byte fp32 scale) shares the same writer
# (head_dim=128 dispatch).
DSV4_FP8_SLOT_BYTES = 584  # SWA / CSA / HCA canonical fp8_model1_mla
DSV4_FP8_INDEXER_BYTES = 132  # Indexer: fp8[128] + fp32 scale


def is_fp8_swa_slot_pool(pool_view: Optional[torch.Tensor]) -> bool:
    """True iff pool is the 584B uint8 FP8 slot layout (SWA/CSA/HCA)."""
    return (
        pool_view is not None
        and pool_view.dtype == torch.uint8
        and pool_view.shape[-1] == DSV4_FP8_SLOT_BYTES
    )


def is_fp8_indexer_pool(pool_view: Optional[torch.Tensor]) -> bool:
    """True iff pool is the 132B uint8 FP8 indexer layout."""
    return (
        pool_view is not None
        and pool_view.dtype == torch.uint8
        and pool_view.shape[-1] == DSV4_FP8_INDEXER_BYTES
    )


# Canonical REGION_COUNT (mirror C++ ``CacheGroupType.h:29``); pinned by
# M06 §2.4. Replaces the brittle ``_DSV4_MAX_GROUPS_PER_LAYER`` constant
# in the descriptor-driven branch.
_REGION_COUNT = 8


def _gid_for_legacy(
    kv_cache: Any, attn_inputs: Any, layer_id: int, attn_type: int
) -> int:
    """Legacy 5-slot walk preserved for warmup / pre-M05 paths.

    Scheduled for removal in M06 Step 7 once descriptor surface is
    universal (per M06 §6).
    """
    tensor = getattr(attn_inputs, "kv_cache_layer_to_group_dpsk_v4", None)
    if tensor is None or tensor.numel() == 0:
        return -1
    group_region_names = getattr(kv_cache, "group_region_names", None)
    if not group_region_names:
        return -1
    base = layer_id * _DSV4_MAX_GROUPS_PER_LAYER
    for slot in range(_DSV4_MAX_GROUPS_PER_LAYER):
        gid = int(tensor[base + slot].item())
        if gid < 0:
            continue
        if gid < len(group_region_names) and int(group_region_names[gid]) == attn_type:
            return gid
    return -1


def gid_for(kv_cache: Any, attn_inputs: Any, layer_id: int, attn_type: int) -> int:
    """Resolve ``(layer_id, attn_type) → group_id``.

    Fast path (M05 / F02 unified, per M06 §2.4): O(1) lookup into the
    ``[num_layers * REGION_COUNT]`` int32 descriptor table published as
    ``PyAttentionInputs.kv_cache_layer_region_descs``. ``-1`` cells mark
    "layer doesn't own this region".

    Slow path (legacy): falls through to ``_gid_for_legacy`` which walks
    the 5-slot dense list ``kv_cache_layer_to_group_dpsk_v4`` populated
    by ``NormalModelInputGatherer``. This branch survives only until M05
    PR-2 producer wiring lands.
    """
    descs = getattr(attn_inputs, "kv_cache_layer_region_descs", None)
    if descs is not None and descs.numel() > 0:
        idx = layer_id * _REGION_COUNT + int(attn_type)
        if 0 <= idx < descs.numel():
            return int(descs[idx].item())
        return -1
    return _gid_for_legacy(kv_cache, attn_inputs, layer_id, attn_type)


class PoolBackedModule(nn.Module):
    """Base class for modules that bind/scatter state and KV cache from
    framework-managed paged pools (CompressorFP8, IndexerFP8).

    Subclasses must set ``_state_rows``, ``_state_dim``, ``_kv_cache_t``,
    and ``_kv_cache_d`` before calling the bind helpers.
    """

    def __init__(self) -> None:
        super().__init__()
        self._kv_pool_view: Optional[torch.Tensor] = None
        self._kv_block_table: Optional[torch.Tensor] = None
        self._kv_eb: int = 0
        self._state_pool_view: Optional[torch.Tensor] = None
        self._state_block_table: Optional[torch.Tensor] = None
        self._state_eb: int = 0
        # M06: optional PoolHandles surfaced via :meth:`set_pool_handle`.
        # None on the legacy / pre-M05 path; new consumers may use them
        # for TMA-pad / cyclic-modulus / scale-stride introspection.
        self._kv_pool_handle: Optional[PoolHandle] = None
        self._state_pool_handle: Optional[PoolHandle] = None
        # M08 §4.2: unified [B, max_super_blocks] int32 block table.
        # None preserves the legacy two-table contract (bit-equal); when
        # set, KV-side consumers MAY pre-slice from this single source.
        self._unified_bt: Optional[torch.Tensor] = None

        self.kv_state: Optional[torch.Tensor] = None
        self.score_state: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None

        self._state_rows: int = 0
        self._state_dim: int = 0
        self._kv_cache_t: int = 0
        self._kv_cache_d: int = 0

    def set_pool_context(
        self,
        kv_pool_view: Optional[torch.Tensor],
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
        state_pool_view: Optional[torch.Tensor],
        state_block_table: Optional[torch.Tensor],
        state_eb: int,
        *,
        unified_bt: Optional[torch.Tensor] = None,
    ) -> None:
        """Bind framework pool views + block tables.

        M08 §4.2: optional ``unified_bt`` kwarg threads the unified
        [B, max_super_blocks] int32 block table into the module. Under bps=1
        + non-CP it is alias-equal to ``kv_block_table`` (and to
        ``state_block_table`` when both pools exist); under CP page-RR
        ``unified_bt`` aliases the KV side only (M08 §4.2 CP-conditional
        rule). Default ``None`` preserves the bit-equal legacy two-table
        contract — every consumer falls back to ``kv_block_table`` /
        ``state_block_table``.
        """
        self._kv_pool_view = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._state_pool_view = state_pool_view
        self._state_block_table = state_block_table
        self._state_eb = state_eb
        self._unified_bt = unified_bt

    def clear_pool_context(self) -> None:
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0
        self._state_pool_view = None
        self._state_block_table = None
        self._state_eb = 0
        self._kv_pool_handle = None
        self._state_pool_handle = None
        self._unified_bt = None

    def set_pool_handle(
        self,
        kv_handle: Optional[PoolHandle],
        kv_block_table: Optional[torch.Tensor],
        state_handle: Optional[PoolHandle],
        state_block_table: Optional[torch.Tensor],
    ) -> None:
        """M06 §3.1.3 additive migration: bind one PoolHandle per region.

        Wraps :meth:`set_pool_context` so downstream consumers (compressor,
        indexer) continue receiving the legacy tuple form (kv_view,
        kv_bt, kv_eb, state_view, state_bt, state_eb). New consumers may
        read ``self._kv_pool_handle`` / ``self._state_pool_handle``
        directly (None on Path B / legacy fallback).

        Bit-equal to ``set_pool_context`` when handles are derived from
        the same underlying tensors (eb is read from ``handle.eb``).
        """
        kv_view = (
            (kv_handle.base_3d if kv_handle.base_3d is not None else kv_handle.base_2d)
            if kv_handle is not None
            else None
        )
        kv_eb = int(kv_handle.eb) if kv_handle is not None else 0
        state_view = (
            (
                state_handle.base_3d
                if state_handle.base_3d is not None
                else state_handle.base_2d
            )
            if state_handle is not None
            else None
        )
        state_eb = int(state_handle.eb) if state_handle is not None else 0
        self.set_pool_context(
            kv_view,
            kv_block_table,
            kv_eb,
            state_view,
            state_block_table,
            state_eb,
        )
        # Attach handles for new consumers that need full descriptor info
        # (TMA pad, cyclic ring depth, scale stride, ...).
        self._kv_pool_handle = kv_handle
        self._state_pool_handle = state_handle

    def _compute_pool_slots(
        self,
        bsz: int,
        T: int,
        block_table: torch.Tensor,
        eb: int,
        device: torch.device,
        handle: Optional[PoolHandle] = None,
    ) -> tuple:
        """Forward to :func:`build_slot_mapping`.

        Preserves the historical ``(valid, safe_slot)`` tuple contract
        (4 call sites destructure: lines 174, 197, 244, 266) and the
        ``GE_ZERO`` sentinel (block 0 valid for warmup). Pre-existing
        callers that only know ``eb`` pass ``handle=None``; a degenerate
        STATE / KV ad-hoc handle is synthesised so resolver arithmetic
        is bit-equal to the historical inline body.
        """
        if handle is None:
            handle = _adhoc_state_handle(eb=eb)
        sm = build_slot_mapping(
            handle,
            block_table,
            bsz,
            T,
            device,
            sentinel_strict=SentinelStrict.GE_ZERO,
        )
        return sm.valid, sm.safe_slot

    def _bind_state_from_pool(
        self, bsz: int, is_fresh_prefill: bool, device: torch.device
    ) -> None:
        T = self._state_rows
        half_dim = self._state_dim
        if (
            is_fresh_prefill
            or self._state_pool_view is None
            or self._state_block_table is None
            or self._state_eb <= 0
        ):
            self.kv_state = torch.zeros(
                bsz, T, half_dim, dtype=torch.float32, device=device
            )
            self.score_state = torch.full(
                (bsz, T, half_dim),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )
            return
        valid, safe_slot = self._compute_pool_slots(
            bsz, T, self._state_block_table, self._state_eb, device
        )
        gathered = self._state_pool_view.index_select(0, safe_slot.reshape(-1))
        valid_bcast = valid.reshape(-1).unsqueeze(-1)
        zero_row_kv = torch.zeros((), dtype=torch.float32, device=device)
        neg_inf_row = torch.full((), float("-inf"), dtype=torch.float32, device=device)
        kv_rows = torch.where(valid_bcast, gathered[:, :half_dim], zero_row_kv)
        sc_rows = torch.where(valid_bcast, gathered[:, half_dim:], neg_inf_row)
        self.kv_state = kv_rows.view(bsz, T, half_dim).contiguous()
        self.score_state = sc_rows.view(bsz, T, half_dim).contiguous()

    def _scatter_state_to_pool(self, bsz: int) -> None:
        if (
            self._state_pool_view is None
            or self._state_block_table is None
            or self._state_eb <= 0
            or self.kv_state is None
            or self.score_state is None
        ):
            return
        device = self.kv_state.device
        T = self._state_rows
        valid, safe_slot = self._compute_pool_slots(
            bsz, T, self._state_block_table, self._state_eb, device
        )
        merged = torch.cat([self.kv_state[:bsz], self.score_state[:bsz]], dim=-1)
        merged_flat = merged.reshape(bsz * T, -1)
        slot_mapping = torch.where(
            valid, safe_slot, torch.full_like(safe_slot, -1)
        ).reshape(-1)
        from rtp_llm.models_py.modules.dsv4.fp8.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        write_kv_to_pool(
            merged_flat, slot_mapping, self._state_pool_view, mask_negative=True
        )

    def _bind_kv_cache_from_pool(
        self,
        bsz: int,
        is_fresh_prefill: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self._kv_pool_view is not None and self._kv_eb > 0:
            T = (
                self._kv_cache_t
                if self._kv_cache_t > 0
                else (
                    self._kv_block_table.shape[1] * self._kv_eb
                    if self._kv_block_table is not None
                    else 0
                )
            )
        else:
            T = self._kv_cache_t
        if T <= 0:
            self.kv_cache = None
            return
        D = self._kv_cache_d
        if (
            is_fresh_prefill
            or self._kv_pool_view is None
            or self._kv_block_table is None
            or self._kv_eb <= 0
        ):
            self.kv_cache = torch.zeros(bsz, T, D, dtype=dtype, device=device)
            return
        valid, safe_slot = self._compute_pool_slots(
            bsz, T, self._kv_block_table, self._kv_eb, device
        )
        gathered = self._kv_pool_view.index_select(0, safe_slot.reshape(-1))
        if gathered.dtype != dtype:
            gathered = gathered.to(dtype)
        zero_row = torch.zeros((), dtype=dtype, device=device)
        out_flat = torch.where(valid.reshape(-1).unsqueeze(-1), gathered, zero_row)
        self.kv_cache = out_flat.view(bsz, T, D).contiguous()

    def _scatter_kv_cache_to_pool(
        self, bsz: int, block_mask: Optional[torch.Tensor] = None
    ) -> None:
        if (
            self._kv_pool_view is None
            or self._kv_block_table is None
            or self._kv_eb <= 0
            or self.kv_cache is None
        ):
            return
        device = self.kv_cache.device
        T = int(self.kv_cache.shape[1])
        valid, safe_slot = self._compute_pool_slots(
            bsz, T, self._kv_block_table, self._kv_eb, device
        )
        if block_mask is not None:
            valid = valid & block_mask[:bsz].to(device)
        slot_mapping = torch.where(
            valid, safe_slot, torch.full_like(safe_slot, -1)
        ).reshape(-1)
        D = int(self.kv_cache.shape[2])
        flat = self.kv_cache[:bsz].reshape(bsz * T, D)
        from rtp_llm.models_py.modules.dsv4.fp8.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        # No bf16 → FP8 packing here. CSA/HCA prefill/decode go through
        # the vLLM-vendored fused writer in _compressor_vllm_triton.py
        # (canonical 584B); SWA-only writes go through
        # _prefill_write_swa_fp8_paged. This generic scatter only handles
        # same-dtype pools (bf16 KV, fp32 STATE).
        assert not is_fp8_swa_slot_pool(self._kv_pool_view) or D != 512, (
            "FP8 584B KV pool reached generic _scatter_kv_cache_to_pool — "
            "writes must go through the fused canonical path "
            "(_compressor_vllm_triton / _prefill_write_swa_fp8_paged)."
        )
        write_kv_to_pool(flat, slot_mapping, self._kv_pool_view, mask_negative=True)

"""DSV4 KV-cache lookup utilities.

Generic ``(layer_id, attn_type) → group_id`` and ``(layer, attn_type) →
block_table`` helpers shared between prefill and decode. Kept separate
from :mod:`attn_type` (pure int constants, no torch) and from path-
specific forward helpers in :mod:`prefill.forward` / :mod:`decode.forward`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# DSV4 per-layer gid slot count. Must match the alloc size used in
# ``NormalModelInputGatherer`` for ``kv_cache_layer_to_group_dpsk_v4``.
# A CSA layer touches 5 groups (SWA_KV, CSA_KV, INDEXER_KV, INDEXER_STATE,
# CSA_STATE); HCA touches 3; SWA-only touches 1. Unused slots hold -1.
_DSV4_MAX_GROUPS_PER_LAYER = 5

# FP8 KV slot byte layouts. See cpp/cache/DSV4CacheConfig.h:78-91 for the
# canonical 584B SWA/CSA/HCA fp8_model1_mla layout (448 fp8 NoPE + 64 bf16
# RoPE + 7 UE8M0 scales + 1 pad). _compressor_kv_fused_triton.py writes
# it; _kv_fp8_pool_io.py inverts it. _compressor_fused_triton.py
# handles the indexer 132B layout.
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


def gid_for(kv_cache: Any, attn_inputs: Any, layer_id: int, attn_type: int) -> int:
    """Resolve ``(layer_id, attn_type) → group_id`` via the DSV4 dense gid list.

    Reads ``attn_inputs.kv_cache_layer_to_group_dpsk_v4`` — a flat int32 tensor
    of length ``num_layers * 5`` populated by ``NormalModelInputGatherer`` from
    ``CacheConfig::layer_to_group_ids``. Each row holds up to 5 gids the layer
    participates in (order from C++), padded with -1. Walks the row and returns
    the gid whose ``group_region_names[gid]`` matches ``attn_type``; -1 otherwise
    (tensor undefined on warmup / non-DSV4 / this attn_type inactive at layer).
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


def build_block_tables(
    kv_cache: Optional[Any],
    attn_inputs: Any,
    batch_offset: int = 0,
) -> Optional[Dict[int, torch.Tensor]]:
    """Build the per-attn_type block-table dict for one prefill request.

    The framework emits per-request block tables as a list indexed by
    ``group_id`` (``attn_inputs.kv_cache_kernel_block_id_device_by_group``,
    one entry per pool group in the order declared by
    ``DSV4ConfigCreator.cc::pool_attn_types``). This helper joins that list
    against ``kv_cache.group_region_names`` to produce a dict keyed by attn_type
    (the abstraction model code wants — it holds attn_type, not gid).

    The ``batch_offset`` arg slices out a single-request row
    ``[batch_offset : batch_offset + 1]`` so the returned block table is
    per-request, matching how ``DeepSeekV4Model.forward`` unrolls batched
    prefill into one-request-at-a-time layer calls.

    Returns ``None`` when no block tables are available (warmup / paged-KV
    disabled / missing framework state).
    """
    if kv_cache is None or attn_inputs is None:
        return None
    by_group = getattr(attn_inputs, "kv_cache_kernel_block_id_device_by_group", None)
    if by_group is None or len(by_group) == 0:
        return None
    group_region_names = getattr(kv_cache, "group_region_names", None)
    if not group_region_names:
        return None
    block_tables_by_type: Dict[int, torch.Tensor] = {}
    for group_id, attn_type_enum in enumerate(group_region_names):
        if group_id >= len(by_group):
            continue
        group_block_table = by_group[group_id]
        if group_block_table is None or group_block_table.numel() == 0:
            continue
        block_tables_by_type[int(attn_type_enum)] = group_block_table[
            batch_offset : batch_offset + 1
        ]
    return block_tables_by_type or None


def build_block_tables_batched(
    kv_cache: Optional[Any],
    attn_inputs: Any,
) -> Optional[Dict[int, torch.Tensor]]:
    """Build the per-attn_type block-table dict for an entire prefill batch.

    Same semantics as :func:`build_block_tables` but returns the full
    ``[B, max_blocks]`` block table per attn_type (no ``batch_offset`` slice).
    Used by the batched ``forward_prefill`` main path so a single ``v4()`` call
    can cover the whole batch.

    Returns ``None`` when no block tables are available (warmup / paged-KV
    disabled / missing framework state).
    """
    if kv_cache is None or attn_inputs is None:
        return None
    by_group = getattr(attn_inputs, "kv_cache_kernel_block_id_device_by_group", None)
    if by_group is None or len(by_group) == 0:
        return None
    group_region_names = getattr(kv_cache, "group_region_names", None)
    if not group_region_names:
        return None
    block_tables_by_type: Dict[int, torch.Tensor] = {}
    for group_id, attn_type_enum in enumerate(group_region_names):
        if group_id >= len(by_group):
            continue
        group_block_table = by_group[group_id]
        if group_block_table is None or group_block_table.numel() == 0:
            continue
        block_tables_by_type[int(attn_type_enum)] = group_block_table
    return block_tables_by_type or None


class PoolBackedModule(nn.Module):
    """Base class for modules that bind/scatter state and KV cache from
    framework-managed paged pools (Compressor, Indexer).

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
    ) -> None:
        self._kv_pool_view = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._state_pool_view = state_pool_view
        self._state_block_table = state_block_table
        self._state_eb = state_eb

    def clear_pool_context(self) -> None:
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0
        self._state_pool_view = None
        self._state_block_table = None
        self._state_eb = 0

    def _compute_pool_slots(
        self,
        bsz: int,
        T: int,
        block_table: torch.Tensor,
        eb: int,
        device: torch.device,
        pool_rows: Optional[int] = None,
    ) -> tuple:
        max_blocks = block_table.shape[1]
        pool_capacity = max_blocks * eb
        pos = torch.arange(T, device=device, dtype=torch.long)
        in_capacity_row = pos < pool_capacity
        safe_pos = torch.where(in_capacity_row, pos, torch.zeros_like(pos))
        block_in_seq = safe_pos // eb
        in_block = safe_pos % eb
        bt_long = block_table.to(torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
        block_id = bt_long[:bsz][b_idx, block_in_seq.unsqueeze(0)]
        in_capacity = in_capacity_row.unsqueeze(0).expand(bsz, -1)
        candidate_slot = block_id * eb + in_block.unsqueeze(0)
        valid = (block_id > 0) & in_capacity
        if pool_rows is not None:
            valid = valid & (candidate_slot < int(pool_rows))
        safe_slot = torch.where(valid, candidate_slot, torch.zeros_like(block_id))
        return valid, safe_slot

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
            bsz,
            T,
            self._state_block_table,
            self._state_eb,
            device,
            pool_rows=int(self._state_pool_view.shape[0]),
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
            bsz,
            T,
            self._state_block_table,
            self._state_eb,
            device,
            pool_rows=int(self._state_pool_view.shape[0]),
        )
        merged = torch.cat([self.kv_state[:bsz], self.score_state[:bsz]], dim=-1)
        merged_flat = merged.reshape(bsz * T, -1)
        slot_mapping = torch.where(
            valid, safe_slot, torch.full_like(safe_slot, -1)
        ).reshape(-1)
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
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
            bsz,
            T,
            self._kv_block_table,
            self._kv_eb,
            device,
            pool_rows=int(self._kv_pool_view.shape[0]),
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
            bsz,
            T,
            self._kv_block_table,
            self._kv_eb,
            device,
            pool_rows=int(self._kv_pool_view.shape[0]),
        )
        if block_mask is not None:
            valid = valid & block_mask[:bsz].to(device)
        slot_mapping = torch.where(
            valid, safe_slot, torch.full_like(safe_slot, -1)
        ).reshape(-1)
        D = int(self.kv_cache.shape[2])
        flat = self.kv_cache[:bsz].reshape(bsz * T, D)
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        # No bf16 → FP8 packing here. CSA/HCA prefill/decode go through
        # _compressor_kv_fused_triton.v4_compressor_kv_fused (canonical 584B);
        # SWA-only writes go through _prefill_write_swa_fp8_paged. This
        # generic scatter only handles same-dtype pools (bf16 KV, fp32 STATE).
        assert not is_fp8_swa_slot_pool(self._kv_pool_view) or D != 512, (
            "FP8 584B KV pool reached generic _scatter_kv_cache_to_pool — "
            "writes must go through the fused canonical path "
            "(v4_compressor_kv_fused / _prefill_write_swa_fp8_paged)."
        )
        write_kv_to_pool(flat, slot_mapping, self._kv_pool_view, mask_negative=True)

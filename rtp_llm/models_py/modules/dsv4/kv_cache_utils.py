"""DSV4 KV-cache lookup utilities.

Generic ``(layer_id, attn_type) → group_id`` and ``(layer, attn_type) →
block_table`` helpers shared between prefill and decode. Kept separate
from :mod:`attn_type` (pure int constants, no torch) and from path-
specific forward helpers in :mod:`prefill.forward` / :mod:`decode.forward`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

# DSV4 per-layer gid slot count. Must match the alloc size used in
# ``NormalModelInputGatherer`` for ``kv_cache_layer_to_group_dpsk_v4``.
# A CSA layer touches 5 groups (SWA_KV, CSA_KV, INDEXER_KV, INDEXER_STATE,
# CSA_STATE); HCA touches 3; SWA-only touches 1. Unused slots hold -1.
_DSV4_MAX_GROUPS_PER_LAYER = 5


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

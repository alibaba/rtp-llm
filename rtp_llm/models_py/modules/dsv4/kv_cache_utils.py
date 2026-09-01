"""DSV4 KV-cache lookup utilities.

Generic ``(layer, region_name) -> block_table`` helpers shared between
prefill and decode. Kept separate from region-name constants (pure int
constants, no torch) and from path-specific forward helpers in
:mod:`prefill.forward` / :mod:`decode.forward`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def resolve_block_table_group_ids(kv_cache: Optional[Any]) -> Dict[int, int]:
    """Resolve typed DSV4 cache regions to process-wide table group ids.

    ``KVCache.group_region_names`` and ``group_seq_size_per_block`` describe
    the model-local layout. MTP models therefore expose compact local group
    ids, while ``PyAttentionInputs`` always carries the process-wide
    ``kv_cache_kernel_block_id_device_by_group`` list.

    ``get_layer_caches(...).physical_group_id`` provides the exact mapping.
    Typed regions must never fall back to their model-local layout ids: those
    ids can address another model's pool in the process-wide table list.
    Region 0 (DEFAULT) is intentionally ignored: a hybrid model may own
    multiple default pools, while DSV4 paged metadata is keyed only by typed
    regions.
    """
    if kv_cache is None:
        return {}

    group_region_names = getattr(kv_cache, "group_region_names", None)
    typed_regions = {
        int(region_name)
        for region_name in (group_region_names or [])
        if int(region_name) != 0
    }

    resolved: Dict[int, int] = {}
    # KVCache stores a container-level physical mapping in C++, but that field
    # is not pybound. Use the exact physical id exposed on each LayerKVCache.
    get_layer_caches = getattr(kv_cache, "get_layer_caches", None)
    layer_mapping = getattr(kv_cache, "layer_region_to_group_id", None)
    if callable(get_layer_caches) and layer_mapping:
        for layer_id in range(len(layer_mapping)):
            for layer_cache in get_layer_caches(layer_id):
                region_id = int(getattr(layer_cache, "region_name", 0))
                physical_group_id = int(
                    getattr(layer_cache, "physical_group_id", -1)
                )
                if region_id == 0 or physical_group_id < 0:
                    continue
                previous = resolved.get(region_id)
                if previous is not None and previous != physical_group_id:
                    raise RuntimeError(
                        "DSV4 cache region %d maps to multiple physical "
                        "groups through get_layer_caches: %d and %d"
                        % (region_id, previous, physical_group_id)
                    )
                resolved[region_id] = physical_group_id
        missing_regions = sorted(typed_regions.difference(resolved))
        if missing_regions:
            raise RuntimeError(
                "DSV4 cache regions %r have no physical group mapping"
                % missing_regions
            )
        if resolved:
            return resolved

    if not typed_regions:
        return {}
    raise RuntimeError(
        "DSV4 typed cache regions require physical group ids, but "
        "LayerKVCache.physical_group_id is unavailable"
    )


def _build_block_tables_from_group_ids(
    table_group_ids: Dict[int, int],
    attn_inputs: Any,
    batch_offset: Optional[int],
) -> Optional[Dict[int, torch.Tensor]]:
    if not table_group_ids or attn_inputs is None:
        return None
    by_group = getattr(attn_inputs, "kv_cache_kernel_block_id_device_by_group", None)
    if by_group is None or len(by_group) == 0:
        return None
    block_tables: Dict[int, torch.Tensor] = {}
    for region_name, table_group_id in table_group_ids.items():
        if table_group_id < 0 or table_group_id >= len(by_group):
            raise RuntimeError(
                "DSV4 cache region %d maps to physical group %d, but only "
                "%d block-table groups are available"
                % (int(region_name), int(table_group_id), len(by_group))
            )
        table = by_group[table_group_id]
        if table is None or table.numel() == 0:
            raise RuntimeError(
                "DSV4 cache region %d maps to physical group %d, but its "
                "block table is empty"
                % (int(region_name), int(table_group_id))
            )
        block_tables[region_name] = (
            table if batch_offset is None else table[batch_offset : batch_offset + 1]
        )
    return block_tables or None


def _build_block_tables_by_region(
    kv_cache: Optional[Any],
    attn_inputs: Any,
    batch_offset: Optional[int],
) -> Optional[Dict[int, torch.Tensor]]:
    if kv_cache is None:
        return None
    return _build_block_tables_from_group_ids(
        resolve_block_table_group_ids(kv_cache), attn_inputs, batch_offset
    )


def build_block_tables(
    kv_cache: Optional[Any],
    attn_inputs: Any,
    batch_offset: int = 0,
) -> Optional[Dict[int, torch.Tensor]]:
    """Build the per-region-name block-table dict for one prefill request.

    The framework emits per-request block tables as a list indexed by
    ``group_id`` (``attn_inputs.kv_cache_kernel_block_id_device_by_group``,
    one entry per process-wide physical pool group). This helper resolves the
    model's typed regions through each ``LayerKVCache.physical_group_id`` and
    produces a dict keyed by ``KVCacheRegionName`` integer value instead of
    local layout group id.

    The ``batch_offset`` arg slices out a single-request row
    ``[batch_offset : batch_offset + 1]`` so the returned block table is
    per-request, matching how ``DeepSeekV4Model.forward`` unrolls batched
    prefill into one-request-at-a-time layer calls.

    Returns ``None`` when no block tables are available (warmup / paged-KV
    disabled / missing framework state).
    """
    return _build_block_tables_by_region(kv_cache, attn_inputs, batch_offset)


def build_block_tables_batched(
    kv_cache: Optional[Any],
    attn_inputs: Any,
) -> Optional[Dict[int, torch.Tensor]]:
    """Build the per-region-name block-table dict for an entire prefill batch.

    Same semantics as :func:`build_block_tables` but returns the full
    ``[B, max_blocks]`` block table per region name (no ``batch_offset`` slice).
    Used by the batched ``forward_prefill`` main path so a single ``v4()`` call
    can cover the whole batch.

    Returns ``None`` when no block tables are available (warmup / paged-KV
    disabled / missing framework state).
    """
    return _build_block_tables_by_region(kv_cache, attn_inputs, None)


def build_block_tables_batched_from_group_ids(
    table_group_ids: Dict[int, int],
    attn_inputs: Any,
) -> Optional[Dict[int, torch.Tensor]]:
    """Build batched block tables without touching the pybind KVCache."""
    return _build_block_tables_from_group_ids(table_group_ids, attn_inputs, None)

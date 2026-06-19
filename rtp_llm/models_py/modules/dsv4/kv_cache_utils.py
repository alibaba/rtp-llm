"""DSV4 KV-cache lookup utilities.

Generic ``(layer, tag) -> block_table`` helpers shared between
prefill and decode. Kept separate from attn-type constants (pure int
constants, no torch) and from path-specific forward helpers in
:mod:`prefill.forward` / :mod:`decode.forward`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import TAG_BY_ATTN_TYPE


ATTN_TYPE_BY_TAG = {tag: attn_type for attn_type, tag in TAG_BY_ATTN_TYPE.items()}


def build_block_tables(
    kv_cache: Optional[Any],
    attn_inputs: Any,
    batch_offset: int = 0,
) -> Optional[Dict[int, torch.Tensor]]:
    """Build the per-attn-type block-table dict for one prefill request.

    The framework emits per-request block tables as a list indexed by
    ``group_id`` (``attn_inputs.kv_cache_kernel_block_id_device_by_group``,
    one entry per pool group in the order declared by
    ``CacheConfig::group_tags``). This helper joins that list against
    ``kv_cache.group_tags`` to produce a dict keyed by DSV4 attn_type integer
    value instead of group id.

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
    group_tags = getattr(kv_cache, "group_tags", None)
    if not group_tags:
        return None
    block_tables_by_attn_type: Dict[int, torch.Tensor] = {}
    for group_id, tag in enumerate(group_tags):
        if group_id >= len(by_group):
            continue
        attn_type = ATTN_TYPE_BY_TAG.get(str(tag))
        if attn_type is None:
            continue
        group_block_table = by_group[group_id]
        if group_block_table is None or group_block_table.numel() == 0:
            continue
        block_tables_by_attn_type[attn_type] = group_block_table[
            batch_offset : batch_offset + 1
        ]
    return block_tables_by_attn_type or None


def build_block_tables_batched(
    kv_cache: Optional[Any],
    attn_inputs: Any,
) -> Optional[Dict[int, torch.Tensor]]:
    """Build the per-attn-type block-table dict for an entire prefill batch.

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
    group_tags = getattr(kv_cache, "group_tags", None)
    if not group_tags:
        return None
    block_tables_by_attn_type: Dict[int, torch.Tensor] = {}
    for group_id, tag in enumerate(group_tags):
        if group_id >= len(by_group):
            continue
        attn_type = ATTN_TYPE_BY_TAG.get(str(tag))
        if attn_type is None:
            continue
        group_block_table = by_group[group_id]
        if group_block_table is None or group_block_table.numel() == 0:
            continue
        block_tables_by_attn_type[attn_type] = group_block_table
    return block_tables_by_attn_type or None

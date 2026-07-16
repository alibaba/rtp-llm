"""DSV4 KV-cache lookup utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable, Optional

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import TAG_BY_ATTN_TYPE

ATTN_TYPE_BY_TAG = {tag: attn_type for attn_type, tag in TAG_BY_ATTN_TYPE.items()}


def iter_tagged_attention_inputs(
    kv_cache: Optional[Any], attention_inputs: Any
) -> Iterable[tuple[str, Any]]:
    """Iterate model cache inputs by semantic tag.

    Multi-group inputs must carry their tags in the mapping itself. The plain
    ``PyAttentionInputs`` form is the 1:1 fast path and is accepted only when
    the cache topology exposes exactly one group tag.
    """
    if attention_inputs is None:
        return []

    if isinstance(attention_inputs, Mapping):
        if not attention_inputs:
            raise RuntimeError("DSV4 attention input tag mapping must not be empty")
        return [(str(tag), value) for tag, value in attention_inputs.items()]

    group_tags = list(getattr(kv_cache, "group_tags", None) or [])
    if len(group_tags) != 1:
        raise RuntimeError(
            "plain DSV4 attention inputs require a single-group cache topology; "
            f"group_tags={group_tags!r}"
        )
    return [(str(group_tags[0]), attention_inputs)]


def _iter_group_block_tables(
    kv_cache: Optional[Any], attention_inputs: Any
) -> Iterable[tuple[int, torch.Tensor]]:
    if kv_cache is None or attention_inputs is None:
        return []

    result = []
    for tag, attn_inputs in iter_tagged_attention_inputs(kv_cache, attention_inputs):
        attn_type = ATTN_TYPE_BY_TAG.get(tag)
        if attn_type is None:
            continue
        block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
        if block_table is None or block_table.numel() == 0:
            continue
        result.append((attn_type, block_table))
    return result


def build_block_tables(
    kv_cache: Optional[Any],
    attention_inputs: Any,
    batch_offset: int = 0,
) -> Optional[Dict[int, torch.Tensor]]:
    """Build per-attn-type block tables for one prefill request."""
    block_tables_by_attn_type: Dict[int, torch.Tensor] = {}
    for attn_type, block_table in _iter_group_block_tables(kv_cache, attention_inputs):
        block_tables_by_attn_type[attn_type] = block_table[
            batch_offset : batch_offset + 1
        ]
    return block_tables_by_attn_type or None


def build_block_tables_batched(
    kv_cache: Optional[Any],
    attention_inputs: Any,
) -> Optional[Dict[int, torch.Tensor]]:
    """Build per-attn-type block tables for an entire prefill batch."""
    block_tables_by_attn_type: Dict[int, torch.Tensor] = {}
    for attn_type, block_table in _iter_group_block_tables(kv_cache, attention_inputs):
        block_tables_by_attn_type[attn_type] = block_table
    return block_tables_by_attn_type or None

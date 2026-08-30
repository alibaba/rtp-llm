"""DSV4 KV-cache tags and lookup utilities.

This module owns two things:

1. The canonical DSV4 cache-group **tags**. The framework KV cache API is
   tag-driven: a cache group is identified by a string tag ("swa_kv",
   "csa_kv", ...) which is the same string used by ``CacheConfig`` ownership
   on the C++ side, by ``KVCache.get_layer_cache(layer, tag)``
   / ``KVCache.get_seq_size_per_block(tag)``, and as the key of
   ``PyModelInputs.attention_inputs`` when the model owns several groups.
   These constants replace the old int ``attn_type`` ids that mirrored the
   deleted C++ ``KVCacheRegionName`` enum.

2. Generic ``tag -> block_table`` helpers shared between prefill and decode,
   plus the normalizers that turn ``PyModelInputs.attention_inputs`` (which is
   *either* a single ``PyAttentionInputs`` *or* a ``{tag: PyAttentionInputs}``
   mapping) into something callers can index by tag.

Path-specific forward helpers live in :mod:`prefill.forward` /
:mod:`decode.forward`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Canonical cache-group tags. These are the *consumer* side of the tags that
# ``rtp_llm/models/dsv4_kv_cache.py`` (``CSA_KV_TAG`` ... ``SWA_KV_TAG``) hands
# to ``ModelConfig.kv_cache_spec_descs`` and that CacheConfig then
# publishes as ``KVCache.group_tags``. They are duplicated rather than imported
# on purpose: ``rtp_llm.models.dsv4_kv_cache`` pulls in ``rtp_llm.ops`` (the
# compiled .so), while this module must stay importable from kernel-level code
# and unit tests that only need torch. Keep both lists in sync.
# ---------------------------------------------------------------------------
SWA_KV = "swa_kv"
CSA_KV = "csa_kv"
HCA_KV = "hca_kv"
INDEXER_KV = "indexer_kv"
INDEXER_STATE = "indexer_state"
CSA_STATE = "csa_state"
HCA_STATE = "hca_state"

# Single-group models expose exactly one group under this tag.
DEFAULT_TAG = "default"

# Paged (FULL) KV pools — a block-table row covers ``kernel_seq_size_per_block``
# raw tokens.
DSV4_KERNEL_ROW_TAGS: Tuple[str, ...] = (CSA_KV, HCA_KV, INDEXER_KV)
# Fixed / ring pools — a block-table row covers ``seq_size_per_block`` raw
# tokens.
DSV4_PHYSICAL_ROW_TAGS: Tuple[str, ...] = (
    SWA_KV,
    CSA_STATE,
    HCA_STATE,
    INDEXER_STATE,
)
DSV4_TAGS: Tuple[str, ...] = DSV4_KERNEL_ROW_TAGS + DSV4_PHYSICAL_ROW_TAGS


def kv_tag_for_compress_ratio(ratio: int) -> Optional[str]:
    """Compressed KV pool tag for a layer's compression ratio (``None`` = SWA-only)."""
    if int(ratio) == 4:
        return CSA_KV
    if int(ratio) == 128:
        return HCA_KV
    return None


def group_tags(kv_cache: Optional[Any]) -> List[str]:
    """Framework cache group tags in canonical sorted order (``[]`` when absent).

    The list is a set of semantic identities; a position in it never identifies a
    cache group.
    """
    if kv_cache is None:
        return []
    tags = getattr(kv_cache, "group_tags", None)
    if not tags:
        return []
    return [str(tag) for tag in tags]


def as_attention_inputs_by_tag(
    attention_inputs: Any,
    kv_cache: Optional[Any] = None,
) -> Dict[str, Any]:
    """Normalize ``attention_inputs`` into a ``{tag: PyAttentionInputs}`` dict.

    ``attention_inputs`` may be

    * a ``{tag: PyAttentionInputs}`` mapping — the multi-group case, which is
      what ``PyModelInputs.attention_inputs`` returns for DSV4 and what C++
      ``callPrepareCudaGraph`` hands to ``prepare_cuda_graph``;
    * a single ``PyAttentionInputs`` — the common/single-group fast path. It is
      keyed by the model's only group tag when ``kv_cache`` exposes exactly one,
      otherwise by ``"default"``;
    * a ``PyModelInputs`` — unwrapped first;
    * ``None`` — yields ``{}``.
    """
    if attention_inputs is None:
        return {}
    if hasattr(attention_inputs, "attention_inputs"):
        attention_inputs = attention_inputs.attention_inputs
        if attention_inputs is None:
            return {}
    if isinstance(attention_inputs, Mapping):
        return {str(tag): value for tag, value in attention_inputs.items()}
    tags = group_tags(kv_cache)
    tag = tags[0] if len(tags) == 1 else DEFAULT_TAG
    return {tag: attention_inputs}


def primary_attention_inputs(
    attention_inputs: Any,
    kv_cache: Optional[Any] = None,
) -> Optional[Any]:
    """Return the per-forward inputs carrying the group-invariant fields.

    Every tagged entry is a copy of the same common ``PyAttentionInputs``
    (``PyWrappedModel::setupKVCacheForAttentionInputs`` clones it per group and
    only overwrites the block-table fields), so any entry is a valid source for
    ``cu_seqlens`` / ``input_lengths`` / ``sequence_lengths`` /
    ``prefix_lengths`` / ``cache_store_inputs`` / ``context_parallel_info``.
    Only block tables are group-local — read those through
    :func:`build_block_tables` / :func:`build_block_tables_batched`.
    """
    if attention_inputs is None:
        return None
    if hasattr(attention_inputs, "attention_inputs"):
        attention_inputs = attention_inputs.attention_inputs
        if attention_inputs is None:
            return None
    if not isinstance(attention_inputs, Mapping):
        return attention_inputs
    if not attention_inputs:
        return None
    # Prefer a known DSV4 tag so repeated reads are stable across steps
    # regardless of mapping iteration order; fall back to first entry.
    for tag in DSV4_TAGS:
        if tag in attention_inputs:
            return attention_inputs[tag]
    return next(iter(attention_inputs.values()))


def _block_table_for_tag(tagged_inputs: Any) -> Optional[torch.Tensor]:
    if tagged_inputs is None:
        return None
    block_table = getattr(tagged_inputs, "kv_cache_kernel_block_id_device", None)
    if block_table is None or block_table.numel() == 0:
        return None
    return block_table


def _build_block_tables(
    attention_inputs: Any,
    kv_cache: Optional[Any],
    batch_slice: Optional[slice],
    keep_tags: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    by_tag = as_attention_inputs_by_tag(attention_inputs, kv_cache)
    if not by_tag:
        return None
    wanted = None if keep_tags is None else set(keep_tags)
    block_tables: Dict[str, torch.Tensor] = {}
    for tag, tagged_inputs in by_tag.items():
        if wanted is not None and tag not in wanted:
            continue
        block_table = _block_table_for_tag(tagged_inputs)
        if block_table is None:
            continue
        block_tables[tag] = (
            block_table if batch_slice is None else block_table[batch_slice]
        )
    return block_tables or None


def build_block_tables(
    kv_cache: Optional[Any],
    attention_inputs: Any,
    batch_offset: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build the per-tag block-table dict for one prefill request.

    The framework hands each cache group its own ``PyAttentionInputs`` copy via
    ``PyModelInputs.attention_inputs`` (a ``{tag: inputs}`` mapping), each
    carrying that group's kernel-granularity block table in
    ``kv_cache_kernel_block_id_device``. This helper collects them into a dict
    keyed by cache tag.

    The ``batch_offset`` arg slices out a single-request row
    ``[batch_offset : batch_offset + 1]`` so the returned block table is
    per-request, matching how ``DeepSeekV4Model.forward`` unrolls batched
    prefill into one-request-at-a-time layer calls.

    Returns ``None`` when no block tables are available (warmup / paged-KV
    disabled / missing framework state).
    """
    return _build_block_tables(
        attention_inputs,
        kv_cache,
        slice(batch_offset, batch_offset + 1),
    )


def build_block_tables_batched(
    kv_cache: Optional[Any],
    attention_inputs: Any,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build the per-tag block-table dict for an entire prefill batch.

    Same semantics as :func:`build_block_tables` but returns the full
    ``[B, max_blocks]`` block table per tag (no ``batch_offset`` slice).
    Used by the batched ``forward_prefill`` main path so a single ``v4()`` call
    can cover the whole batch.

    Returns ``None`` when no block tables are available (warmup / paged-KV
    disabled / missing framework state).
    """
    return _build_block_tables(attention_inputs, kv_cache, None)


def build_block_tables_for_tags(
    kv_cache: Optional[Any],
    attention_inputs: Any,
    tags: Iterable[str],
) -> Optional[Dict[str, torch.Tensor]]:
    """Batched block tables restricted to ``tags`` (decode's paged-pool set)."""
    return _build_block_tables(attention_inputs, kv_cache, None, keep_tags=tags)

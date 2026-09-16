from typing import Optional, Type

import torch
from torch import nn

from rtp_llm.ops.compute_ops import (
    CacheStoreWriter,
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    PyCacheStoreInputs,
)


class WriteCacheStoreOp(nn.Module):
    """Hand per-layer KV cache handles to the C++ cache-store writer.

    Cache-store planning (pinned-host length mirrors, block tables, per-group
    routing, CP canonical keys) lives in the C++ ``CacheStoreWriter``; python
    only pairs the writer with the per-forward inputs.

    The bound ``cache_store_inputs`` contains one cache group's physical block
    table, so callers must pair this op with exactly one ``LayerKVCache`` from
    the same group. Multi-group models route their per-tag ops explicitly.

    ``input_lengths`` / ``prefix_lengths`` / ``kv_cache_block_id_host`` are the
    read-only mirrors of the plan the C++ writer will execute for this forward:
    the CP-aware per-request token counts, the pinned-host prefix lengths and the
    host block table. Nothing in ``forward`` consumes them — they exist so that
    callers and tests can inspect *which* lengths a given forward will store
    without reaching into ``PyCacheStoreInputs`` (whose ``input_lengths_host``
    is the pre-CP-override mirror and therefore not the effective value).
    """

    def __init__(
        self,
        cache_store_writer: Optional[CacheStoreWriter],
        cache_store_inputs: Optional[PyCacheStoreInputs],
        *,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        kv_cache_block_id_host: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cache_store_writer = cache_store_writer
        self.cache_store_inputs = cache_store_inputs
        self.input_lengths = input_lengths
        self.prefix_lengths = prefix_lengths
        self.kv_cache_block_id_host = kv_cache_block_id_host

    def forward(self, kv_cache: Optional[LayerKVCache]) -> None:
        if self.cache_store_writer is None or self.cache_store_inputs is None:
            return
        if kv_cache is None:
            return
        self.cache_store_writer.write(self.cache_store_inputs, kv_cache)


# ``cache_store_writer`` is a C++-owned field of ``PyAttentionInputs`` and is
# always present in production. Duck-typed inputs that only describe the write
# *plan* may omit it entirely, which is not the same statement as "this forward
# stores nothing" (an explicit ``None``); the sentinel keeps the two apart.
_WRITER_FIELD_ABSENT = object()


def _first_non_empty(
    primary: Optional[torch.Tensor], fallback: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if primary is not None and primary.numel() > 0:
        return primary
    return fallback


def create_write_cache_store_impl(
    attn_inputs: PyAttentionInputs,
    kv_cache: Optional[KVCache] = None,
    *,
    op_cls: Type[WriteCacheStoreOp] = WriteCacheStoreOp,
) -> Optional[WriteCacheStoreOp]:
    """Create the per-forward write-cache-store op, or None when not needed.

    ``kv_cache`` is accepted for call-site compatibility with the multi-region
    DSv4 prefill loop. No python-side block-table planning is required: the C++
    writer resolves the owning cache group from each ``LayerKVCache``'s layer id
    and cache tag.

    ``op_cls`` is the class actually instantiated. Platform-specific factories
    (``modules.factory.attention.common``) pass their own module-level
    ``WriteCacheStoreOp`` binding so that overriding/patching it there is
    honoured instead of being silently bypassed by this delegation.
    """
    if not attn_inputs.is_prefill:
        return None

    cache_store_inputs = attn_inputs.cache_store_inputs
    if cache_store_inputs is None:
        return None

    cache_store_writer = getattr(
        attn_inputs, "cache_store_writer", _WRITER_FIELD_ABSENT
    )
    if cache_store_writer is None:
        return None
    if cache_store_writer is _WRITER_FIELD_ABSENT:
        cache_store_writer = None

    # Prefer the pinned-host length mirrors prepared by prepareWriteCacheParams
    # over the device tensors, so inspecting the plan never forces a D2H copy.
    input_lengths = _first_non_empty(
        getattr(cache_store_inputs, "input_lengths_host", None),
        getattr(attn_inputs, "input_lengths", None),
    )
    # Under CP each rank holds only its shard of the sequence, but the cache
    # store covers the whole request, so the un-sharded lengths win.
    cp_info = getattr(attn_inputs, "context_parallel_info", None)
    if cp_info is not None:
        actual_lengths = getattr(cp_info, "prefill_actual_input_lengths_cpu", None)
        if actual_lengths is not None and actual_lengths.numel() > 0:
            input_lengths = actual_lengths

    prefix_lengths = _first_non_empty(
        getattr(cache_store_inputs, "prefix_lengths_host", None),
        getattr(attn_inputs, "prefix_lengths", None),
    )

    # ``PyAttentionInputs`` names the host block table ``kv_cache_block_id``
    # (its device mirror is ``kv_cache_block_id_device``). Duck-typed inputs
    # written against the older ``*_host`` spelling still resolve via fallback.
    block_id_host = _first_non_empty(
        getattr(attn_inputs, "kv_cache_block_id", None),
        getattr(attn_inputs, "kv_cache_block_id_host", None),
    )

    return op_cls(
        cache_store_writer,
        cache_store_inputs,
        input_lengths=input_lengths,
        prefix_lengths=prefix_lengths,
        kv_cache_block_id_host=block_id_host,
    )

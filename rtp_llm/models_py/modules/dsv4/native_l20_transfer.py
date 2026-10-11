"""Bind the qualified L20 byte protocol to native producer/read-replica views.

The worker retains the native allocation lease and serializes forwards. PP
tickets must complete before import and drain before request release. This
adapter does not construct an attention owner or admit a PP topology.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Contract,
    V41L20ReplicaReceiver,
    V41L20Span,
    _cache_pair,
    _Replica,
    export_l20_delta,
)

_TAGS = ("hca_kv", "indexer_kv")


def _require_eager(raw: torch.Tensor) -> None:
    if raw.is_cuda:
        with torch.cuda.device(raw.device):
            if torch.cuda.is_current_stream_capturing():
                raise NotImplementedError(
                    "V4.1 native L20 transfer requires eager execution"
                )


def native_l20_cache_views(
    text_config: Mapping[str, Any], kv_cache: Any, pp_rank: int
) -> dict[str, V41PagedCache]:
    """Return zero-copy L20 views after checking the native stage contract."""
    bindings = dsv41_cache_bindings(text_config, 2, pp_rank)
    if kv_cache.layer_count != (21 if pp_rank == 0 else 19):
        raise ValueError("V4.1 L20 native cache has the wrong stage-local layer count")
    views = {}
    for tag in _TAGS:
        if (
            kv_cache.get_seq_size_per_block(tag) != 256
            or kv_cache.get_kernel_seq_size_per_block(tag) != 256
        ):
            raise ValueError("V4.1 L20 native cache requires pages 256/256")
        binding = next(b for b in bindings if b.source_layer_id == 20 and b.tag == tag)
        layer = kv_cache.get_layer_cache(binding.storage_layer_id, tag)
        raw = layer.kv_cache_base
        if not isinstance(raw, torch.Tensor) or not raw.numel():
            raise ValueError("V4.1 L20 native cache requires allocated byte storage")
        _require_eager(raw)
        views[tag] = V41PagedCache.from_layer_cache(layer, binding, gen_num_per_cycle=0)
    return _cache_pair(views)


def _native_tables(blocks: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, ...]]:
    if not isinstance(blocks, Mapping) or set(blocks) != set(_TAGS):
        raise ValueError("V4.1 native L20 allocation must cover both tagged regions")
    tables = {tag: tuple(blocks[tag]) for tag in _TAGS}
    if any(
        type(page) is not int or page <= 0
        for pages in tables.values()
        for page in pages
    ):
        raise ValueError("V4.1 native L20 allocation cannot use reserved page zero")
    return tables


def export_native_l20_delta(
    text_config: Mapping[str, Any],
    kv_cache: Any,
    contract: V41L20Contract,
    cp_rank: int,
    round_id: int,
    spans: Sequence[V41L20Span],
    block_tables: Mapping[V41CacheIdentity, Mapping[str, Sequence[int]]],
    *,
    backend: str = "triton",
) -> dict[str, torch.Tensor]:
    """Export completed native L20 producer rows through the existing protocol."""
    contract.require_attention_config(text_config)
    views = native_l20_cache_views(text_config, kv_cache, 0)
    tables = {
        identity: _native_tables(blocks) for identity, blocks in block_tables.items()
    }
    return export_l20_delta(
        contract, cp_rank, round_id, spans, views, tables, backend=backend
    )


class NativeL20ReplicaReceiver(V41L20ReplicaReceiver):
    """Existing generation/commit lifecycle with native replica admission."""

    def __init__(
        self,
        text_config: Mapping[str, Any],
        kv_cache: Any,
        contract: V41L20Contract,
        cp_rank: int,
        *,
        backend: str = "triton",
    ) -> None:
        contract.require_attention_config(text_config)
        views = native_l20_cache_views(text_config, kv_cache, 1)
        super().__init__(contract, cp_rank, views, backend=backend)
        self._native_cache = kv_cache

    def _usable(self) -> None:
        _require_eager(self.caches["hca_kv"].raw)
        super()._usable()

    def _state(self, identity: V41CacheIdentity) -> _Replica:
        _require_eager(self.caches["hca_kv"].raw)
        return super()._state(identity)

    def begin_request(
        self, identity: V41CacheIdentity, blocks: Mapping[str, Sequence[int]]
    ) -> None:
        super().begin_request(identity, _native_tables(blocks))

    def extend_request(
        self, identity: V41CacheIdentity, blocks: Mapping[str, Sequence[int]]
    ) -> None:
        super().extend_request(identity, _native_tables(blocks))

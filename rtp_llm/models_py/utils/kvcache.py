from __future__ import annotations

from collections.abc import Sequence

import torch

from rtp_llm.ops.compute_ops import LayerKVCache


class SingleGroupKVCacheAdapter:
    """Small Python-owned cache used by standalone and CUDA-graph tests.

    Production whole-model caches are created only by C++ from a canonical
    GroupedCacheLayerLayout. This adapter intentionally exposes the same
    read-only query surface without pretending to own production topology.
    """

    def __init__(
        self,
        layer_tensors: Sequence[torch.Tensor],
        seq_size_per_block: int,
        *,
        tag: str = "default",
        layer_scale_tensors: Sequence[torch.Tensor | None] | None = None,
    ) -> None:
        if not layer_tensors:
            raise ValueError("SingleGroupKVCacheAdapter requires at least one layer")
        if seq_size_per_block <= 0:
            raise ValueError("seq_size_per_block must be positive")
        if not tag:
            raise ValueError("tag must not be empty")
        if layer_scale_tensors is not None and len(layer_scale_tensors) != len(
            layer_tensors
        ):
            raise ValueError("layer scale tensor count must match layer tensor count")
        self._layer_tensors = tuple(layer_tensors)
        self._layer_scale_tensors = (
            tuple(layer_scale_tensors)
            if layer_scale_tensors is not None
            else (None,) * len(layer_tensors)
        )
        self._seq_size_per_block = int(seq_size_per_block)
        self._tag = str(tag)

    @property
    def group_tags(self) -> list[str]:
        return [self._tag]

    @property
    def layer_count(self) -> int:
        return len(self._layer_tensors)

    def _validate_layer(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.layer_count:
            raise RuntimeError(f"Invalid layer index: {layer_id}")

    def _validate_tag(self, tag: str) -> None:
        if tag != self._tag:
            raise RuntimeError(f"KV cache tag {tag!r} is not available")

    def get_layer_cache(self, layer_id: int, tag: str | None = None) -> LayerKVCache:
        self._validate_layer(layer_id)
        self._validate_tag(self._tag if tag is None else str(tag))
        return LayerKVCache(
            self._layer_tensors[layer_id],
            self._seq_size_per_block,
            layer_id=layer_id,
            tag=self._tag,
            kv_scale_base=self._layer_scale_tensors[layer_id],
        )

    def get_layer_cache_groups(self, layer_id: int) -> list[LayerKVCache]:
        return [self.get_layer_cache(layer_id)]

    def get_seq_size_per_block(self, tag: str) -> int:
        self._validate_tag(str(tag))
        return self._seq_size_per_block

    def get_kernel_seq_size_per_block(self, tag: str) -> int:
        self._validate_tag(str(tag))
        return self._seq_size_per_block

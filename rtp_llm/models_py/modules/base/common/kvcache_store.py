from collections.abc import Sequence
from typing import List, Optional, Union

import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.ops.compute_ops import (
    KVCache,
    KVCacheRegionName,
    LayerKVCache,
    PyAttentionInputs,
    PyCacheStoreInputs,
)


def write_typed_aux_cache_regions(
    write_cache_store_impl: Optional[nn.Module],
    kv_cache: Optional[KVCache],
    layer_idx: int,
) -> None:
    """Publish non-default typed regions after a layer finishes prefill."""
    if write_cache_store_impl is None or kv_cache is None:
        return
    aux_regions = [
        layer_cache
        for layer_cache in kv_cache.get_layer_caches(layer_idx)
        if layer_cache.region_name != KVCacheRegionName.DEFAULT
    ]
    if aux_regions:
        write_cache_store_impl(aux_regions)


def _cache_store_host_i32(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize metadata before a cache-store background thread reads it."""
    if tensor is None or not tensor.numel():
        raise RuntimeError(f"cache-store {name} must be a non-empty tensor")
    if tensor.dtype != torch.int32:
        raise RuntimeError(
            f"cache-store {name} must be int32, got {tensor.dtype}"
        )
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor.contiguous()


class WriteCacheStoreOp(nn.Module):
    def __init__(
        self,
        input_lengths: torch.Tensor,
        prefix_lengths: torch.Tensor,
        kv_cache_block_id_host: Union[torch.Tensor, Sequence[Optional[torch.Tensor]]],
        cache_store_inputs: Optional[PyCacheStoreInputs],
    ):
        super().__init__()
        self.input_lengths = input_lengths
        self.prefix_lengths = prefix_lengths
        self.cache_store_inputs = cache_store_inputs
        if isinstance(kv_cache_block_id_host, torch.Tensor):
            self._block_ids_by_group: Optional[List[Optional[torch.Tensor]]] = None
            self.kv_cache_block_id_host = kv_cache_block_id_host
        else:
            self._block_ids_by_group = list(kv_cache_block_id_host)
            self.kv_cache_block_id_host = None

    def _write_one(
        self,
        kv_cache: Optional[LayerKVCache],
        kv_cache_block_id_host: torch.Tensor,
    ) -> None:
        compute_ops.write_cache_store(
            self.input_lengths,
            self.prefix_lengths,
            kv_cache_block_id_host,
            self.cache_store_inputs,
            kv_cache,
        )

    def _block_ids_for_layer_cache(
        self, kv_cache: Optional[LayerKVCache]
    ) -> Optional[torch.Tensor]:
        if self._block_ids_by_group is None:
            return self.kv_cache_block_id_host
        gid = getattr(
            kv_cache,
            "physical_group_id",
            getattr(kv_cache, "group_id", -1),
        )
        layer_id = getattr(kv_cache, "layer_id", -1)
        region_name = getattr(kv_cache, "region_name", None)
        if gid < 0 or gid >= len(self._block_ids_by_group):
            raise RuntimeError(
                "missing cache-store block table for owned KV cache region: "
                f"layer_id={layer_id}, region_name={region_name}, group_id={gid}, "
                f"group_count={len(self._block_ids_by_group)}"
            )
        block_ids = self._block_ids_by_group[gid]
        if block_ids is None or block_ids.numel() == 0:
            raise RuntimeError(
                "empty cache-store block table for owned KV cache region: "
                f"layer_id={layer_id}, region_name={region_name}, group_id={gid}"
            )
        return block_ids

    def forward(
        self,
        kv_cache: Union[Optional[LayerKVCache], Sequence[LayerKVCache]],
    ) -> None:
        if isinstance(kv_cache, Sequence):
            for layer_kv in kv_cache:
                block_ids = self._block_ids_for_layer_cache(layer_kv)
                if block_ids is not None:
                    self._write_one(layer_kv, block_ids)
            return

        block_ids = self._block_ids_for_layer_cache(kv_cache)
        if block_ids is not None:
            self._write_one(kv_cache, block_ids)


def create_write_cache_store_impl(
    attn_inputs: PyAttentionInputs,
    kv_cache: Optional[KVCache] = None,
) -> Optional[WriteCacheStoreOp]:
    if not (attn_inputs.is_prefill and attn_inputs.cache_store_inputs):
        return None

    cache_store_inputs = attn_inputs.cache_store_inputs

    # Prefer pinned-host length mirrors prepared by prepareWriteCacheParams
    # to avoid synchronous D2H copies on cache-store background threads.
    input_lengths = getattr(cache_store_inputs, "input_lengths_host", None)
    if input_lengths is None or not input_lengths.numel():
        input_lengths = attn_inputs.input_lengths
    cp_info = getattr(attn_inputs, "context_parallel_info", None)
    if cp_info is not None:
        actual_lengths = getattr(cp_info, "prefill_actual_input_lengths_cpu", None)
        if actual_lengths is not None and actual_lengths.numel() > 0:
            input_lengths = actual_lengths

    prefix_lengths = getattr(cache_store_inputs, "prefix_lengths_host", None)
    if prefix_lengths is None or not prefix_lengths.numel():
        prefix_lengths = attn_inputs.prefix_lengths

    input_lengths = _cache_store_host_i32(input_lengths, "input_lengths")
    prefix_lengths = _cache_store_host_i32(prefix_lengths, "prefix_lengths")
    # Persist the normalized mirrors on the shared request metadata so all
    # layer writers reuse them instead of repeating a device-to-host copy.
    cache_store_inputs.input_lengths_host = input_lengths
    cache_store_inputs.prefix_lengths_host = prefix_lengths

    has_multi_region = (
        kv_cache is not None
        and bool(getattr(kv_cache, "layer_region_to_group_id", None))
        and bool(getattr(attn_inputs, "kv_cache_kernel_block_id_host_by_group", None))
    )
    if has_multi_region:
        return WriteCacheStoreOp(
            input_lengths,
            prefix_lengths,
            attn_inputs.kv_cache_kernel_block_id_host_by_group,
            cache_store_inputs,
        )

    return WriteCacheStoreOp(
        input_lengths,
        prefix_lengths,
        attn_inputs.kv_cache_block_id_host,
        cache_store_inputs,
    )

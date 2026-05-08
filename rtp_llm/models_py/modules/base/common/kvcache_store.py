from collections.abc import Sequence
from typing import List, Optional, Union

import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.ops.compute_ops import (
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    PyCacheStoreInputs,
)


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
        gid = getattr(kv_cache, "group_id", -1)
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

    input_lengths = attn_inputs.input_lengths
    cp_info = getattr(attn_inputs, "context_parallel_info", None)
    if cp_info is not None:
        actual_lengths = getattr(cp_info, "prefill_actual_input_lengths_cpu", None)
        if actual_lengths is not None and actual_lengths.numel() > 0:
            input_lengths = actual_lengths

    has_multi_region = (
        kv_cache is not None
        and bool(getattr(kv_cache, "layer_region_to_group_id", None))
        and bool(getattr(attn_inputs, "kv_cache_kernel_block_id_host_by_group", None))
    )
    if has_multi_region:
        return WriteCacheStoreOp(
            input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.kv_cache_kernel_block_id_host_by_group,
            attn_inputs.cache_store_inputs,
        )

    return WriteCacheStoreOp(
        input_lengths,
        attn_inputs.prefix_lengths,
        attn_inputs.kv_cache_block_id_host,
        attn_inputs.cache_store_inputs,
    )

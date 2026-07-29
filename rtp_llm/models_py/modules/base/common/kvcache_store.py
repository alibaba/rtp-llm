from typing import Optional

from torch import nn

from rtp_llm.ops.compute_ops import CacheStoreWriter, LayerKVCache, PyCacheStoreInputs


def write_layer_cache(
    cache_store_writer: CacheStoreWriter,
    cache_store_inputs: PyCacheStoreInputs,
    kv_cache: Optional[LayerKVCache],
) -> None:
    """Single write entry point shared by every cache-store trigger path.

    Both the FMHA WriteCacheStoreOp path and the linear-attention
    write_cache_store_if_needed helper funnel through here so the null-guard
    and the writer call cannot drift apart.
    """
    if kv_cache is None:
        return
    cache_store_writer.write(cache_store_inputs, kv_cache)


class WriteCacheStoreOp(nn.Module):
    def __init__(
        self,
        cache_store_writer: CacheStoreWriter,
        cache_store_inputs: PyCacheStoreInputs,
    ):
        super().__init__()
        self.cache_store_writer = cache_store_writer
        self.cache_store_inputs = cache_store_inputs

    def forward(self, kv_cache: Optional[LayerKVCache]) -> None:
        write_layer_cache(self.cache_store_writer, self.cache_store_inputs, kv_cache)

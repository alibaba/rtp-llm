from typing import Optional

from torch import nn

from rtp_llm.ops.compute_ops import CacheStoreWriter, LayerKVCache, PyCacheStoreInputs


class WriteCacheStoreOp(nn.Module):
    def __init__(
        self,
        cache_store_writer: Optional[CacheStoreWriter],
        cache_store_inputs: Optional[PyCacheStoreInputs],
    ):
        super().__init__()
        self.cache_store_writer = cache_store_writer
        self.cache_store_inputs = cache_store_inputs

    def forward(self, kv_cache: Optional[LayerKVCache]) -> None:
        if (
            self.cache_store_writer is None
            or self.cache_store_inputs is None
            or kv_cache is None
        ):
            return
        self.cache_store_writer.write(self.cache_store_inputs, kv_cache)

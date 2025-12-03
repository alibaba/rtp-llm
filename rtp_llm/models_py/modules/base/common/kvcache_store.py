from typing import Optional

import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.ops.compute_ops import KVCache, PyCacheStoreInputs


class WriteCacheStoreOp(nn.Module):
    def __init__(
        self,
        input_lengths: torch.Tensor,
        prefix_lengths: torch.Tensor,
        kv_cache_block_id_host: torch.Tensor,
        cache_store_inputs: Optional[PyCacheStoreInputs],
    ):
        super().__init__()
        self.input_lengths = input_lengths
        self.prefix_lengths = prefix_lengths
        self.kv_cache_block_id_host = kv_cache_block_id_host
        self.cache_store_inputs = cache_store_inputs

    def forward(self, kv_cache: Optional[KVCache]):
        # TODO: bad usage, should change
        compute_ops.write_cache_store(
            self.input_lengths,
            self.prefix_lengths,
            self.kv_cache_block_id_host,
            self.cache_store_inputs,
            kv_cache,
        )

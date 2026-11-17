from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention.common import reshape_paged_kv_cache
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.triton_kernels.attention.batch_invariant import (
    batch_invariant_paged_attention,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
)


class BatchInvariantAttention(FMHAImplBase):
    """Reuse RTP's RoPE and KV writers, with one attention reduction for both phases."""

    def __init__(self, config: AttentionConfigs, inputs: PyAttentionInputs):
        self.config = config
        self.block_table = inputs.kv_cache_kernel_block_id_device
        device = self.block_table.device
        if inputs.is_prefill:
            self.rope = FusedRopeKVCachePrefillOpQOut(config)
            self.request_ids = torch.repeat_interleave(
                torch.arange(inputs.input_lengths.numel(), device=device),
                inputs.input_lengths_device,
                output_size=int(inputs.input_lengths.sum()),
            )
            starts = inputs.cu_seqlens_device[self.request_ids]
            prefixes = inputs.prefix_lengths_device[self.request_ids]
            self.kv_lengths = (
                torch.arange(self.request_ids.numel(), device=device, dtype=torch.int32)
                - starts
                + prefixes
                + 1
            )
        else:
            self.rope = FusedRopeKVCacheDecodeOp(config)
            self.request_ids = torch.arange(
                inputs.sequence_lengths.numel(), device=device, dtype=torch.int32
            )
            self.kv_lengths = inputs.sequence_lengths_plus_1_device
        self.rope_params = self.rope.prepare(inputs)

    @staticmethod
    def support(config: AttentionConfigs, inputs: PyAttentionInputs) -> bool:
        return (
            config.is_causal
            and config.need_rope_kv_cache
            and not config.use_mla
            and config.kv_cache_dtype == KvCacheDataType.BASE
            and not inputs.is_target_verify
        )

    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[LayerKVCache], layer_idx: int = 0
    ) -> torch.Tensor:
        if kv_cache is None:
            raise ValueError("batch-invariant attention requires paged KV cache")
        config = self.config
        q = self.rope.forward(qkv, kv_cache, self.rope_params)
        q = q.reshape(-1, config.head_num, config.size_per_head)
        cache = reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            config.kv_head_num,
            config.kernel_tokens_per_block,
            config.size_per_head,
        )
        return batch_invariant_paged_attention(
            q,
            cache,
            self.block_table,
            self.request_ids,
            self.kv_lengths,
            config.size_per_head**-0.5,
        )

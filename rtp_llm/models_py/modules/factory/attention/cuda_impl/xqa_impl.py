from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.common import (
    apply_rope_and_kv_cache,
    create_params,
    create_write_cache_store_op,
    update_params_for_cuda_graph,
    write_to_cache_store,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    KVCache,
    PyAttentionInputs,
    XQAAttnOp,
)


class XQAImpl(FMHAImplBase):
    """XQA attention implementation."""

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        self.fmha_impl = XQAAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens
        self.fmha_params = None
        self.rope_params = None

        self.attn_inputs = attn_inputs
        self.write_cache_store_impl = create_write_cache_store_op(attn_inputs)
        self.fmha_params, self.rope_params = create_params(
            self.fmha_impl, self.rope_kvcache_impl, attn_inputs
        )

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """检查当前实现是否支持给定的输入。"""
        fmha_impl = XQAAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        """执行前向传播计算。"""
        # Apply RoPE and KV cache operations
        fmha_input = apply_rope_and_kv_cache(
            qkv, kv_cache, self.rope_kvcache_impl, self.rope_params, need_rope_kv_cache
        )
        
        # Write to cache store if needed
        write_to_cache_store(kv_cache, self.attn_inputs, self.write_cache_store_impl)
        
        # Run attention
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
        return res

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        update_params_for_cuda_graph(
            self.fmha_impl, self.rope_kvcache_impl,
            self.fmha_params, self.rope_params, attn_inputs
        )

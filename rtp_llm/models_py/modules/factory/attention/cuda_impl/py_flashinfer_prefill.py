from typing import Optional

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

from rtp_llm.models_py.modules.factory.attention.cuda_impl.common import (
    apply_rope_and_kv_cache,
    create_write_cache_store_op,
    write_to_cache_store,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQKVOut,
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    get_scalar_type,
)


class PyFlashinferPrefillImpl(FMHAImplBase):
    """PyFlashinfer Prefill implementation."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs
    ) -> None:
        # Initialize flashinfer prefill wrapper directly
        self.g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device="cuda",
        )
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            "NHD",
            backend="auto",
        )

        # Initialize rope and cache operations
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens
        self.fmha_params = None
        self.rope_params = None

        self.attn_inputs = attn_inputs
        self.write_cache_store_impl = create_write_cache_store_op(attn_inputs)
        
        # Prepare parameters
        self.fmha_params = self._prepare_fmha_params(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

    def _prepare_fmha_params(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        """Prepare FMHA parameters."""
        cu_seqlen_without_padding = attn_inputs.cu_seqlens[
            : attn_inputs.input_lengths.size(0) + 1
        ]
        self.prefill_wrapper.plan(
            cu_seqlen_without_padding,
            cu_seqlen_without_padding,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=True,
            q_data_type=get_scalar_type(attn_inputs.dtype),
        )
        return ParamsBase()

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if current implementation supports given inputs."""
        return True
    
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
        
        # Run attention directly using prefill_wrapper
        fmha_input = fmha_input.reshape(fmha_input.shape[0], -1)
        q, k, v = torch.split(
            fmha_input,
            [
                self.head_dim_qk * self.local_head_num,
                self.head_dim_qk * self.local_kv_head_num,
                self.head_dim_vo * self.local_kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        k = k.reshape(k.shape[0], self.local_kv_head_num, self.head_dim_qk)
        v = v.reshape(v.shape[0], self.local_kv_head_num, self.head_dim_vo)
        return self.prefill_wrapper.run(q, k, v)

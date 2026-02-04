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
    FusedRopeKVCachePrefillOpQKVOut,
    FusedRopeKVCachePrefillOpQOut,
    KVCache,
    PyAttentionInputs,
    TRTAttnOp,
    TRTPagedAttnOp,
    cuda_graph_copy_large2small,
    cuda_graph_copy_small2large,
)


class TRTMHAImpl(FMHAImplBase):
    """TRT MHA implementation with CUDA graph support."""

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        self.fmha_impl = TRTAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens
        self.fmha_params = None
        self.rope_params = None
        
        # Only TRTMHAImpl uses prefill_cuda_graph_copy_params
        self.prefill_cuda_graph_copy_params = attn_inputs.prefill_cuda_graph_copy_params
        
        self.attn_inputs = attn_inputs
        self.write_cache_store_impl = create_write_cache_store_op(attn_inputs)
        self.fmha_params, self.rope_params = create_params(
            self.fmha_impl, self.rope_kvcache_impl, attn_inputs
        )

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """检查当前实现是否支持给定的输入。"""
        fmha_impl = TRTAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        # Apply RoPE and KV cache operations
        fmha_input = apply_rope_and_kv_cache(
            qkv, kv_cache, self.rope_kvcache_impl, self.rope_params, need_rope_kv_cache
        )
        
        # Write to cache store if needed
        write_to_cache_store(kv_cache, self.attn_inputs, self.write_cache_store_impl)
        
        # CUDA graph copy logic specific to TRTMHAImpl
        if self.prefill_cuda_graph_copy_params:
            # Infer qkv_dim from fmha_input tensor shape
            qkv_dim = fmha_input.shape[1]
            total_len = (
                self.prefill_cuda_graph_copy_params.max_seq_len
                * self.prefill_cuda_graph_copy_params.max_batch_size
            )
            aligned_attn_buf = torch.zeros(
                (total_len, qkv_dim),
                dtype=fmha_input.dtype,
                device=fmha_input.device,
            )
            cuda_graph_copy_small2large(
                fmha_input,
                aligned_attn_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                qkv_dim,
                self.cu_seq_lens,
            )
            fmha_input = aligned_attn_buf
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
        if self.prefill_cuda_graph_copy_params:
            # Infer hidden_size from res tensor shape
            hidden_size = res.shape[1]
            compact_attn_buf = torch.zeros(
                (qkv.shape[0], hidden_size), dtype=res.dtype, device=res.device
            )
            cuda_graph_copy_large2small(
                res,
                compact_attn_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                hidden_size,
                self.cu_seq_lens,
            )
            res = compact_attn_buf
        return res

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        pass


class TRTPagedMHAImpl(FMHAImplBase):
    """TRT Paged MHA implementation."""

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        self.fmha_impl = TRTPagedAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
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
        fmha_impl = TRTPagedAttnOp(attn_configs)
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
        if not attn_inputs.is_prefill and (
            attn_inputs.prefix_lengths is None
            or attn_inputs.prefix_lengths.numel() == 0
        ):
            attn_inputs.prefix_lengths = torch.zeros_like(
                attn_inputs.input_lengths, device=attn_inputs.input_lengths.device
            )
        update_params_for_cuda_graph(
            self.fmha_impl, self.rope_kvcache_impl,
            self.fmha_params, self.rope_params, attn_inputs
        )

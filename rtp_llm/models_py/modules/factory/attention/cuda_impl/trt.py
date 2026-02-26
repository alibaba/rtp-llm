from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAType
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

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = TRTAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(
            attn_configs, max_seq_len
        )

        # Store input info
        self.attn_inputs = attn_inputs
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens

        # Only TRTMHAImpl uses prefill_cuda_graph_copy_params
        self.prefill_cuda_graph_copy_params = attn_inputs.prefill_cuda_graph_copy_params

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> bool:
        # Create temporary instance to check support
        fmha_impl = TRTAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

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

        # Execute FMHA forward
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

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = TRTPagedAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(
            attn_configs, max_seq_len
        )

        # Store input info
        self.attn_inputs = attn_inputs
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> bool:
        # Create temporary instance to check support
        fmha_impl = TRTPagedAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        if not attn_inputs.is_prefill and (
            attn_inputs.prefix_lengths is None
            or attn_inputs.prefix_lengths.numel() == 0
        ):
            attn_inputs.prefix_lengths = torch.zeros_like(
                attn_inputs.input_lengths, device=attn_inputs.input_lengths.device
            )
        common.update_trt_params(
            self.fmha_impl,
            self.rope_kvcache_impl,
            self.fmha_params,
            self.rope_params,
            attn_inputs,
        )

from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAPrefillImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOp,
    KVCache,
    PyAttentionInputs,
    TRTAttnOp,
    TRTPagedAttnOp,
    cuda_graph_copy_large2small,
    cuda_graph_copy_small2large,
)


class TRTMHAImpl(FMHAPrefillImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> None:
        super().__init__(
            TRTAttnOp(attn_configs),
            FusedRopeKVCachePrefillOp(attn_configs, max_seq_len),
            attn_inputs,
            max_seq_len,
        )
        # Only TRTMHAImpl uses prefill_cuda_graph_copy_params
        self.prefill_cuda_graph_copy_params = attn_inputs.prefill_cuda_graph_copy_params

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.TRT_V2

    def support_cuda_graph(self) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        if need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(
                qkv, self.fmha_type(), kv_cache, self.rope_params
            )
        else:
            fmha_input = qkv
        if (
            self.attn_inputs.is_prefill
            and self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)
        assert self.fmha_impl is not None
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


class TRTPagedMHAImpl(FMHAPrefillImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> None:
        super().__init__(
            TRTPagedAttnOp(attn_configs),
            FusedRopeKVCachePrefillOp(attn_configs, max_seq_len),
            attn_inputs,
            max_seq_len,
        )

    def create_params(self, attn_inputs: PyAttentionInputs):
        assert self.fmha_impl is not None
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        assert self.rope_kvcache_impl is not None
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PAGED_TRT_V2

    def support_cuda_graph(self) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        if not attn_inputs.is_prefill and (
            attn_inputs.prefix_lengths is None
            or attn_inputs.prefix_lengths.numel() == 0
        ):
            attn_inputs.prefix_lengths = torch.zeros_like(
                attn_inputs.input_lengths, device=attn_inputs.input_lengths.device
            )
        self._update_trt_params(attn_inputs)

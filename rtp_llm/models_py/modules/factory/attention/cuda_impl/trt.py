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


class TRTMHAImplBase(FMHAPrefillImplBase):
    """Base class for TRT attention implementations, encapsulating common TRT kernel logic"""

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        print(f"qkv input shape: {qkv.shape}")
        # Step 1: RoPE and KV cache
        if need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(
                qkv, self.fmha_type(), kv_cache, self.rope_params
            )
        else:
            fmha_input = qkv
        print(f"fmha input shape: {fmha_input.shape}")
        # Step 2: Write cache store
        if (
            self.attn_inputs.is_prefill
            and self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)

        # Step 3: CUDA graph preprocessing
        fmha_input = self._cuda_graph_preprocess(fmha_input, qkv)

        # Step 4: FMHA forward
        assert self.fmha_impl is not None
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

        # Step 5: CUDA graph postprocessing
        res = self._cuda_graph_postprocess(res, qkv)

        return res

    def _cuda_graph_preprocess(
        self, fmha_input: torch.Tensor, qkv: torch.Tensor
    ) -> torch.Tensor:
        """CUDA graph preprocessing: copy small buffer to aligned large buffer"""
        if (
            not hasattr(self, "prefill_cuda_graph_copy_params")
            or not self.prefill_cuda_graph_copy_params
        ):
            return fmha_input

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
        return aligned_attn_buf

    def _cuda_graph_postprocess(
        self, res: torch.Tensor, qkv: torch.Tensor
    ) -> torch.Tensor:
        """CUDA graph postprocessing: copy large buffer back to compact small buffer"""
        if (
            not hasattr(self, "prefill_cuda_graph_copy_params")
            or not self.prefill_cuda_graph_copy_params
        ):
            return res

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
        return compact_attn_buf


class TRTMHAImpl(TRTMHAImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            TRTAttnOp(attn_configs),
            FusedRopeKVCachePrefillOp(attn_configs),
            attn_inputs,
        )
        self.prefill_cuda_graph_copy_params = attn_inputs.prefill_cuda_graph_copy_params

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.TRT_V2

    def support_cuda_graph(self) -> bool:
        return True


class TRTPagedMHAImpl(TRTMHAImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            TRTPagedAttnOp(attn_configs),
            FusedRopeKVCachePrefillOp(attn_configs),
            attn_inputs,
        )
        self.prefill_cuda_graph_copy_params = attn_inputs.prefill_cuda_graph_copy_params

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PAGED_TRT_V2

    def support_cuda_graph(self) -> bool:
        return True

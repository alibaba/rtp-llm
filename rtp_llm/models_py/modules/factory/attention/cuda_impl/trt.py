import logging
from typing import Any, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAPrefillImplBase,
)
from rtp_llm.ops import FMHAType
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
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            TRTAttnOp(config.gpt_init_params),
            FusedRopeKVCachePrefillOp(config.gpt_init_params),
            attn_inputs,
        )
        # Only TRTMHAImpl uses prefill_cuda_graph_copy_params
        self.support_ = self.support_ and (config.use_mla == False)
        self.prefill_cuda_graph_copy_params = None
        if self.support_ and attn_inputs.enable_cuda_graph:
            batch_size = attn_inputs.input_lengths.shape[0]
            hidden_size = config.hidden_size
            self.prefill_cuda_graph_copy_params = cuda_graph_copy_large2small(
                batch_size, hidden_size, config.gpt_init_params.device
            )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.TRT

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        if (
            self.attn_inputs.enable_cuda_graph
            and self.prefill_cuda_graph_copy_params is not None
        ):
            qkv_small = self.prefill_cuda_graph_copy_params(qkv)
            attn_output_small = super().forward(qkv_small, kv_cache, need_rope_kv_cache)
            return cuda_graph_copy_small2large(attn_output_small, qkv.shape[0])
        else:
            return super().forward(qkv, kv_cache, need_rope_kv_cache)

    def support_cuda_graph(self) -> bool:
        return True


class TRTPagedMHAImpl(FMHAPrefillImplBase):

    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            TRTPagedAttnOp(config.gpt_init_params),
            FusedRopeKVCachePrefillOp(config.gpt_init_params),
            attn_inputs,
        )
        self.support_ = self.support_ and (config.use_mla == False)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.TRT_PAGED

    def support_cuda_graph(self) -> bool:
        return True

import logging
from typing import Any

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.mha.base import FMHAPrefillImplBase
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOp,
    PyAttentionInputs,
    TRTAttnOp,
    TRTPagedAttnOp,
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

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.TRT_V2

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

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PAGED_TRT_V2

    def support_cuda_graph(self) -> bool:
        return True

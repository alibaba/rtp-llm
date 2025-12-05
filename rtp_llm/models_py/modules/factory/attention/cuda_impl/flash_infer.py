import logging
from typing import Any

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
)
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import (
    FlashInferDecodeOp,
    FlashInferPrefillOp,
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOp,
    PyAttentionInputs,
)


class FlashInferPrefillImpl(FMHAPrefillImplBase):

    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferPrefillOp(config.gpt_init_params),
            FusedRopeKVCachePrefillOp(config.gpt_init_params),
            attn_inputs,
        )
        self.support_ = self.support_ and (config.use_mla == False)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True


class FlashInferDecodeImpl(FMHADecodeImplBase):

    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferDecodeOp(config.gpt_init_params),
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )
        self.support_ = self.support_ and (config.use_mla == False)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True

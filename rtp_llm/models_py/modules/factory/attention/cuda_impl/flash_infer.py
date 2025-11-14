import logging
from typing import Any

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAType
from rtp_llm.ops.compute_ops import (
    FlashInferDecodeOp,
    FlashInferPrefillOp,
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOp,
    PyAttentionInputs,
)


class FlashInferPrefillImpl(FMHAPrefillImplBase):

    def __init__(
        self, 
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferPrefillOp(attn_configs),
            FusedRopeKVCachePrefillOp(attn_configs),
            attn_inputs,
        )
        self.support_ = self.support_ and (not attn_configs.use_mla)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True


class FlashInferDecodeImpl(FMHADecodeImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferDecodeOp(attn_configs),
            FusedRopeKVCacheDecodeOp(attn_configs),
            attn_inputs,
        )
        self.support_ = self.support_ and (not attn_configs.use_mla)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True

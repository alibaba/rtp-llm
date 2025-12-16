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
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
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
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        self.seq_size_per_block = attn_configs.tokens_per_block
        super().__init__(
            FlashInferDecodeOp(attn_configs),
            FusedRopeKVCacheDecodeOp(attn_configs),
            attn_inputs,
        )
        self.support_ = self.support_ and (not attn_configs.use_mla)

    def create_params(self, attn_inputs: PyAttentionInputs):
        assert self.fmha_impl is not None
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        assert self.rope_kvcache_impl is not None
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        batch_size = attn_inputs.input_lengths.size(0)
        self.fmha_params.fill_params(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            batch_size,
            self.seq_size_per_block,
        )

        # Update rope params by copying offsets
        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)

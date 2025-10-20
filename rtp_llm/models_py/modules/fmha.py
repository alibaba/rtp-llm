import logging
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from librtp_compute_ops.rtp_llm_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOp,
    cuda_graph_copy_large2small,
    cuda_graph_copy_small2large,
)

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import KVCache, ParamsBase, PyAttentionInputs
from rtp_llm.utils.model_weight import W


class FMHAImplBase(object):
    fmha_impl: Any
    fmha_params: ParamsBase
    rope_params: Any
    rope_kvcache_impl: Any
    write_cache_store_impl: Any
    attn_inputs: PyAttentionInputs
    support_: bool = False

    def __init__(
        self,
        fmha_impl: Any,
        rope_kvcache_impl: Any,
        attn_inputs: PyAttentionInputs,
        init_params: bool = True,
    ) -> None:
        self.fmha_impl = fmha_impl
        self.prefill_cuda_graph_copy_params = attn_inputs.prefill_cuda_graph_copy_params
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens
        self.support_: bool = self.fmha_impl.support(attn_inputs)
        self.fmha_params = None
        self.rope_params = None
        self.write_cache_store_impl = None
        if self.support_ and init_params:
            self.rope_kvcache_impl = rope_kvcache_impl
            self.prepare(attn_inputs)
            self.attn_inputs = attn_inputs
            if self.attn_inputs.is_prefill and self.attn_inputs.cache_store_inputs:
                self.write_cache_store_impl = WriteCacheStoreOp(
                    self.attn_inputs.input_lengths,
                    self.attn_inputs.prefix_lengths,
                    self.attn_inputs.kv_cache_block_id_host,
                    self.attn_inputs.cache_store_inputs,
                )

    def forward(self, qkv: torch.Tensor, kv_cache: Optional[KVCache]) -> torch.Tensor:
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        fmha_input = self.rope_kvcache_impl.forward(
            qkv, self.fmha_type(), kv_cache, self.rope_params
        )
        if (
            self.attn_inputs.is_prefill
            and self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)
        assert self.fmha_impl is not None
        if self.prefill_cuda_graph_copy_params:
            cuda_graph_copy_small2large(
                fmha_input,
                self.prefill_cuda_graph_copy_params.aligned_attn_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                self.prefill_cuda_graph_copy_params.aligned_attn_buf.shape[1],
                self.cu_seq_lens,
            )
            fmha_input = self.prefill_cuda_graph_copy_params.aligned_attn_buf
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
        if self.prefill_cuda_graph_copy_params:
            cuda_graph_copy_large2small(
                res,
                self.prefill_cuda_graph_copy_params.compact_attn_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                self.prefill_cuda_graph_copy_params.hidden_size,
                self.cu_seq_lens,
            )
            res = self.prefill_cuda_graph_copy_params.compact_attn_buf
        return res

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.NONE

    def support(self):
        return self.support_

    def support_cuda_graph(self) -> bool:
        return False

    def prepare(self, attn_inputs: PyAttentionInputs):
        assert self.fmha_impl is not None
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        assert self.rope_kvcache_impl is not None
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)


class FMHAPrefillImplBase(FMHAImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        rope_kvcache_impl: Any,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        super().__init__(fmha_impl, rope_kvcache_impl, attn_inputs)


class FMHADecodeImplBase(FMHAImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        rope_kvcache_impl: Any,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        super().__init__(fmha_impl, rope_kvcache_impl, attn_inputs)


PREFILL_MHA_IMPS: List[type[FMHAPrefillImplBase]] = []
DECODE_MHA_IMPS: List[type[FMHADecodeImplBase]] = []

try:
    from rtp_llm.ops.compute_ops import FlashInferPrefillOp, FusedRopeKVCachePrefillOp

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

    PREFILL_MHA_IMPS.append(FlashInferPrefillImpl)

except ImportError:
    logging.info("FlashInferPrefillImpl not available, skipped.")


try:
    from rtp_llm.ops.compute_ops import FlashInferDecodeOp, FusedRopeKVCacheDecodeOp

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

    DECODE_MHA_IMPS.append(FlashInferDecodeImpl)

except ImportError:
    logging.info("FlashInferDecodeOp not available, skipped.")

try:
    from rtp_llm.ops.compute_ops import TRTAttnOp

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

    PREFILL_MHA_IMPS.append(TRTMHAImpl)
    # PREFILL_MHA_IMPS.insert(0, TRTMHAImpl)

except ImportError:
    logging.info("TRTMHAImpl not available, skipped.")


try:
    from rtp_llm.ops.compute_ops import FusedRopeKVCacheDecodeOp, XQAAttnOp

    class XQAImpl(FMHADecodeImplBase):

        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(
                XQAAttnOp(config.gpt_init_params),
                FusedRopeKVCacheDecodeOp(config.gpt_init_params),
                attn_inputs,
            )

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.XQA

        def support_cuda_graph(self) -> bool:
            return True

    DECODE_MHA_IMPS.append(XQAImpl)
except ImportError:
    logging.info("XQAAttnOp not available, skipped.")

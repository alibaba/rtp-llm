import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.mla import (
    MlaFlashInferDecodeOp,
    MlaFlashInferPrefillOp,
    MlaRotaryEmbeddingOp,
    TrtV2PrefillAttentionOp,
)
from rtp_llm.utils.model_weight import W

try:
    from librtp_compute_ops.rtp_llm_ops import (
        FusedRopeKVCacheDecodeOp,
        FusedRopeKVCachePrefillOp,
    )
except ImportError:
    logging.info("rope kv cache not available, skipped.")

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops import FMHAType, KVCache, ParamsBase, PyAttentionInputs


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
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
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

PREFILL_MLA_IMPS: List[type[FMHAPrefillImplBase]] = []
DECODE_MLA_IMPS: List[type[FMHADecodeImplBase]] = []

try:
    from librtp_compute_ops.rtp_llm_ops import FlashInferPrefillOp

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

    class MlaFlashInferPrefillImpl(FMHAPrefillImplBase):

        def __init__(
            self,
            config: GptInitModelParameters,
            attn_inputs: PyAttentionInputs,
            weights: List[Dict[str, torch.Tensor]],
            cos_sin_cache: torch.Tensor,
        ) -> None:

            super().__init__(
                MlaFlashInferPrefillOp(
                    config,
                    config.head_num,
                    config.kv_lora_rank,
                    config.rope_head_dim,
                    config.nope_head_dim,
                    config.seq_size_per_block,
                    config.softmax_extra_scale,
                    config.use_mla,
                    weights,
                ),
                # TrtV2PrefillAttentionOp(
                #     config,
                #     config.head_num,
                #     config.kv_lora_rank,
                #     config.rope_head_dim,
                #     config.nope_head_dim,
                #     config.use_mla,
                #     weights,
                # ),
                MlaRotaryEmbeddingOp(
                    head_size=config.nope_head_dim,
                    cos_sin_cache=cos_sin_cache,
                    kv_lora_rank=config.kv_lora_rank,
                    rope_head_dim=config.rope_head_dim,
                    token_per_block=config.seq_size_per_block,
                    is_neox_style=False,
                ),
                attn_inputs,
            )

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.FLASH_INFER

        def forward(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):
            assert self.rope_kvcache_impl is not None and self.rope_params is not None
            q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]
            self.rope_kvcache_impl.forward(
                q_pe, k_pe, compressed_kv, self.rope_params, kv_cache
            )

            if (
                self.attn_inputs.is_prefill
                and self.attn_inputs.cache_store_inputs
                and self.write_cache_store_impl is not None
            ):
                self.write_cache_store_impl(kv_cache)
            assert self.fmha_impl is not None
            res = self.fmha_impl.forward(
                q, compressed_kv, k_pe, self.fmha_params, layer_id
            )
            return res

    PREFILL_MLA_IMPS.append(MlaFlashInferPrefillImpl)

except ImportError:
    logging.info("FlashInferPrefillImpl not available, skipped.")


try:
    from librtp_compute_ops.rtp_llm_ops import FlashInferDecodeOp

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

    class MlaFlashInferDecodeImpl(FMHADecodeImplBase):

        def __init__(
            self,
            config: GptInitModelParameters,
            attn_inputs: PyAttentionInputs,
            weights: List[Dict[str, torch.Tensor]],
            cos_sin_cache: torch.Tensor,
        ) -> None:
            super().__init__(
                MlaFlashInferDecodeOp(
                    config.head_num,
                    config.kv_lora_rank,
                    config.rope_head_dim,
                    config.nope_head_dim,
                    config.seq_size_per_block,
                    config.softmax_extra_scale,
                    config.use_mla,
                    weights,
                ),
                MlaRotaryEmbeddingOp(
                    head_size=config.nope_head_dim,
                    cos_sin_cache=cos_sin_cache,
                    kv_lora_rank=config.kv_lora_rank,
                    rope_head_dim=config.rope_head_dim,
                    token_per_block=config.seq_size_per_block,
                    is_neox_style=False,
                ),
                attn_inputs,
            )

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.FLASH_INFER

        def forward(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):
            assert self.rope_kvcache_impl is not None and self.rope_params is not None
            q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]
            self.rope_kvcache_impl.forward(
                q_pe, k_pe, compressed_kv, self.rope_params, kv_cache
            )

            if (
                self.attn_inputs.is_prefill
                and self.attn_inputs.cache_store_inputs
                and self.write_cache_store_impl is not None
            ):
                self.write_cache_store_impl(kv_cache)
            q_nope, q_pe = torch.split(
                q,
                [self.fmha_impl.qk_nope_head_dim, self.fmha_impl.qk_rope_head_dim],
                dim=-1,
            )
            assert self.fmha_impl is not None
            res = self.fmha_impl.forward(
                q_nope, q_pe, kv_cache, self.fmha_params, layer_id
            )
            return res

    DECODE_MLA_IMPS.append(MlaFlashInferDecodeImpl)

except ImportError:
    logging.info("FlashInferDecodeOp not available, skipped.")

try:
    from librtp_compute_ops.rtp_llm_ops import TRTAttnOp

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
    from librtp_compute_ops.rtp_llm_ops import XQAAttnOp

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

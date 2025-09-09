import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.flashinfer_mla import (
    MlaFlashInferDecodeOp,
    MlaFlashInferPrefillOp,
)
from rtp_llm.models_py.modules.rotary_emb import MlaRotaryEmbeddingOp
from rtp_llm.utils.model_weight import W

try:
    from libth_transformer.rtp_llm_ops import (
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
    from libth_transformer.rtp_llm_ops import FlashInferPrefillOp

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
            use_torch: bool = True,
        ) -> None:

            super().__init__(
                MlaFlashInferPrefillOp(
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
            self.support_ = self.support_ and use_torch

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.FLASH_INFER

        def forward(
            self,
            q: torch.Tensor,
            q_pe: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):
            assert self.rope_kvcache_impl is not None and self.rope_params is not None

            q_pe, k_pe = self.rope_kvcache_impl.forward(
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
                q, q_pe, compressed_kv, k_pe, self.fmha_params, layer_id
            )
            return res

    PREFILL_MLA_IMPS.append(MlaFlashInferPrefillImpl)

    from libth_transformer.rtp_llm_ops import MlaContextAttentionOp

    class MlaFlashInferPrefillCppImpl(FMHAPrefillImplBase):

        def __init__(
            self,
            config: GptInitModelParameters,
            attn_inputs: PyAttentionInputs,
            weights: List[Dict[str, torch.Tensor]],
            cos_sin_cache: torch.Tensor,
            use_torch: bool = False,
        ) -> None:
            super().__init__(
                MlaContextAttentionOp(config.gpt_init_params),
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
            self.weights = weights
            self.num_heads = config.head_num
            self.kv_lora_rank = config.kv_lora_rank
            self.qk_rope_head_dim = config.rope_head_dim
            self.qk_nope_head_dim = config.nope_head_dim
            self.support_ = self.support_ and not use_torch
            logging.info(
                f"using MlaFlashInferPrefillCppImpl support_ is {self.support_}"
            )

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.FLASH_INFER

        def forward(
            self,
            q: torch.Tensor,
            q_pe: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):
            assert self.rope_kvcache_impl is not None and self.rope_params is not None

            q_pe, k_pe = self.rope_kvcache_impl.forward(
                q_pe, k_pe, compressed_kv, self.rope_params, kv_cache
            )

            if (
                self.attn_inputs.is_prefill
                and self.attn_inputs.cache_store_inputs
                and self.write_cache_store_impl is not None
            ):
                self.write_cache_store_impl(kv_cache)

            assert self.fmha_impl is not None
            kv_offset = self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
            fused_q_input_t = torch.cat((q, q_pe), dim=-1).view(-1, kv_offset)
            k_nope_weight = self.weights[layer_id].get(W.mla_k_nope_w, None)
            v_weight = self.weights[layer_id].get(W.mla_v_w, None)

            # k_nope = F.linear(compressed_kv, k_nope_weight.transpose(0, 1), None)
            # k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
            # value_states = F.linear(compressed_kv, v_weight.transpose(0, 1), None)
            # value_states = value_states.view(-1, self.num_heads, self.qk_nope_head_dim)

            res = self.fmha_impl.forward(
                fused_q_input_t,
                compressed_kv,
                k_pe,
                kv_offset,
                self.fmha_params,
                k_nope_weight,
                v_weight,
            )
            return res

    PREFILL_MLA_IMPS.append(MlaFlashInferPrefillCppImpl)
except ImportError:
    logging.info("FlashInferPrefillImpl not available, skipped.")


try:
    from libth_transformer.rtp_llm_ops import FlashInferDecodeOp

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

    class MlaFlashInferDecodeImpl(FMHAPrefillImplBase):

        def __init__(
            self,
            config: GptInitModelParameters,
            attn_inputs: PyAttentionInputs,
            weights: List[Dict[str, torch.Tensor]],
            cos_sin_cache: torch.Tensor,
            use_torch: bool = True,
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
            self.support_ = self.support_ and use_torch

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.FLASH_INFER

        def forward(
            self,
            q: torch.Tensor,
            q_pe: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):
            assert self.rope_kvcache_impl is not None and self.rope_params is not None

            q_pe, k_pe = self.rope_kvcache_impl.forward(
                q_pe, k_pe, compressed_kv, self.rope_params, kv_cache
            )

            if (
                self.attn_inputs.is_prefill
                and self.attn_inputs.cache_store_inputs
                and self.write_cache_store_impl is not None
            ):
                self.write_cache_store_impl(kv_cache)

            assert self.fmha_impl is not None
            res = self.fmha_impl.forward(q, q_pe, kv_cache, self.fmha_params, layer_id)
            return res

    DECODE_MLA_IMPS.append(MlaFlashInferDecodeImpl)

    from libth_transformer.rtp_llm_ops import MlaAbsorbAttentionOp

    class MlaFlashInferDecodeCppImpl(FMHAPrefillImplBase):

        def __init__(
            self,
            config: GptInitModelParameters,
            attn_inputs: PyAttentionInputs,
            weights: List[Dict[str, torch.Tensor]],
            cos_sin_cache: torch.Tensor,
            use_torch: bool = False,
        ) -> None:
            super().__init__(
                MlaAbsorbAttentionOp(config.gpt_init_params),
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
            self.weights = weights
            self.num_heads = config.head_num
            self.kv_lora_rank = config.kv_lora_rank
            self.qk_rope_head_dim = config.rope_head_dim
            self.qk_nope_head_dim = config.nope_head_dim
            self.support_ = self.support_ and not use_torch
            logging.info(
                f"using MlaFlashInferDecodeCppImpl support_ is {self.support_}"
            )

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.FLASH_INFER

        def forward(
            self,
            q: torch.Tensor,
            q_pe: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):
            assert self.rope_kvcache_impl is not None and self.rope_params is not None

            q_pe, k_pe = self.rope_kvcache_impl.forward(
                q_pe, k_pe, compressed_kv, self.rope_params, kv_cache
            )

            if (
                self.attn_inputs.is_prefill
                and self.attn_inputs.cache_store_inputs
                and self.write_cache_store_impl is not None
            ):
                self.write_cache_store_impl(kv_cache)

            assert self.fmha_impl is not None
            q = q.view(-1, self.num_heads, self.qk_nope_head_dim)
            q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)
            fuse_q = torch.cat((q, q_pe), dim=-1)
            fused_q_input_t = (
                torch.empty(
                    q.shape[0],
                    self.num_heads,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                )
                .to(fuse_q.device)
                .to(fuse_q.dtype)
            )
            fused_q_input_t[:, :, self.kv_lora_rank :] = q_pe
            k_weight = self.weights[layer_id].get(W.mla_kc, None)
            v_weight = self.weights[layer_id].get(W.mla_vc, None)
            res = self.fmha_impl.forward(
                fuse_q, fused_q_input_t, kv_cache, self.fmha_params, k_weight, v_weight
            )
            return res

    DECODE_MLA_IMPS.append(MlaFlashInferDecodeCppImpl)
except ImportError:
    logging.info("FlashInferDecodeOp not available, skipped.")

try:
    from libth_transformer.rtp_llm_ops import TRTAttnOp

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
    from libth_transformer.rtp_llm_ops import XQAAttnOp

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

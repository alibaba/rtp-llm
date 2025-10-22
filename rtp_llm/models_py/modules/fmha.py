import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.mla import (
    MlaFlashInferDecodeOp,
    MlaFlashInferPrefillOp,
    MlaRotaryEmbeddingOp,
    TrtV2PrefillAttentionOp,
)
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

    class MlaFlashInferPrefillImpl(FMHAPrefillImplBase):

        def __init__(
            self,
            config: GptInitModelParameters,
            attn_inputs: PyAttentionInputs,
            weights: List[Dict[str, torch.Tensor]],
            cos_sin_cache: torch.Tensor,
            absorb_opt_len: int = 192,
            use_trt_fmha: bool = False,
        ) -> None:
            # trt prefill not support reuse cache yet
            self.use_trt_fmha = use_trt_fmha
            self.fmha_impl = self._get_fmha_impl(
                config,
                attn_inputs.input_lengths.sum().item(),
                weights,
                absorb_opt_len,
                use_trt_fmha,
            )

            super().__init__(
                self.fmha_impl,
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
            self.absorb_opt_len = absorb_opt_len
            self.aborb_fmha = MlaFlashInferDecodeOp(
                config.head_num // config.tp_size,
                config.kv_lora_rank,
                config.rope_head_dim,
                config.nope_head_dim,
                config.seq_size_per_block,
                config.softmax_extra_scale,
                config.use_mla,
                weights,
            )

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.FLASH_INFER

        def _get_fmha_impl(
            self,
            config: GptInitModelParameters,
            q_len: int,
            weights: List[Dict[str, torch.Tensor]],
            absorb_opt_len: int,
            use_trt_fmha: bool,
        ):
            # if q_len < absorb_opt_len:
            #     return MlaFlashInferDecodeOp(
            #         config.head_num,
            #         config.kv_lora_rank,
            #         config.rope_head_dim,
            #         config.nope_head_dim,
            #         config.seq_size_per_block,
            #         config.softmax_extra_scale,
            #         config.use_mla,
            #         weights,
            #     )
            # if use_trt_fmha:
            #     return TrtV2PrefillAttentionOp(
            #         config,
            #         config.head_num,
            #         config.kv_lora_rank,
            #         config.rope_head_dim,
            #         config.nope_head_dim,
            #         config.use_mla,
            #         weights,
            #     )
            return MlaFlashInferPrefillOp(
                config,
                config.head_num // config.tp_size,
                config.kv_lora_rank,
                config.rope_head_dim,
                config.nope_head_dim,
                config.seq_size_per_block,
                config.softmax_extra_scale,
                config.use_mla,
                weights,
            )

        def compute_prefill_context(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):
            """Compute prefill context with optimized cache reuse logic."""
            # Early return for short sequences using absorb operation
            # if self.use_trt_fmha:
            #     return self.fmha_impl.forward(
            #         q, compressed_kv, k_pe, self.fmha_params, layer_id
            #     )
            num_blocks = self.fmha_params.reuse_cache_page_indice.size(0)
            if q.size(0) < self.absorb_opt_len and num_blocks > 0:
                return self._handle_short_sequence(q, kv_cache, layer_id)
            else:
                return self._handle_long_sequence(
                    q, compressed_kv, k_pe, kv_cache, layer_id
                )

        def _handle_long_sequence(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
            layer_id: int,
        ):
            """Handle long sequences using cache reuse operation."""
            # Handle cache reuse for longer sequences
            # if self.use_trt_fmha:
            #     return self.fmha_impl.forward(
            #         q, compressed_kv, k_pe, self.fmha_params, layer_id
            #     )
            compressed_kv, k_pe = self._reuse_kv_cache_indexed_batched(
                compressed_kv, k_pe, kv_cache
            )

            return self.fmha_impl.forward(
                q, compressed_kv, k_pe, self.fmha_params, layer_id
            )

        def _handle_short_sequence(
            self, q: torch.Tensor, kv_cache: Optional[KVCache], layer_id: int
        ) -> torch.Tensor:
            """Handle short sequences using absorb operation."""
            # Update page_indptr for short sequences

            # Split query into nope and pe components
            q_nope, q_pe = torch.split(
                q,
                [self.aborb_fmha.qk_nope_head_dim, self.aborb_fmha.qk_rope_head_dim],
                dim=-1,
            )

            return self.aborb_fmha.forward(
                q_nope, q_pe, kv_cache, self.fmha_params, layer_id
            )

        def _reuse_kv_cache_indexed_batched(
            self,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            kv_cache: Optional[KVCache],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """使用索引操作的优化版本 - 根据kv_len和q_len的差值确定concat位置"""

            # 获取参数
            reuse_cache_page_indice = self.fmha_params.reuse_cache_page_indice  # [5, 3]
            num_blocks = reuse_cache_page_indice.size(0)  # 2

            if num_blocks == 0:
                return compressed_kv, k_pe

            compressed_kv_dim = compressed_kv.size(1)
            qo_indptr = self.fmha_params.qo_indptr  # [0, 17, 29, 47, 63]

            # 准备结果tensor
            batch_reuse_info = self.fmha_params.batch_reuse_info_vec.cpu().tolist()
            qo_indptr_list = qo_indptr.cpu().tolist()
            total_reuse_len = sum(info[1] for info in batch_reuse_info)
            if total_reuse_len == 0:
                return compressed_kv, k_pe

            # 创建最终的tensor
            final_compressed_kv = torch.empty(
                (compressed_kv.size(0) + total_reuse_len, compressed_kv.size(1)),
                dtype=compressed_kv.dtype,
                device=compressed_kv.device,
            )
            final_k_pe = torch.empty(
                (k_pe.size(0) + total_reuse_len, k_pe.size(1)),
                dtype=k_pe.dtype,
                device=k_pe.device,
            )

            # 按batch处理，将reuse cache和compressed_kv按正确位置concat
            compressed_kv_offset = 0
            final_offset = 0

            for (
                batch_idx,
                reuse_len,
                block_start_idx,
                blocks_needed,
            ) in batch_reuse_info:
                batch_q_len = qo_indptr_list[batch_idx + 1] - qo_indptr_list[batch_idx]

                if reuse_len > 0:
                    # 获取这个batch需要的reuse blocks
                    batch_cache_indices = reuse_cache_page_indice[
                        block_start_idx : block_start_idx + blocks_needed
                    ]

                    # 从kv_cache中获取对应的blocks
                    batch_cache_blocks = kv_cache.k_cache_base[batch_cache_indices]
                    batch_cache_blocks = batch_cache_blocks.view(
                        -1, batch_cache_blocks.size(-1)
                    )

                    # 将reuse cache放到最终tensor的前面部分
                    final_compressed_kv[final_offset : final_offset + reuse_len] = (
                        batch_cache_blocks[:, :compressed_kv_dim]
                    )
                    final_k_pe[final_offset : final_offset + reuse_len] = (
                        batch_cache_blocks[:, compressed_kv_dim:]
                    )
                    final_offset += reuse_len

                # 将当前batch的compressed_kv放到对应位置
                batch_compressed_kv_start = compressed_kv_offset
                batch_compressed_kv_end = compressed_kv_offset + batch_q_len

                final_compressed_kv[final_offset : final_offset + batch_q_len] = (
                    compressed_kv[batch_compressed_kv_start:batch_compressed_kv_end]
                )
                final_k_pe[final_offset : final_offset + batch_q_len] = k_pe[
                    batch_compressed_kv_start:batch_compressed_kv_end
                ]

                final_offset += batch_q_len
                compressed_kv_offset += batch_q_len

            return final_compressed_kv, final_k_pe

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
            return self.compute_prefill_context(
                q, compressed_kv, k_pe, kv_cache, layer_id
            )

    PREFILL_MLA_IMPS.append(MlaFlashInferPrefillImpl)

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
                    config.head_num // config.tp_size,
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

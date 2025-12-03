import logging
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
)
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs

from .flashinfer_mla import (
    MlaFlashInferDecodeOp,
    MlaFlashInferPrefillOp,
    warmup_flashinfer_python,
)
from .rotary_emb import MlaRotaryEmbeddingOp


class MlaFlashInferPrefillImpl(FMHAPrefillImplBase):
    def __init__(
        self,
        config: GptInitModelParameters,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        absorb_opt_len: int = 1024,
        use_trt_fmha: bool = False,
    ) -> None:
        # trt prefill not support reuse cache yet
        super().__init__(
            MlaFlashInferPrefillOp(
                config,
                config.head_num // config.tp_size,
                config.kv_lora_rank,
                config.rope_head_dim,
                config.nope_head_dim,
                config.seq_size_per_block,
                config.softmax_extra_scale,
                config.use_mla,
                weights,
                use_trt_fmha,
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
        self.warm_up = config.warm_up
        self.has_reuse_cache = False
        if attn_inputs.prefix_lengths is not None:
            self.has_reuse_cache = attn_inputs.prefix_lengths.max().item() > 0

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
        self.aborb_fmha.prepare(attn_inputs)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def compute_prefill_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
    ):
        """Compute prefill context with optimized cache reuse logic."""
        if self.warm_up:
            self.warm_up = False
            warmup_flashinfer_python()

        if q.size(0) < self.absorb_opt_len and self.has_reuse_cache:
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
        return self.fmha_impl.forward(
            q, compressed_kv, k_pe, kv_cache, self.fmha_params, layer_id
        )

    def _handle_short_sequence(
        self, q: torch.Tensor, kv_cache: Optional[KVCache], layer_id: int
    ) -> torch.Tensor:
        """Handle short sequences using absorb operation."""
        # Split query into nope and pe components
        q_nope, q_pe = torch.split(
            q,
            [self.aborb_fmha.qk_nope_head_dim, self.aborb_fmha.qk_rope_head_dim],
            dim=-1,
        )

        return self.aborb_fmha.forward(
            q_nope, q_pe, kv_cache, self.fmha_params, layer_id
        )

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
        return self.compute_prefill_context(q, compressed_kv, k_pe, kv_cache, layer_id)


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
        res = self.fmha_impl.forward(q_nope, q_pe, kv_cache, self.fmha_params, layer_id)
        return res

import logging
import os
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops

from .flashinfer_mla import (
    MlaFlashInferDecodeOp,
    MlaFlashInferPrefillOp,
    check_attention_inputs,
    warmup_flashinfer_python,
)
from .rope_emb_new import NewMlaRotaryEmbeddingOp, NewMlaRotaryEmbeddingParams
from .rotary_emb import MlaRotaryEmbeddingOp


class MlaFlashInferImplBase(object):

    def __init__(
        self,
        fmha_impl: Any,
        rope_kvcache_impl: Any,
        attn_inputs: PyAttentionInputs,
        seq_size_per_block: int,
        is_cuda_graph: bool = False,
    ) -> None:
        warmup_flashinfer_python()
        self.seq_size_per_block = seq_size_per_block
        self.fmha_impl = fmha_impl
        self.fmha_params = None
        self.rope_params = None
        self.write_cache_store_impl = None
        self.support_: bool = self.fmha_impl.support(attn_inputs)
        if self.support_:
            self.rope_kvcache_impl = rope_kvcache_impl
            self.attn_inputs = attn_inputs
            if self.attn_inputs.is_prefill and self.attn_inputs.cache_store_inputs:
                self.write_cache_store_impl = WriteCacheStoreOp(
                    self.attn_inputs.input_lengths,
                    self.attn_inputs.prefix_lengths,
                    self.attn_inputs.kv_cache_block_id_host,
                    self.attn_inputs.cache_store_inputs,
                )
            self.create_params(attn_inputs)
            if attn_inputs.is_cuda_graph is False:
                self.prepare(attn_inputs)

    def create_params(self, attn_inputs: PyAttentionInputs):
        if self.support_ and self.fmha_impl is not None:
            self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
            self.rope_params = self.fmha_params

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.NONE

    def support(self):
        return self.support_

    def support_cuda_graph(self) -> bool:
        return False

    def prepare(self, attn_inputs: PyAttentionInputs):
        """Update fmha_params for prepare or CUDA Graph replay.

        Note: fmha_params is initialized in __init__, this method only updates it.
        """
        assert self.fmha_impl is not None
        assert (
            self.fmha_params is not None
        ), "fmha_params should be initialized in __init__"
        check_attention_inputs(attn_inputs)
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.seq_size_per_block,
        )
        self.fmha_impl.plan(self.fmha_params)
        self.rope_params = self.fmha_params

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
        res = self.fmha_impl.forward(q_nope, q_pe, kv_cache, layer_id)
        return res


class MlaFlashInferPrefillImpl(MlaFlashInferImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
    ) -> None:
        super().__init__(
            MlaFlashInferPrefillOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                attn_configs.is_sparse,
                attn_configs.indexer_topk,
                weights,
                quant_config,
            ),
            NewMlaRotaryEmbeddingOp(
                head_size=attn_configs.nope_head_dim,
                cos_sin_cache=cos_sin_cache,
                kv_lora_rank=attn_configs.kv_lora_rank,
                rope_head_dim=attn_configs.rope_head_dim,
                token_per_block=attn_configs.tokens_per_block,
                is_neox_style=False,
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            ),
            attn_inputs,
            attn_configs.tokens_per_block,
            is_cuda_graph,
        )
        self.has_reuse_cache = False
        if attn_inputs.prefix_lengths is not None:
            self.has_reuse_cache = attn_inputs.prefix_lengths.max().item() > 0

        self.absorb_opt_len = (
            fmha_config.absorb_opt_len if fmha_config is not None else 1024
        )
        q_len = attn_inputs.input_lengths.sum().item()
        if q_len < self.absorb_opt_len and self.has_reuse_cache:
            self.absorb_fmha = MlaFlashInferDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                weights,
            )
            self.absorb_fmha.plan(self.fmha_params)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASHINFER_PREFILL

    def create_params(self, attn_inputs: PyAttentionInputs):
        super().create_params(attn_inputs)
        self.indexer_params = rtp_llm_ops.IndexerParams()

    def prepare(self, attn_inputs: PyAttentionInputs):
        super().prepare(attn_inputs)
        self.indexer_params.fill_params(attn_inputs, self.seq_size_per_block)
        self.rope_params = NewMlaRotaryEmbeddingParams(
            self.fmha_params, self.indexer_params
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
        return self.fmha_impl.forward(q, compressed_kv, k_pe, kv_cache, layer_id)

    def _handle_short_sequence(
        self, q: torch.Tensor, kv_cache: Optional[KVCache], layer_id: int
    ) -> torch.Tensor:
        """Handle short sequences using absorb operation."""
        # Split query into nope and pe components
        q_nope, q_pe = torch.split(
            q,
            [self.absorb_fmha.qk_nope_head_dim, self.absorb_fmha.qk_rope_head_dim],
            dim=-1,
        )

        return self.absorb_fmha.forward(q_nope, q_pe, kv_cache, layer_id)

    def compute_prefill_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
    ):
        """Compute prefill context with optimized cache reuse logic."""

        if q.size(0) < self.absorb_opt_len and self.has_reuse_cache:
            return self._handle_short_sequence(q, kv_cache, layer_id)
        else:
            return self._handle_long_sequence(
                q, compressed_kv, k_pe, kv_cache, layer_id
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


class MlaFlashInferDecodeImpl(MlaFlashInferImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
    ) -> None:
        super().__init__(
            MlaFlashInferDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                attn_configs.is_sparse,
                weights,
                max_bs=attn_inputs.sequence_lengths.size(0),
                max_context_len=max_seq_len,
                num_tokens=attn_inputs.sequence_lengths.sum().item(),
                is_cuda_graph=is_cuda_graph,
            ),
            MlaRotaryEmbeddingOp(
                head_size=attn_configs.nope_head_dim,
                cos_sin_cache=cos_sin_cache,
                kv_lora_rank=attn_configs.kv_lora_rank,
                rope_head_dim=attn_configs.rope_head_dim,
                token_per_block=attn_configs.tokens_per_block,
                is_neox_style=False,
            ),
            attn_inputs,
            attn_configs.tokens_per_block,
            is_cuda_graph,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASHINFER_DECODE

    def support_cuda_graph(self) -> bool:
        return True

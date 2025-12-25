import logging
import os
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs

from .flashinfer_mla import (
    MlaFlashInferDecodeOp,
    MlaFlashInferPrefillOp,
    warmup_flashinfer_python,
)
from .rotary_emb import MlaRotaryEmbeddingOp

warm_up_done = False


class MlaFlashInferPrefillImpl(FMHAPrefillImplBase):
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
    ) -> None:
        # trt prefill not support reuse cache yet
        super().__init__(
            MlaFlashInferPrefillOp(
                attn_configs,
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                weights,
                use_trt_fmha,
                quant_config,
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
        )
        self.rope_params = self.fmha_params
        self.has_reuse_cache = False
        if attn_inputs.prefix_lengths is not None:
            self.has_reuse_cache = attn_inputs.prefix_lengths.max().item() > 0

        self.absorb_opt_len = (
            fmha_config.absorb_opt_len if fmha_config is not None else 1024
        )
        q_len = attn_inputs.input_lengths.sum().item()
        self.absorb_fmha = None
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

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def prepare(self, attn_inputs: PyAttentionInputs):
        super().prepare(attn_inputs)
        self.rope_params = self.fmha_params

        if self.absorb_fmha is not None:
            self.absorb_fmha.init_mla_wrapper(self.fmha_params)
            self.absorb_fmha.plan(self.fmha_params)

    def compute_prefill_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
    ):
        """Compute prefill context with optimized cache reuse logic."""
        global warm_up_done
        if not warm_up_done:
            warmup_flashinfer_python()
            warm_up_done = True

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
            [self.absorb_fmha.qk_nope_head_dim, self.absorb_fmha.qk_rope_head_dim],
            dim=-1,
        )

        return self.absorb_fmha.forward(q_nope, q_pe, kv_cache, layer_id)

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
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
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
                weights,
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
        )
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.bs = attn_inputs.input_lengths.size(0)
        self.max_context_len = max_seq_len

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs, use_cuda_graph: bool = False):
        assert self.fmha_impl is not None
        self.fmha_params = self.fmha_impl.prepare(attn_inputs, use_cuda_graph)
        self.rope_params = self.fmha_params

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.fmha_impl.cuda_graph_kv_indices = torch.empty(
            (
                (self.max_context_len + self.seq_size_per_block - 1)
                // self.seq_size_per_block
            )
            * self.bs,
            dtype=torch.int32,
            device="cuda",
        )
        self.rope_kvcache_impl.cuda_graph_kv_indices = (
            self.fmha_impl.cuda_graph_kv_indices
        )
        self.prepare(attn_inputs, True)

    def prepare_replay(self, attn_inputs: PyAttentionInputs):
        assert self.fmha_impl is not None
        batch_size = attn_inputs.input_lengths.size(0)
        self.fmha_params.fill_params(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            batch_size,
            self.seq_size_per_block,
        )
        self.fmha_impl.plan(self.fmha_params)

        assert self.rope_kvcache_impl is not None
        self.rope_params.positions_d.copy_(
            self.fmha_params.positions_d, non_blocking=True
        )
        self.rope_params.batch_indice_d.copy_(
            self.fmha_params.batch_indice_d, non_blocking=True
        )
        self.rope_kvcache_impl.cuda_graph_kv_indices[
            : len(self.fmha_params.page_indice_d)
        ].copy_(self.fmha_params.page_indice_d, non_blocking=True)
        self.rope_params.decode_page_indptr_d.copy_(
            self.fmha_params.decode_page_indptr_d, non_blocking=True
        )
        self.rope_params.paged_kv_last_page_len_d.copy_(
            self.fmha_params.paged_kv_last_page_len_d, non_blocking=True
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
        q_nope, q_pe = torch.split(
            q,
            [self.fmha_impl.qk_nope_head_dim, self.fmha_impl.qk_rope_head_dim],
            dim=-1,
        )
        assert self.fmha_impl is not None
        res = self.fmha_impl.forward(q_nope, q_pe, kv_cache, layer_id)
        return res

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
        is_cuda_graph: bool = False,
    ) -> None:
        # trt prefill not support reuse cache yet
        warmup_flashinfer_python()
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
            self.absorb_fmha.plan(self.fmha_params)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def prepare(self, attn_inputs: PyAttentionInputs):
        super().prepare(attn_inputs)
        self.rope_params = self.fmha_params

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
        is_cuda_graph: bool = False,
    ) -> None:
        warmup_flashinfer_python()
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
        )
        self.seq_size_per_block = attn_configs.tokens_per_block

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        """Unified prepare method supporting initial preparation and replay.

        Automatically detects whether this is first-time preparation or replay
        based on whether fmha_params exists.
        """
        assert self.fmha_impl is not None

        # Setup shared kv_indices_d reference (needed for both init and replay)
        self.rope_kvcache_impl.kv_indices_d = self.fmha_impl.kv_indices_d

        # Detect if this is first call or replay
        is_first_call = self.fmha_params is None

        if is_first_call:
            # First-time: create new params
            self.fmha_params = self.fmha_impl.prepare(attn_inputs)
            # Share reference: rope_params and fmha_params point to same object
            self.rope_params = self.fmha_params
        else:
            # Replay: update existing params
            self._update_params(attn_inputs)

        self.rope_kvcache_impl.kv_indices_d[
            : len(self.fmha_params.page_indice_d)
        ].copy_(self.fmha_params.page_indice_d, non_blocking=True)

    def _update_params(self, attn_inputs: PyAttentionInputs):
        """Update existing fmha_params for CUDA Graph replay.

        Note: Since rope_params and fmha_params share the same object reference,
        updating fmha_params automatically reflects in rope_params.
        """
        batch_size = attn_inputs.input_lengths.size(0)
        self.fmha_params.fill_params(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            batch_size,
            self.seq_size_per_block,
        )
        self.fmha_impl.plan(self.fmha_params)

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

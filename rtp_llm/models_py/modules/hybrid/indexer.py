from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.models_py.modules import IndexerOp, LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import (
    AttentionConfigs,
    CPProcessorType,
    HWKernelConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import KVCache, rtp_llm_ops
from rtp_llm.utils.model_weight import W


class Indexer(nn.Module):
    """
    Indexer for DeepSeek-V3.2 DSA (DeepSeek Sparse Attention) mechanism.
    Adapted from sglang's Indexer implementation.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        parallelism_config: Optional[ParallelismConfig] = None,
        scale_fmt: Optional[str] = "none",
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.index_n_heads = attn_config.indexer_head_num
        self.index_head_dim = attn_config.indexer_head_dim
        self.index_topk = attn_config.indexer_topk

        self.rope_head_dim = attn_config.rope_head_dim
        self.block_size = 128  # quantization block size (128)
        self.head_kv = 1
        self.scale_fmt = scale_fmt  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.blocksize = attn_config.tokens_per_block  # page size, typically 64
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2
        self.is_neox_style = attn_config.rope_config.indexer_is_neox_style
        self.parallelism_config = parallelism_config

        cp = parallelism_config.prefill_cp_config
        self._is_cp = cp.is_enabled()
        self._is_roundrobin = (
            self._is_cp and cp.processor_type == CPProcessorType.ROUND_ROBIN
        )

        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_qb_w,
            W.mla_indexer_qb_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_k_w,
            W.mla_indexer_k_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.k_norm = LayerNorm(
            weights[W.mla_indexer_k_norm_w],
            weights[W.mla_indexer_k_norm_b],
            eps=layernorm_eps,
        )

        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.cos_sin_cache = global_weights[W.rope_cos_sin_cache]

        # Initialize IndexerOp for low-level operations
        self.indexer_op = IndexerOp(
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            rope_head_dim=self.rope_head_dim,
            cos_sin_cache=self.cos_sin_cache,
            blocksize=self.blocksize,
            block_size=self.block_size,
            scale_fmt=self.scale_fmt,
            is_neox_style=self.is_neox_style,
        )

    def _is_cp_prefill(self) -> bool:
        """Check if any form of CP is enabled (zigzag or round-robin)."""
        return self._is_cp

    # TODO: fuse kernel here
    def _get_logits_head_gate(
        self, x: torch.Tensor, q_scale: torch.Tensor
    ) -> torch.Tensor:
        x = x.float()
        weights = self.weights_proj(x)
        scale = self.softmax_scale * self.weights_scale
        weights = weights.unsqueeze(-1) * q_scale * scale
        return weights

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        flashmla_params: Any,
        attention_inputs: Any,
    ):
        # Linear projections
        q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        k = self.wk(x)
        k = self.k_norm(k)

        # Apply RoPE and Hadamard transform
        if self._is_cp:
            query, key = self.indexer_op.apply_rope_and_rotate_q_k_cp(
                q,
                k,
                positions=flashmla_params.positions_d,
            )
        else:
            query, key = self.indexer_op.apply_rope_and_rotate_q_k(
                q, k, flashmla_params.positions_d
            )
        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        flashmla_params: Any,
    ):
        k = self.wk(x)
        k = self.k_norm(k)
        key = self.indexer_op.apply_rope_and_rotate_k(k, flashmla_params.positions_d)
        return key

    # ---- CP helpers ----

    def _setup_cp_params(self, cp_params: Any) -> None:
        """Copy CP indices from cp_params into indexer_op state."""
        op = self.indexer_op
        op.q0_idx = cp_params.q0_idx
        op.q1_idx = cp_params.q1_idx
        op.q0_idx_global = cp_params.q0_idx_global
        op.q1_idx_global = cp_params.q1_idx_global
        op.kv_restore_unpad_indices = cp_params.kv_restore_unpad_indices
        op.total_global_ids = cp_params.total_global_ids
        op.total_local_ids = cp_params.total_local_ids
        op.cu_kv_seqlens_global = cp_params.cu_kv_seqlens_global
        op.total_kv_len = cp_params.total_kv_len

        kv_cache_sharded = self._is_roundrobin and getattr(
            cp_params, "kv_cache_sharded", False
        )
        if kv_cache_sharded:
            op._cu_local_kv_seqlens = cp_params.cu_local_kv_seqlens
            op._total_local_kv = cp_params.total_local_kv
            op._kv_allgather_restore_indices = cp_params.kv_allgather_restore_indices
            op._local_indexer_slot_mapping = cp_params.local_indexer_slot_mapping
        else:
            op._cu_local_kv_seqlens = None
            op._total_local_kv = None
            op._kv_allgather_restore_indices = None
            op._local_indexer_slot_mapping = None

    def _quant_q_k_cp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: KVCache,
        slot_mapping: torch.Tensor,
        attention_inputs: Any,
        cp_params: Any,
    ):
        """CP quant: zigzag uses full cache, round-robin writes local K directly."""
        kv_cache_sharded = self._is_roundrobin and getattr(
            cp_params, "kv_cache_sharded", False
        )
        local_indexer_slot_mapping = (
            getattr(cp_params, "local_indexer_slot_mapping", None)
            if kv_cache_sharded
            else None
        )
        return self.indexer_op.quant_q_k_cp(
            query,
            key,
            kv_cache,
            slot_mapping,
            attention_inputs,
            kv_cache_sharded=kv_cache_sharded,
            local_indexer_slot_mapping=local_indexer_slot_mapping,
        )

    def _get_topk_cp(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
    ):
        """CP topk: zigzag reads from full cache, round-robin reads from workspace or gathers from sharded cache."""
        if self._is_roundrobin:
            return self.indexer_op._get_topk_ragged_cp_roundrobin(
                q_fp8,
                weights,
                fmha_params,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
            )
        return self.indexer_op._get_topk_ragged_cp_zigzag(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attention_inputs,
        )

    # ---- forward ----

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        use_fast_path: bool,
        cp_params: Any = None,
    ) -> torch.Tensor:
        # fast path: only compute and store k cache, skip all q and weights ops
        if use_fast_path:
            key = self._get_k_bf16(hidden_states, fmha_params)
            self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)
            return None

        is_cp_prefill = attention_inputs.is_prefill and self._is_cp
        if is_cp_prefill:
            assert cp_params is not None, "cp_params is required for prefill CP"
            self._setup_cp_params(cp_params)

        # Compute query and key with RoPE and rotation
        query, key = self._get_q_k_bf16(
            q_lora, hidden_states, fmha_params, attention_inputs
        )

        # Quantize query and key
        if is_cp_prefill:
            q_fp8, q_scale = self._quant_q_k_cp(
                query,
                key,
                kv_cache,
                fmha_params.slot_mapping,
                attention_inputs,
                cp_params,
            )
        else:
            q_fp8, q_scale = self.indexer_op.quant_q_k(
                query, key, kv_cache, fmha_params.slot_mapping
            )

        # Compute weights for attention
        weights = self._get_logits_head_gate(hidden_states, q_scale)

        # Compute TopK
        if not attention_inputs.is_prefill:
            return self.indexer_op._get_topk_paged(
                q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
        if is_cp_prefill:
            return self._get_topk_cp(
                q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
        return self.indexer_op._get_topk_ragged(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )

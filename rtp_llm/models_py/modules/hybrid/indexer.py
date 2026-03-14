from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from rtp_llm.models_py.modules import IndexerOp, LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache
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
        self.blocksize = attn_config.kernel_tokens_per_block  # page size, typically 64
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2
        self.is_neox_style = attn_config.rope_config.indexer_is_neox_style
        self.parallelism_config = parallelism_config

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

    def _prefill_cp_enabled(self) -> bool:
        if self.parallelism_config is None:
            return False
        return self.parallelism_config.prefill_cp_config.is_enabled()

    def _is_sparse_prefill_cp(self, attention_inputs: Any) -> bool:
        return bool(attention_inputs.is_prefill) and self._prefill_cp_enabled()

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
        cp_params: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        k = self.wk(x)
        k = self.k_norm(k)

        positions = flashmla_params.positions_d
        if self._prefill_cp_enabled():
            assert cp_params is not None
            query, key = self.indexer_op.apply_rope_and_rotate_q_k_cp(
                q,
                k,
                positions,
                cp_params.total_local_ids,
                cp_params.total_global_ids,
            )
        else:
            query, key = self.indexer_op.apply_rope_and_rotate_q_k(q, k, positions)

        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        flashmla_params: Any,
    ) -> torch.Tensor:
        k = self.wk(x)
        k = self.k_norm(k)
        return self.indexer_op.apply_rope_and_rotate_k(k, flashmla_params.positions_d)

    def _quantize_q_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None
            return self.indexer_op.quant_q_k_cp(
                query,
                key,
                kv_cache,
                fmha_params.slot_mapping,
                cp_params.kv_restore_unpad_indices,
            )
        return self.indexer_op.quant_q_k(query, key, kv_cache, fmha_params.slot_mapping)

    def _compute_topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ) -> torch.Tensor:
        if not attention_inputs.is_prefill:
            return self.indexer_op._get_topk_paged(
                q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
        if self._prefill_cp_enabled():
            assert cp_params is not None
            kv_n = cp_params.kv_restore_unpad_indices.shape[0]
            return self.indexer_op._get_topk_ragged_cp(
                q_fp8,
                weights,
                kv_cache,
                fmha_params,
                attention_inputs,
                cp_params.total_local_ids,
                cp_params.total_global_ids,
                cp_params.cu_kv_seqlens_global,
                kv_n,
            )
        return self.indexer_op._get_topk_ragged(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )

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
        if use_fast_path:
            key = self._get_k_bf16(hidden_states, fmha_params)
            self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)
            return None

        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None, "cp_params is required for sparse prefill CP"

        query, key = self._get_q_k_bf16(q_lora, hidden_states, fmha_params, cp_params)
        q_fp8, q_scale = self._quantize_q_k(
            query, key, kv_cache, fmha_params, attention_inputs, cp_params
        )
        weights = self._get_logits_head_gate(hidden_states, q_scale)
        return self._compute_topk(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs, cp_params
        )

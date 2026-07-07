"""
DecoderLayer for DeepSeek V3.2, new-loader style.

Orchestrates:
  - Pre-norm → MLA attention (with optional sparse Indexer) → post-norm → FFN
  - FFN is either a dense MLP (first K layers) or a MoE block (remaining layers).
  - Indexer is a stateful submodule with HF ckpt keys:
      model.layers.{i}.self_attn.indexer.wq_b.weight
      model.layers.{i}.self_attn.indexer.wk.weight
      model.layers.{i}.self_attn.indexer.k_norm.weight
      model.layers.{i}.self_attn.indexer.k_norm.bias
      model.layers.{i}.self_attn.indexer.weights_proj.weight
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import LayerNorm, RMSNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import IndexerOp
from rtp_llm.models_py.modules.factory.attention.attn_factory import MlaImplBase
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W

from .attention import DeepSeekV32MlaAttention
from .moe import DeepSeekV32MoEBlock


class DeepSeekV32Indexer(RtpModule):
    """Sparse Indexer for DeepSeek V3.2 DSA, new-loader style.

    HF ckpt keys (per layer):
      model.layers.{i}.self_attn.indexer.wq_b.weight
      model.layers.{i}.self_attn.indexer.wk.weight
      model.layers.{i}.self_attn.indexer.k_norm.weight
      model.layers.{i}.self_attn.indexer.k_norm.bias
      model.layers.{i}.self_attn.indexer.weights_proj.weight
    """

    def __init__(
        self,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rope_head_dim: int,
        hidden_size: int,
        q_lora_rank: int,
        layer_idx: int,
        layernorm_eps: float,
        blocksize: int,
        is_neox_style: bool,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.bfloat16,
        cos_sin_cache: Optional[torch.Tensor] = None,
        parallelism_config: Any = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.rope_head_dim = rope_head_dim
        self.block_size = 128
        self.head_kv = 1
        self.scale_fmt = "none"
        self.softmax_scale = index_head_dim**-0.5
        self.weights_scale = index_n_heads**-0.5
        self.blocksize = blocksize
        self.indexer_size = index_head_dim / 2 + index_head_dim / 128 * 2
        self.is_neox_style = is_neox_style
        self.parallelism_config = parallelism_config

        # wq_b: q_lora_rank → index_n_heads * index_head_dim
        self.wq_b = ColumnParallelLinear(
            input_size=q_lora_rank,
            output_size=index_n_heads * index_head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix="wq_b",
            bias=False,
            params_dtype=params_dtype,
        )
        # wk: hidden_size → index_head_dim. The HF Indexer is MQA-style for
        # the key path: a single shared k vector of size `index_head_dim`,
        # not n_heads independent keys. ckpt shape is [head_dim, hidden].
        self.wk = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=index_head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix="wk",
            bias=False,
            params_dtype=params_dtype,
        )
        # HF Indexer applies k_norm per-head over `index_head_dim`, not over
        # the full `index_n_heads * index_head_dim` flat feature. The ckpt
        # bias/weight tensors are sized `index_head_dim`.
        self.k_norm = LayerNorm(
            index_head_dim, eps=layernorm_eps, params_dtype=params_dtype
        )
        # weights_proj: hidden_size → index_n_heads
        self.weights_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=index_n_heads,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix="weights_proj",
            bias=False,
            params_dtype=torch.float32,
        )

        self.cos_sin_cache = cos_sin_cache

        self.indexer_op = IndexerOp(
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope_head_dim=rope_head_dim,
            cos_sin_cache=cos_sin_cache,
            blocksize=blocksize,
            block_size=self.block_size,
            scale_fmt=self.scale_fmt,
            is_neox_style=is_neox_style,
        )

    def _prefill_cp_enabled(self) -> bool:
        if self.parallelism_config is None:
            return False
        return self.parallelism_config.prefill_cp_config.is_enabled()

    def _is_sparse_prefill_cp(self, attention_inputs: Any) -> bool:
        return bool(attention_inputs.is_prefill) and self._prefill_cp_enabled()

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
    ):
        q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        k = self.wk(x)
        k = self.k_norm(k)

        if self._prefill_cp_enabled():
            assert cp_params is not None
            query, key = self.indexer_op.apply_rope_and_rotate_q_k_cp(
                q,
                k,
                cp_params.full_rope_pos_ids,
            )
        else:
            positions = flashmla_params.positions_d
            query, key = self.indexer_op.apply_rope_and_rotate_q_k(q, k, positions)

        return query, key

    def _get_k_bf16(self, x: torch.Tensor, flashmla_params: Any) -> torch.Tensor:
        k = self.wk(x)
        k = self.k_norm(k)
        return self.indexer_op.apply_rope_and_rotate_k(k, flashmla_params.positions_d)

    def _quantize_q_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ):
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
        kv_cache,
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
            return self.indexer_op._get_topk_ragged_cp(
                q_fp8,
                weights,
                kv_cache,
                fmha_params,
                attention_inputs,
                cp_params.total_local_ids,
                cp_params.cu_kv_seqlens_global,
                cp_params.total_kv_len,
                cp_params.precomputed_ks,
                cp_params.precomputed_ke,
                cp_params.precomputed_lengths,
                cp_params.precomputed_topk_off,
            )
        return self.indexer_op._get_topk_ragged(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        kv_cache,
        fmha_params: Any,
        attention_inputs: Any,
        use_fast_path: bool,
        cp_params: Any = None,
    ) -> Optional[torch.Tensor]:
        if use_fast_path:
            key = self._get_k_bf16(hidden_states, fmha_params)
            self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)
            return None

        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None, "cp_params required for sparse prefill CP"

        query, key = self._get_q_k_bf16(q_lora, hidden_states, fmha_params, cp_params)
        q_fp8, q_scale = self._quantize_q_k(
            query, key, kv_cache, fmha_params, attention_inputs, cp_params
        )
        weights = self._get_logits_head_gate(hidden_states, q_scale)
        return self._compute_topk(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs, cp_params
        )


class DeepSeekV32DenseMLP(RtpModule):
    """Dense MLP for the first K layers (before MoE layers), new-loader style.

    HF ckpt keys:
      model.layers.{i}.mlp.gate_proj.weight
      model.layers.{i}.mlp.up_proj.weight
      model.layers.{i}.mlp.down_proj.weight
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="gate_up_proj",
            bias=False,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=params_dtype,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="down_proj",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = torch.nn.functional.silu(gate) * up
        x = self.down_proj(x)
        if self.tp_size > 1:
            x = all_reduce(x, group=Group.TP)
        return x


class DeepSeekV32DecoderLayer(RtpModule):
    """Single decoder layer: MLA attention + FFN (dense or MoE)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        nope_head_dim: int,
        rope_head_dim: int,
        v_head_dim: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        params_dtype: torch.dtype,
        layernorm_eps: float,
        quant_config: Optional[QuantizationConfig],
        model_config: Any,
        parallelism_config: Any,
        moe_config: Any,
        # --- FFN config ---
        is_moe_layer: bool = False,
        dense_intermediate_size: int = 0,
        moe_intermediate_size: int = 0,
        num_experts: int = 0,
        top_k: int = 0,
        shared_expert_intermediate_size: int = 0,
        has_shared_expert: bool = True,
        scoring_func: int = 1,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        has_moe_norm: bool = False,
        correction_bias: bool = False,
        # --- Indexer config ---
        is_sparse: bool = False,
        index_n_heads: int = 0,
        index_head_dim: int = 0,
        index_topk: int = 0,
        indexer_is_neox_style: bool = False,
        cos_sin_cache: Optional[torch.Tensor] = None,
        blocksize: int = 64,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sparse = is_sparse
        self.q_lora_rank = q_lora_rank

        self.input_layernorm = RMSNorm(
            hidden_size, eps=layernorm_eps, params_dtype=params_dtype
        )
        self.self_attn = DeepSeekV32MlaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            v_head_dim=v_head_dim,
            layer_idx=layer_idx,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            layernorm_eps=layernorm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=layernorm_eps, params_dtype=params_dtype
        )

        if is_moe_layer:
            self.mlp = DeepSeekV32MoEBlock(
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                layer_idx=layer_idx,
                tp_size=tp_size,
                tp_rank=tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                model_config=model_config,
                parallelism_config=parallelism_config,
                moe_config=moe_config,
                quant_config=quant_config,
                params_dtype=params_dtype,
                has_shared_expert=has_shared_expert,
                shared_expert_intermediate_size=shared_expert_intermediate_size,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                n_group=n_group,
                topk_group=topk_group,
                has_moe_norm=has_moe_norm,
                correction_bias=correction_bias,
            )
        else:
            self.mlp = DeepSeekV32DenseMLP(
                hidden_size=hidden_size,
                intermediate_size=dense_intermediate_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
                quant_config=quant_config,
                params_dtype=params_dtype,
            )

        # Register the indexer as a *child of self_attn*, not of this layer,
        # so HF ckpt keys `model.layers.{i}.self_attn.indexer.{wq_b,wk,...}`
        # route correctly via the streaming dispatch.  self_attn.forward
        # invokes self.indexer internally (see attention.py
        # DeepSeekV32MlaAttention._run_sparse_indexer); the `self.indexer`
        # property here is just a convenience accessor for callers that
        # want to reach the indexer from the layer level.
        if is_sparse:
            self.self_attn.indexer = DeepSeekV32Indexer(
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                rope_head_dim=rope_head_dim,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                layer_idx=layer_idx,
                layernorm_eps=layernorm_eps,
                blocksize=blocksize,
                is_neox_style=indexer_is_neox_style,
                tp_size=tp_size,
                tp_rank=tp_rank,
                quant_config=quant_config,
                params_dtype=params_dtype,
                cos_sin_cache=cos_sin_cache,
                parallelism_config=parallelism_config,
            )
        else:
            self.self_attn.indexer = None

    @property
    def indexer(self):
        return getattr(self.self_attn, "indexer", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # self_attn.forward owns the sparse Indexer call internally,
        # mirroring legacy MlaAttention (modules/hybrid/mla_attention.py).
        attn_output = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(self.mlp, DeepSeekV32MoEBlock):
            hidden_states = self.mlp(hidden_states)
            if self.mlp.ep_size <= 1 and self.mlp.tp_size > 1:
                hidden_states = all_reduce(hidden_states, group=Group.TP)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states

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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import LayerKVCache
    from rtp_llm.models_py.modules.factory.attention.attn_factory import MlaImplBase

import torch

from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import IndexerOp
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.models_py.quant_methods.base import QuantizationConfig

from .attention import DeepSeekV32MlaAttention
from .indexer_parameters import SparseIndexerParameters
from .mlp import DeepSeekV32MLP
from .moe import DeepSeekV32MoEBlock


class DeepSeekV32Indexer(Indexer, SparseIndexerParameters):
    """Sparse Indexer for DeepSeek V3.2 DSA, new-loader style.

    Inherit the production Indexer forward and own the projections as
    independently loadable RtpModule children.

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
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.bfloat16,
        cos_sin_cache: Optional[torch.Tensor] = None,
        parallelism_config: Any = None,
        prefix: str = "self_attn.indexer",
        hw_kernel_config=None,
        owner_tokens_per_block: Optional[int] = None,
        stable_topk: bool = False,
    ):
        # The main Indexer forward uses these direct projection parameters;
        # only its old checkpoint-driven constructor is replaced.
        SparseIndexerParameters.__init__(
            self,
            index_n_heads,
            index_head_dim,
            hidden_size,
            q_lora_rank,
            layernorm_eps,
            quant_config,
            params_dtype,
            prefix,
        )
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
        self.is_neox_style = is_neox_style
        self.parallelism_config = parallelism_config

        self._configure_indexer_runtime(
            hw_kernel_config, owner_tokens_per_block or blocksize
        )
        self._fuse_kernels_enabled = self._can_fuse_rope(cos_sin_cache)

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
            stable_topk=stable_topk,
            enable_fuse_kernels=self._fuse_kernels_enabled,
        )

    def bind_rope_cache(self, cos_sin_cache: torch.Tensor) -> None:
        self.indexer_op.cos_sin_cache = cos_sin_cache
        self._fuse_kernels_enabled = self._can_fuse_rope(cos_sin_cache)
        self.indexer_op.enable_fuse_kernels = self._fuse_kernels_enabled


class DeepSeekV32DecoderLayer(RtpModule):
    """Single decoder layer: MLA attention + FFN (dense or MoE)."""

    moe_cls = DeepSeekV32MoEBlock

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
        attn_tp_size: int,
        attn_tp_rank: int,
        ffn_tp_size: int,
        ffn_tp_rank: int,
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
        topk_method: str = "greedy",
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
        prefix: str = "layer",
        hw_kernel_config=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sparse = is_sparse
        self.q_lora_rank = q_lora_rank

        self.input_layernorm = RMSResNorm(
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
            tp_size=attn_tp_size,
            tp_rank=attn_tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            layernorm_eps=layernorm_eps,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSResNorm(
            hidden_size, eps=layernorm_eps, params_dtype=params_dtype
        )

        if is_moe_layer:
            self.mlp = self.moe_cls(
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                layer_idx=layer_idx,
                tp_size=ffn_tp_size,
                tp_rank=ffn_tp_rank,
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
                topk_method=topk_method,
                has_moe_norm=has_moe_norm,
                correction_bias=correction_bias,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepSeekV32MLP(
                hidden_size=hidden_size,
                intermediate_size=dense_intermediate_size,
                tp_size=ffn_tp_size,
                tp_rank=ffn_tp_rank,
                quant_config=quant_config,
                params_dtype=params_dtype,
                prefix=f"{prefix}.mlp",
            )

        # Register the indexer as a *child of self_attn*, not of this layer,
        # so HF ckpt keys `model.layers.{i}.self_attn.indexer.{wq_b,wk,...}`
        # route correctly via the streaming dispatch.  self_attn.forward
        # invokes self.indexer internally (see attention.py
        # DeepSeekV32MlaAttention._run_sparse_indexer); the `self.indexer`
        # property here is just a convenience accessor for callers that
        # want to reach the indexer from the layer level.
        self.carries_dsa_topk = (
            is_sparse and getattr(model_config, "model_type", None) == "glm_5"
        )
        if self.carries_dsa_topk:
            from rtp_llm.utils.dsa_indexing import dsa_layer_skips_topk

            self.self_attn.reuse_topk_indices = dsa_layer_skips_topk(
                model_config, layer_idx
            )
        if is_sparse and not self.self_attn.reuse_topk_indices:
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
                quant_config=quant_config,
                params_dtype=params_dtype,
                cos_sin_cache=cos_sin_cache,
                parallelism_config=parallelism_config,
                prefix=f"{prefix}.self_attn.indexer",
                hw_kernel_config=hw_kernel_config,
                owner_tokens_per_block=getattr(
                    getattr(model_config, "attn_config", None),
                    "tokens_per_block",
                    blocksize,
                ),
                stable_topk=getattr(model_config, "model_type", "") in ("glm_5", "glm_5_mtp"),
            )
        else:
            self.self_attn.indexer = None

    @property
    def indexer(self):
        return self.self_attn.indexer

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: MlaImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        previous_topk: Optional[torch.Tensor] = None,
    ) -> DecodeLayerOutput:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # self_attn.forward owns the sparse Indexer call internally,
        # mirroring legacy MlaAttention (modules/hybrid/mla_attention.py).
        if self.carries_dsa_topk:
            attn_output, topk = self.self_attn(
                hidden_states,
                fmha_impl,
                kv_cache,
                previous_topk=previous_topk,
                return_topk=True,
            )
        else:
            attn_output = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = attn_output

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        output = DecodeLayerOutput(hidden_states, residual)
        if self.carries_dsa_topk:
            output.topk_indices = topk
        return output

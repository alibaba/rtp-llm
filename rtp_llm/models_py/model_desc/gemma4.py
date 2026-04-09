import logging
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    RMSNorm,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import get_fmha_impl
from rtp_llm.ops import (
    AttentionConfigs,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Determine layer type
        layer_types = getattr(model_config, "gemma4_layer_types", [])
        self.is_sliding = True
        if layer_types and layer_idx < len(layer_types):
            self.is_sliding = layer_types[layer_idx] == "sliding_attention"

        # Get per-type attention config
        if self.is_sliding:
            attn_info = getattr(model_config, "gemma4_sliding_attn_config", {})
        else:
            attn_info = getattr(model_config, "gemma4_global_attn_config", {})

        kv_head_num = attn_info.get("kv_head_num", model_config.attn_config.kv_head_num)
        head_dim = attn_info.get("head_dim", model_config.attn_config.size_per_head)

        # Build per-layer AttentionConfigs
        attn_config = AttentionConfigs()
        attn_config.head_num = model_config.attn_config.head_num
        attn_config.kv_head_num = kv_head_num
        attn_config.size_per_head = head_dim
        attn_config.tokens_per_block = model_config.attn_config.tokens_per_block

        # RoPE config per layer type
        rope_theta = attn_info.get("rope_theta", 10000.0)
        partial_rotary_factor = attn_info.get("partial_rotary_factor", 1.0)
        attn_config.rope_config.style = 1  # RopeStyle::Base
        attn_config.rope_config.base = rope_theta
        attn_config.rope_config.dim = int(head_dim * partial_rotary_factor)

        self.attn_config = attn_config

        # Attention module
        self.self_attn = CausalAttention(
            attn_config,
            parallelism_config,
            weights,
            model_config.layernorm_eps,
            layer_idx=layer_idx,
        )

        # FFN (dense MLP with gated GELU activation)
        self.mlp = DenseMLP(
            model_config.activation_type,
            parallelism_config,
            weights,
            model_config.quant_config,
        )

        # Layer norms (pre-norm architecture)
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=model_config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=model_config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma4Model(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int = 1,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )

        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                )
                for idx in range(self.layer_num)
            ]
        )

        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

        # Final logit softcapping
        self.final_logit_softcapping = getattr(
            model_config, "final_logit_softcapping", 0.0
        )

        # Build per-type attention configs for FMHA dispatch
        self._sliding_attn_config = self._build_attn_config(
            model_config, "gemma4_sliding_attn_config"
        )
        self._global_attn_config = self._build_attn_config(
            model_config, "gemma4_global_attn_config"
        )

        # Layer type lookup
        layer_types = getattr(model_config, "gemma4_layer_types", [])
        self._layer_is_sliding = []
        for i in range(self.layer_num):
            if layer_types and i < len(layer_types):
                self._layer_is_sliding.append(
                    layer_types[i] == "sliding_attention"
                )
            else:
                self._layer_is_sliding.append(True)

    def _build_attn_config(self, model_config: ModelConfig, attr_name: str) -> AttentionConfigs:
        info = getattr(model_config, attr_name, {})
        config = AttentionConfigs()
        config.head_num = model_config.attn_config.head_num
        config.kv_head_num = info.get("kv_head_num", model_config.attn_config.kv_head_num)
        config.size_per_head = info.get("head_dim", model_config.attn_config.size_per_head)
        config.tokens_per_block = model_config.attn_config.tokens_per_block

        rope_theta = info.get("rope_theta", 10000.0)
        partial_rotary_factor = info.get("partial_rotary_factor", 1.0)
        config.rope_config.style = 1
        config.rope_config.base = rope_theta
        config.rope_config.dim = int(config.size_per_head * partial_rotary_factor)

        return config

    def _prepare_fmha_for_config(
        self,
        attn_config: AttentionConfigs,
        inputs: PyModelInputs,
    ) -> FMHAImplBase:
        """Create FMHA impl with a specific attention config."""
        # tokens_per_block is set by C++ engine AFTER Python model construction,
        # so re-read it from the live config at forward time
        live_tpb = self.config.attn_config.tokens_per_block
        if live_tpb > 0 and attn_config.tokens_per_block != live_tpb:
            attn_config.tokens_per_block = live_tpb
        return get_fmha_impl(
            attn_config,
            self.weight,
            inputs.attention_inputs,
            self.fmha_config,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)

        attention_inputs = inputs.attention_inputs

        # Prepare two FMHA implementations (one per attention type)
        fmha_impl_sliding = self._prepare_fmha_for_config(
            self._sliding_attn_config, inputs
        )
        fmha_impl_global = self._prepare_fmha_for_config(
            self._global_attn_config, inputs
        )

        for i, decoder_layer in enumerate(self.layers):
            select_block_map_for_layer(attention_inputs, i)

            if self._layer_is_sliding[i]:
                layer_fmha = fmha_impl_sliding
            else:
                layer_fmha = fmha_impl_global

            hidden_states = decoder_layer(
                hidden_states,
                layer_fmha,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )

        hidden_states = self.norm(hidden_states)

        # NOTE: final_logit_softcapping should be applied to logits (after lm_head),
        # but lm_head lives in C++ PyWrappedModel. Skipped here; handled separately.

        return PyModelOutputs(hidden_states, fmha_impl_sliding.fmha_params)

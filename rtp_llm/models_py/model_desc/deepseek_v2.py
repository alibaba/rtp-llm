from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.ep.layers import FusedMoE
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.models_py.modules.mla import DeepSeekV2Attention
from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
from rtp_llm.models_py.modules.moe import FusedMoe
from rtp_llm.models_py.modules.moe.fused_moe_factory import FusedMoeFactory
from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class DeepSeekV2NormalMoeLayer(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.config = config
        self.top_k = config.moe_k
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, config
        )
        self.fused_moe = FusedMoE(config, weights, layer_id=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(hidden_states)
        return self.fused_moe(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )


class DeepSeekV2MoeLayer(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.config = config
        self.top_k = config.moe_k
        # Create gate layer
        use_fp8_path = self._should_use_fp8_linear(config, weights)
        if use_fp8_path:
            gate_scale_key = "partial_moe_weights.gate.weight_only_quant_scale"
            has_gate_scale = gate_scale_key in weights
            if has_gate_scale:
                # Create FP8 gate layer
                self.gate = LinearFactory.create_linear(
                    weight=weights[W.moe_gate],
                    weight_scales=weights[gate_scale_key],
                    bias=None,
                    config=config,
                    force_fp8=True,
                )
            else:
                # Create regular gate layer
                self.gate = LinearFactory.create_linear_from_weights(
                    weights, W.moe_gate, None, None, config
                )
        else:
            # Create regular gate layer
            self.gate = LinearFactory.create_linear_from_weights(
                weights, W.moe_gate, None, None, config
            )

        # Always use FusedMoeFactory.create_fused_moe
        self.fused_moe: FusedMoe = FusedMoeFactory.create_fused_moe(config, weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for FusedMoE implementation."""
        from rtp_llm.models_py.modules.ep.topk import select_experts

        router_logits = self.gate(hidden_states)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=False,
            renormalize=True,
        )

        # Convert topk_ids to int64 for DeepEP compatibility
        topk_ids = topk_ids.long()

        return self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="silu",
        )

    def _should_use_fp8_linear(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> bool:
        """Check if FP8 linear layers should be used."""
        if not hasattr(config, "quant_config"):
            return False

        # Check if any MoE weights are FP8
        gate_weight = weights.get(W.moe_gate)
        moe_w1 = weights.get(W.moe_w1)
        moe_w2 = weights.get(W.moe_w2)

        # Use EPMoE for FP8 support
        for weight in [gate_weight, moe_w1, moe_w2]:
            if weight is not None and weight.dtype == torch.float8_e4m3fn:
                return True

        return False


class DeepSeekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.self_attn = DeepSeekV2Attention(config, weights, layer_idx)

        if len(config.moe_layer_index) > 0 and layer_idx < config.moe_layer_index[0]:
            self.is_dense_layer = True
        else:
            self.is_dense_layer = False
            self.moe_mlp = DeepSeekV2NormalMoeLayer(config, weights)
        self.add_shared_expert = config.moe_style == 2

        if self.add_shared_expert:
            self.shared_mlp = FusedSiluActDenseMLP(config, weights)

        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_dense_layer:
            hidden_states = self.shared_mlp(hidden_states)
        else:
            experts_output = self.moe_mlp(hidden_states)
            if self.add_shared_expert:
                shared_mlp_output = self.shared_mlp(hidden_states)
                hidden_states = experts_output + shared_mlp_output
            else:
                hidden_states = experts_output
        hidden_states = residual + hidden_states

        return hidden_states


class DeepSeekV2Model(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        config.head_num = config.head_num // config.tp_size
        super().__init__(config, weights)
        self.layer_num = config.layer_num
        self.vocab_size = config.vocab_size
        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))

        self.layers = nn.ModuleList(
            [
                DeepSeekV2DecoderLayer(
                    config,
                    weights.weights[idx],
                    idx,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        attention_inputs: PyAttentionInputs = inputs.attention_inputs

        fmha_impl = self.get_mla_impl(attention_inputs)

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)

        return PyModelOutputs(hidden_states)


__all__ = [
    "DeepSeekV2Model",
]

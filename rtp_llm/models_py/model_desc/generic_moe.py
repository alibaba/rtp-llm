import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    AttnImplFactory,
    CausalAttention,
    Embedding,
    FMHAImplBase,
    FusedMoeFactory,
    FusedSiluActDenseMLP,
    GroupTopK,
    LinearFactory,
    MlaAttention,
    RMSNorm,
    SelectTopk,
)
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class GenericMoeLayer(nn.Module):
    """Generic MoE layer supporting both Qwen3 and internal model."""

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.config = config

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_inter_padding_size
        self.num_experts = config.phy_exp_num
        self.top_k = config.moe_k

        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, config
        )
        self.select_topk = SelectTopk(config)
        self.fused_moe = FusedMoeFactory().create_fused_moe(config, weights)
        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]
        self.expert_map = self.build_expert_map()
        self.add_shared_expert = config.moe_style == 2
        if self.add_shared_expert:
            self.shared_expert = FusedSiluActDenseMLP(config, weights)
        else:
            self.shared_expert = None
        if weights.get(W.shared_expert_gate, None) is not None:
            self.shared_expert_gate = LinearFactory.create_linear_from_weights(
                weights, W.shared_expert_gate, None, None, config
            )
        else:
            self.shared_expert_gate = None

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)

    def build_expert_map(self):
        """Build expert mapping for EP (Expert Parallelism)."""
        num_local_experts = self.num_local_experts
        global_num_experts = self.num_experts
        expert_map = torch.full((global_num_experts,), fill_value=-1, dtype=torch.int32)
        start_id = self.config.ep_rank * num_local_experts
        end_id = start_id + num_local_experts
        expert_map[start_id:end_id] = torch.tensor(list(range(num_local_experts)))
        return expert_map.to(device=torch.cuda.current_device(), dtype=torch.int32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        # different executor may need different topk_ids dtype
        topk_ids_dtype = self.fused_moe.topk_ids_dtype
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=topk_ids_dtype,
            device=hidden_states.device,
        )

        if self.correction_bias is not None:
            self.group_topk = GroupTopK()
            self.renormalize = self.config.has_moe_norm
            self.num_expert_group = self.config.moe_n_group

            self.topk_group = self.config.moe_topk_group
            self.n_routed_experts = self.config.expert_num  # config.n_routed_experts
            self.routed_scaling_factor = self.config.routed_scaling_factor
            self.group_topk(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                scores=router_logits_fp32,
                correction_bias=self.correction_bias,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            # Top-K selection using C++ SelectTopkOp
            self.select_topk(router_logits_fp32, topk_ids, topk_weights)

        self.select_topk(router_logits_fp32, topk_ids, topk_weights)
        experts_output = self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
            expert_map=self.expert_map,
        )
        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_expert_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states))
                    * shared_expert_output
                )
            experts_output = experts_output + shared_expert_output
        return experts_output


class GenericMoeDecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        if config.use_mla:
            self.self_attn = MlaAttention(config, weights, layer_idx)
        else:
            self.self_attn = CausalAttention(config, weights)

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.mlp = FusedSiluActDenseMLP(config, weights)
        else:
            self.mlp = GenericMoeLayer(config, weights)
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
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        # MLP (Dense or MoE with optional shared experts)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GenericMoeModel(GptModelBase):
    """Generic MoE model supporting Qwen3-MoE, internal model, and other MoE architectures."""

    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(config, weights.weights[idx], idx)
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
        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config, self.weight, attention_inputs
        )

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )

        hidden_states = self.norm(hidden_states)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "GenericMoeLayer",
    "GenericMoeDecoderLayer",
    "GenericMoeModel",
]

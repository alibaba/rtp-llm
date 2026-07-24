"""Five-layer-friendly end-to-end Kimi K3 Torch correctness model.

This model is the executable modeling oracle used before RTP weight/cache ABI
integration.  It contains every K3-specific architectural component: hybrid
KDA/MLA attention, no-RoPE MLA suffix, SiTU dense and latent MoE blocks, and
attention residual mixing.  It is not the final serving class because its
attention/cache implementations favor clarity over production memory use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.kda_state import (
    KDAExecutionMode,
    KimiKDAState,
)
from rtp_llm.models_py.modules.kimi_k3.reference.attn_residual import (
    KimiAttentionResidualMixer,
)
from rtp_llm.models_py.modules.kimi_k3.reference.common import KimiRMSNorm
from rtp_llm.models_py.modules.kimi_k3.reference.kda_attention import (
    KimiDeltaAttentionReference,
)
from rtp_llm.models_py.modules.kimi_k3.reference.mla_attention import (
    KimiMLAAttentionReference,
    KimiMLAState,
)
from rtp_llm.models_py.modules.kimi_k3.reference.mlp import (
    KimiMLPReference,
    KimiSparseMoeBlockReference,
)


KimiLayerState = Union[KimiKDAState, KimiMLAState]


@dataclass(frozen=True)
class KimiK3ReferenceConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    kda_layer_indices: Tuple[int, ...]
    num_attention_heads: int
    kda_head_dim: int
    conv_kernel_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_suffix_head_dim: int
    v_head_dim: int
    dense_intermediate_size: int
    first_k_dense_replace: int
    num_experts: int
    num_experts_per_token: int
    num_shared_experts: int
    moe_intermediate_size: int
    routed_expert_hidden_size: int
    attn_res_block_size: int
    rms_norm_eps: float = 1e-5
    gate_lower_bound: float = -5.0
    kda_chunk_size: int = 64
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    latent_moe_use_norm: bool = True
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    num_expert_group: int = 1
    topk_group: int = 1
    mla_use_output_gate: bool = True
    use_full_rank_gate: bool = True

    def __post_init__(self) -> None:
        all_layers = set(range(self.num_hidden_layers))
        kda_layers = set(self.kda_layer_indices)
        if not kda_layers.issubset(all_layers):
            raise ValueError("kda_layer_indices contains an out-of-range layer")
        if self.num_hidden_layers <= 0 or self.attn_res_block_size <= 0:
            raise ValueError("layer count and attn_res_block_size must be positive")
        if self.hidden_size <= 0 or self.vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")


@dataclass(frozen=True)
class KimiK3ReferenceCache:
    layer_states: Tuple[Optional[KimiLayerState], ...]


class KimiK3ReferenceDecoderLayer(nn.Module):
    def __init__(self, config: KimiK3ReferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = config.hidden_size
        self.attn_res_block_size = config.attn_res_block_size
        self.is_kda = layer_idx in config.kda_layer_indices
        if self.is_kda:
            self.self_attn = KimiDeltaAttentionReference(
                config.hidden_size,
                config.num_attention_heads,
                config.kda_head_dim,
                conv_kernel_size=config.conv_kernel_size,
                rms_norm_eps=config.rms_norm_eps,
                gate_lower_bound=config.gate_lower_bound,
                use_full_rank_gate=config.use_full_rank_gate,
                chunk_size=config.kda_chunk_size,
            )
        else:
            self.self_attn = KimiMLAAttentionReference(
                config.hidden_size,
                config.num_attention_heads,
                config.q_lora_rank,
                config.kv_lora_rank,
                config.qk_nope_head_dim,
                config.qk_suffix_head_dim,
                config.v_head_dim,
                rms_norm_eps=config.rms_norm_eps,
                use_output_gate=config.mla_use_output_gate,
            )

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = KimiSparseMoeBlockReference(
                config.hidden_size,
                config.routed_expert_hidden_size,
                config.moe_intermediate_size,
                config.num_experts,
                config.num_experts_per_token,
                config.num_shared_experts,
                rms_norm_eps=config.rms_norm_eps,
                latent_moe_use_norm=config.latent_moe_use_norm,
                routed_scaling_factor=config.routed_scaling_factor,
                renormalize=config.moe_renormalize,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                situ_beta=config.activation_situ_beta,
                situ_linear_beta=config.activation_situ_linear_beta,
            )
            self.is_moe = True
        else:
            self.mlp = KimiMLPReference(
                config.hidden_size,
                config.dense_intermediate_size,
                situ_beta=config.activation_situ_beta,
                situ_linear_beta=config.activation_situ_linear_beta,
            )
            self.is_moe = False

        self.input_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attention_res = KimiAttentionResidualMixer(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp_res = KimiAttentionResidualMixer(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        layer_state: Optional[KimiLayerState],
        *,
        mode: KDAExecutionMode,
    ) -> tuple[torch.Tensor, torch.Tensor, KimiLayerState]:
        batch, sequence_length, hidden_size = hidden_states.shape
        prefix_sum: Optional[torch.Tensor] = hidden_states

        if block_residual.shape[1] > 0:
            hidden_states = self.self_attention_res(
                prefix_sum.reshape(-1, hidden_size), block_residual
            ).reshape(batch, sequence_length, hidden_size)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (
                    block_residual,
                    prefix_sum.reshape(-1, hidden_size).unsqueeze(1),
                ),
                dim=1,
            )
            prefix_sum = None

        attention_input = self.input_layernorm(hidden_states)
        if self.is_kda:
            if layer_state is not None and not isinstance(layer_state, KimiKDAState):
                raise TypeError("KDA layer received an MLA cache")
            attention_output, new_state = self.self_attn(
                attention_input, layer_state, mode=mode
            )
        else:
            if layer_state is not None and not isinstance(layer_state, KimiMLAState):
                raise TypeError("MLA layer received a KDA cache")
            attention_output, new_state = self.self_attn(
                attention_input, layer_state
            )

        prefix_sum = (
            attention_output
            if prefix_sum is None
            else prefix_sum + attention_output
        )
        mlp_input = self.mlp_res(
            prefix_sum.reshape(-1, hidden_size), block_residual
        ).reshape(batch, sequence_length, hidden_size)
        mlp_output = self.mlp(self.post_attention_layernorm(mlp_input))
        return prefix_sum + mlp_output, block_residual, new_state


class KimiK3ReferenceModel(nn.Module):
    def __init__(self, config: KimiK3ReferenceConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                KimiK3ReferenceDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.output_attn_res = KimiAttentionResidualMixer(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[KimiK3ReferenceCache] = None,
        *,
        mode: KDAExecutionMode,
    ) -> tuple[torch.Tensor, KimiK3ReferenceCache]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T]")
        if cache is None:
            previous_states: Tuple[Optional[KimiLayerState], ...] = (
                None,
            ) * len(self.layers)
        else:
            previous_states = cache.layer_states
            if len(previous_states) != len(self.layers):
                raise ValueError("cache layer count does not match the model")

        hidden_states = self.embed_tokens(input_ids)
        block_residual = hidden_states.new_empty(
            hidden_states.shape[0] * hidden_states.shape[1],
            0,
            hidden_states.shape[2],
        )
        new_states = []
        for layer, layer_state in zip(self.layers, previous_states):
            hidden_states, block_residual, new_state = layer(
                hidden_states,
                block_residual,
                layer_state,
                mode=mode,
            )
            new_states.append(new_state)

        hidden_states = self.output_attn_res(
            hidden_states.reshape(-1, hidden_states.shape[-1]), block_residual
        ).reshape_as(hidden_states)
        return self.norm(hidden_states), KimiK3ReferenceCache(tuple(new_states))


class KimiK3ReferenceForCausalLM(nn.Module):
    def __init__(self, config: KimiK3ReferenceConfig):
        super().__init__()
        self.model = KimiK3ReferenceModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[KimiK3ReferenceCache] = None,
        *,
        mode: KDAExecutionMode,
    ) -> tuple[torch.Tensor, KimiK3ReferenceCache]:
        hidden_states, cache = self.model(input_ids, cache, mode=mode)
        return self.lm_head(hidden_states), cache

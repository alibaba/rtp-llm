import torch
from torch import nn
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import Qwen3Attention, RMSNorm, Qwen3MLP, AttentionKwargs
from typing_extensions import Unpack
from rtp_llm.utils.model_weight import W
from typing import Dict

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor], layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, weights, layer_idx)
        self.mlp = Qwen3MLP(config, weights)
        self.input_layernorm = RMSNorm(weights[W.pre_ln_gamma], eps=config.layernorm_eps)
        self.post_attention_layernorm = RMSNorm(weights[W.post_ln_gamma], eps=config.layernorm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[AttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            **kwargs
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

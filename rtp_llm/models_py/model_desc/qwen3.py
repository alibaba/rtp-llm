from typing import Optional, Tuple, List, Dict

import logging
import torch
from torch import nn

from typing_extensions import Unpack
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.attention import AttentionKwargs
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.utils.model_weight import W

from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import PyModelInputs, PyModelOutputs, PyAttentionInputs

try:
    from libth_transformer.rtp_llm_ops import FlashInferOp
except ImportError:
    logging.info("FlashInferOp not available, using fallback implementation.")

from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.models_py.modules.mlp import DenseMLP

from rtp_llm.models_py.utils.debug import set_trace_on_tty

class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor], layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.head_num
        self.head_num = config.head_num
        self.num_key_value_groups = config.head_num // config.head_num_kv
        self.q_size = config.head_num * self.head_dim
        self.kv_size = config.head_num_kv * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.qkv_proj = Linear(weights[W.attn_qkv_w], weights.get(W.attn_qkv_b, None))
        self.o_proj = Linear(weights[W.attn_o_w], weights.get(W.attn_o_b, None))
        if W.q_ln_gamma in weights:
            self.q_norm = RMSNorm(weights[W.q_ln_gamma], eps=config.layernorm_eps)  # unlike olmo, only on the head dim!
            self.k_norm = RMSNorm(weights[W.k_ln_gamma], eps=config.layernorm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = None
        self.flash_infer = FlashInferOp(config)

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_cache_base: Optional[torch.Tensor] = None,
        v_cache_base: Optional[torch.Tensor] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        # TODO: need qk norm if have

        attn_output = torch.empty_like(hidden_states)
        k_cache = k_cache_base[self.layer_idx] if k_cache_base is not None else None
        v_cache = v_cache_base[self.layer_idx] if v_cache_base is not None else None
        self.flash_infer.forward(qkv, attn_output, k_cache, v_cache, attention_inputs)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor], layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, weights, layer_idx)
        self.mlp = DenseMLP(config, weights)
        self.input_layernorm = RMSNorm(weights[W.pre_ln_gamma], eps=config.layernorm_eps)
        self.post_attention_layernorm = RMSNorm(weights[W.post_ln_gamma], eps=config.layernorm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_cache_base: Optional[torch.Tensor] = None,
        v_cache_base: Optional[torch.Tensor] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            k_cache_base=k_cache_base,
            v_cache_base=v_cache_base,
            attention_inputs=attention_inputs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Qwen3Model(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)

        self.embed_tokens = Embedding(weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, weights.weights[idx], idx) for idx in range(self.layer_num)]
        )
        self.norm = RMSNorm(weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps)
        self.lm_head = Linear(weights.get_global_weight(W.lm_head))

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids

        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs

        for decoder_layer in self.layers[: self.layer_num]:
            hidden_states = decoder_layer(
                hidden_states,
                self.k_cache_base,
                self.v_cache_base,
                attention_inputs,
            )

        return PyModelOutputs(hidden_states)

__all__ = [
    "Qwen3Model",
]

from typing import Optional, Tuple, List, Dict

import logging
import torch
from torch import nn
import torch.nn.functional as F

from typing_extensions import Unpack
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.attention import FlashInferAttention
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import PyModelInputs, PyModelOutputs, PyAttentionInputs

try:
    from libth_transformer.rtp_llm_ops import SelectTopkOp, FusedMoEOp
except ImportError:
    logging.info("SelectTopkOp/FusedMoEOp not available, using fallback implementation.")

from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.models_py.modules.mlp import DenseMLP

from rtp_llm.models_py.utils.debug import set_trace_on_tty

class TBStars2MoESparseBlock(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]):
        super().__init__()
        
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_inter_padding_size
        self.num_experts = config.expert_num
        self.top_k = config.moe_k
        
        self.gate = Linear(weights[W.moe_gate], None)
        self.up_proj = weights.get(W.moe_w1, None)
        self.down_proj = weights.get(W.moe_w2, None)
        self.select_topk_op = SelectTopkOp(config)
        self.fused_moe_op = FusedMoEOp(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        sequence_length, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states)

        router_logits_fp32 = router_logits.float()
        routing_weights = torch.zeros((sequence_length, self.top_k), dtype=torch.float32, device=hidden_states.device)
        selected_experts = torch.zeros((sequence_length, self.top_k), dtype=torch.int32, device=hidden_states.device)
        self.select_topk_op.forward(router_logits_fp32, selected_experts, routing_weights)
        
        final_hidden_states = torch.zeros((sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        self.fused_moe_op.forward(hidden_states, self.up_proj, self.down_proj, routing_weights, selected_experts, final_hidden_states)
        
        return final_hidden_states, router_logits

class TBStars2MoEDecoderLayer(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor], layer_idx: int):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.self_attn = FlashInferAttention(config, weights, layer_idx)
        
        if len(config.moe_layer_index) > 0 and layer_idx < config.moe_layer_index[0]:
            self.is_dense_layer = True
        else:
            self.is_dense_layer = False
            self.moe_mlp = TBStars2MoESparseBlock(config, weights)
        self.add_shared_expert = config.moe_style == 2
            
        if self.add_shared_expert:
            self.shared_mlp = DenseMLP(config, weights)
            
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
        
        if self.is_dense_layer:
            hidden_states = self.shared_mlp(hidden_states)
            router_logits = None
        else:
            experts_output, router_logits = self.moe_mlp(hidden_states)
            if self.add_shared_expert:
                shared_mlp_output = self.shared_mlp(hidden_states)
                hidden_states = experts_output + shared_mlp_output
            else:
                hidden_states = experts_output
        hidden_states = residual + hidden_states

        return hidden_states

class TBStars2MoEModel(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.layer_num = config.layer_num
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [TBStars2MoEDecoderLayer(config, weights.weights[idx], idx) for idx in range(self.layer_num)]
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
        hidden_states = self.norm(hidden_states)
        
        return PyModelOutputs(hidden_states)

__all__ = [
    "TBStars2Model",
]

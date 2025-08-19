import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import Unpack

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.attention import CausalAttention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
from rtp_llm.models_py.modules.ep.layers import FusedMoE, EPMoE

try:
    from libth_transformer.rtp_llm_ops import FusedMoEOp, SelectTopkOp
except ImportError:
    logging.info(
        "SelectTopkOp/FusedMoEOp not available, using fallback implementation."
    )
    FusedMoEOp = None
    SelectTopkOp = None

try:
    from rtp_llm.models_py.modules.fp8_linear import Fp8Linear
    FP8_LINEAR_AVAILABLE = True
except ImportError:
    Fp8Linear = None
    FP8_LINEAR_AVAILABLE = False

from rtp_llm.models_py.utils.debug import set_trace_on_tty

class TBStars2MoESparseBlock(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor], layer_idx: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_inter_padding_size
        self.num_experts = config.expert_num
        self.top_k = config.moe_k
        
        # Check if FP8 quantization should be used
        use_fp8_path = self._should_use_fp8_linear(config, weights)
        
        if use_fp8_path:
            # Use FP8 EP MoE path
            gate_scale_key = 'partial_moe_weights.gate.weight_only_quant_scale'
            has_gate_scale = gate_scale_key in weights
            if has_gate_scale:
                self.gate = self._create_fp8_linear(weights[W.moe_gate], weights[gate_scale_key], None, config)
            else:
                self.gate = Linear(weights[W.moe_gate], None)
            # Use EP MoE for FP8 quantized models
            self.experts = EPMoE(config, weights, layer_idx)
            self.use_fp8_path = True
        else:
            # Use traditional MoE path
            self.gate = Linear(weights[W.moe_gate], None)
            if SelectTopkOp is not None and FusedMoEOp is not None:
                # Use fused ops if available
                self.up_proj = weights.get(W.moe_w1, None)
                self.down_proj = weights.get(W.moe_w2, None)
                self.select_topk_op = SelectTopkOp(config)
                self.fused_moe_op = FusedMoEOp(config)
                self.use_fp8_path = False
            else:
                # Fallback to EP MoE
                self.experts = EPMoE(config, weights, layer_idx)
                self.use_fp8_path = True
        
    def _should_use_fp8_linear(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]) -> bool:
        """Check if FP8 linear layers should be used."""
        if not hasattr(config, 'quant_config') or config.quant_config is None:
            return False
        
        gate_weight = weights.get(W.moe_gate)
        if gate_weight is None:
            return False
        
        return gate_weight.dtype == torch.float8_e4m3fn
    
    def _create_fp8_linear(self, weight: torch.Tensor, weight_scales: torch.Tensor, 
                          bias: Optional[torch.Tensor], config: GptInitModelParameters) -> Fp8Linear:
        """Create FP8 linear layer."""
        return Fp8Linear(weight, weight_scales, bias, config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        
        if self.use_fp8_path:
            # Use EP MoE path
            final_hidden_states = self.experts(hidden_states, router_logits)
        else:
            # Use traditional fused ops path
            sequence_length, hidden_dim = hidden_states.shape
            router_logits_fp32 = router_logits.float()
            routing_weights = torch.zeros(
                (sequence_length, self.top_k),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            selected_experts = torch.zeros(
                (sequence_length, self.top_k),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            self.select_topk_op.forward(
                router_logits_fp32, selected_experts, routing_weights
            )

            final_hidden_states = torch.zeros(
                (sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self.fused_moe_op.forward(
                hidden_states,
                self.up_proj,
                self.down_proj,
                routing_weights,
                selected_experts,
                final_hidden_states,
            )
        return final_hidden_states, router_logits
    
    # def forward_normal(
    #     self,
    #     hidden_states: torch.Tensor
    # ) -> torch.Tensor:
    #     sequence_length, hidden_dim = hidden_states.shape
    #     router_logits = self.gate(hidden_states)
    #     final_hidden_states = self.experts(hidden_states, router_logits)

    #     return final_hidden_states, router_logits
    
    # def forward_deepep(
    #     self,
    #     hidden_states: torch.Tensor
    # ) -> torch.Tensor:
    #     sequence_length, hidden_dim = hidden_states.shape
    #     router_logits = self.gate(hidden_states)
    #     topk_weights, topk_idx = select_experts(
    #         hidden_states=hidden_states,
    #         router_logits=router_logits,
    #         top_k=self.top_k,
    #         use_grouped_topk=False,
    #         renormalize=self.renormalize,
    #         expert_location_dispatch_info=None,
    #     )
    #     final_hidden_states = self.experts(
    #         hidden_states=hidden_states,
    #         topk_idx=topk_idx,
    #         topk_weights=topk_weights,
    #         reorder_topk_ids=None,
    #         seg_indptr=None,
    #         masked_m=None,
    #         expected_m=None,
    #         num_recv_tokens_per_expert=None,
    #     )
    #     # if self.config.ep_size > 1:
    #     #     (
    #     #         hidden_states,
    #     #         topk_idx,
    #     #         topk_weights,
    #     #         reorder_topk_ids,
    #     #         num_recv_tokens_per_expert,
    #     #         seg_indptr,
    #     #         masked_m,
    #     #         expected_m,
    #     #     ) = self.deepep_dispatcher.dispatch(
    #     #         hidden_states,
    #     #         topk_idx,
    #     #         topk_weights
    #     #     )
    #     # final_hidden_states = self.experts(
    #     #     hidden_states=hidden_states,
    #     #     topk_idx=topk_idx,
    #     #     topk_weights=topk_weights,
    #     #     reorder_topk_ids=reorder_topk_ids,
    #     #     seg_indptr=seg_indptr,
    #     #     masked_m=masked_m,
    #     #     expected_m=expected_m,
    #     #     num_recv_tokens_per_expert=num_recv_tokens_per_expert,
    #     # )
    #     # if self.ep_size > 1:
    #     #     final_hidden_states = self.deepep_dispatcher.combine(
    #     #         final_hidden_states,
    #     #         topk_idx,
    #     #         topk_weights,
    #     #     )
    #     return final_hidden_states, router_logits


class TBStars2MoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.self_attn = CausalAttention(config, weights, layer_idx)

        if len(config.moe_layer_index) > 0 and layer_idx < config.moe_layer_index[0]:
            self.is_dense_layer = True
        else:
            self.is_dense_layer = False
            self.moe_mlp = TBStars2MoESparseBlock(config, weights, layer_idx)
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
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
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
            [
                TBStars2MoEDecoderLayer(config, weights.weights[idx], idx)
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self.lm_head = Linear(weights.get_global_weight(W.lm_head))

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        logging.error(f'vvv in TBStars2MoEModel, input_ids = {input_ids}')

        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs

        fmha_impl = self.get_fmha_impl(attention_inputs)

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)

        return PyModelOutputs(hidden_states)


__all__ = [
    "TBStars2MoEModel",
]

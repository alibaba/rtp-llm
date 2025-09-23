import logging
from typing import Dict, Optional, Tuple

import torch
from torch import nn

import rtp_llm.models_py.modules.utils as utils
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.utils.model_weight import W

if utils.is_cuda():
    from libth_transformer.rtp_llm_ops import FusedMoEOp, SelectTopkOp

    # Import EP MoE implementations
    from rtp_llm.models_py.modules.ep.ep_moe import create_ep_moe_instance
else:
    logging.info("can't import from rtp_llm_ops and ep.layers, only support cuda!")
    FusedMoEOp = None
    SelectTopkOp = None
    create_ep_moe_instance = None


class ModelMoESparseBlock(nn.Module):
    """Generic MoE sparse block implementation."""

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = 0,
    ):
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
            # Use EP MoE for FP8 quantized models
            self.experts = create_ep_moe_instance(config, weights, layer_idx)
            self.use_fp8_path = True
        else:
            # Use traditional MoE path
            self.gate = LinearFactory.create_linear_from_weights(
                weights, W.moe_gate, None, None, config
            )
            if SelectTopkOp is not None and FusedMoEOp is not None:
                # Use fused ops if available
                self.up_proj = weights.get(W.moe_w1, None)
                self.down_proj = weights.get(W.moe_w2, None)
                self.select_topk_op = SelectTopkOp(config)
                self.fused_moe_op = FusedMoEOp(config)
                self.use_fp8_path = False
            else:
                # Use EP MoE implementation
                self.experts = create_ep_moe_instance(config, weights, layer_idx)
                self.use_fp8_path = True

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

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        router_logits = self.gate(hidden_states)

        if self.use_fp8_path:
            # Use EP MoE path
            final_hidden_states = self.experts(hidden_states, router_logits)
        else:
            # Use fused ops path
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


__all__ = [
    "ModelMoESparseBlock",
]

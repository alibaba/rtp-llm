from typing import Dict

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.utils.model_weight import W


class DenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()

        # Create linear layers using LinearFactory
        self.gate_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w1, W.ffn_s1, W.ffn_b1, config
        )
        self.up_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w3, W.ffn_s3, W.ffn_b3, config
        )
        self.down_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, config
        )

        if config.activation_type == "SiGLU":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {config.activation_type}")

    def forward(self, x: torch.Tensor):
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        activated = self.act_fn(gate_output)
        product = activated * up_output
        down_output = self.down_proj(product)
        return down_output


class BertGeluActDenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()

        # For BERT model, use traditional FFN structure with GeLU activation
        # BERT uses: intermediate_weight3 -> GeLU -> intermediate_weight2
        self.intermediate_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w3, W.ffn_s3, W.ffn_b3, config
        )
        self.output_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, config
        )

        # Use GeLU activation
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor):
        # Traditional BERT FFN: intermediate -> GeLU -> output
        intermediate = self.intermediate_proj(x)
        activated = self.act_fn(intermediate)
        output = self.output_proj(activated)
        return output

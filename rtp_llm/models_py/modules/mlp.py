import torch
from torch import nn
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from typing import Optional, Dict
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.utils.model_weight import W

from libth_transformer import rtp_llm_ops

class DenseMLP(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]):
        super().__init__()

        self.gate_proj = Linear(weights[W.ffn_w1], weights.get(W.ffn_b1, None))
        self.up_proj = Linear(weights[W.ffn_w3], weights.get(W.ffn_b3, None))
        self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))

        if config.activation_type == "SiGLU":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {config.activation_type}")

    def forward(self, x: torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class FusedSiluActDenseMLP(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]):
        super().__init__()
        assert config.activation_type == "SiGLU", "FusedSiluActDenseMLP only supports SiGLU activation"

        gate_proj_bias = weights.get(W.ffn_b1, None)
        up_proj_bias = weights.get(W.ffn_b3, None)
        gate_up_proj_bias = None
        if gate_proj_bias is not None and up_proj_bias is not None:
            gate_up_proj_bias = torch.cat([gate_proj_bias, up_proj_bias], dim=-1)
        gate_up_proj_weight = torch.cat([weights[W.ffn_w1], weights[W.ffn_w3]], dim=-1)
        self.gate_up_proj = Linear(gate_up_proj_weight, gate_up_proj_bias)
        self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))

    def forward(self, x: torch.Tensor):
        gate_up = self.gate_up_proj(x)

        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        rtp_llm_ops.silu_and_mul(output, gate_up, 0)
        down_proj = self.down_proj(output)
        return down_proj

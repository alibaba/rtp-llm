import torch
from torch import nn
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from typing import Optional, Dict
from rtp_llm.models_py.modules import Linear
from rtp_llm.utils.model_weight import W

class Qwen3MLP(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]):
        super().__init__()
        self.gate_proj = Linear(weights[W.ffn_w1], weights.get(W.ffn_b1, None))
        self.up_proj = Linear(weights[W.ffn_w3], weights.get(W.ffn_b3, None))
        self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

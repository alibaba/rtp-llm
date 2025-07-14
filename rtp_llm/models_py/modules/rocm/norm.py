import torch
from aiter import rmsnorm2d_fwd as rms_norm

from rtp_llm.models_py.modules.norm import BaseNorm


class RMSNorm(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states: torch.Tensor):
        return rms_norm(hidden_states, self.weight.data, self.variance_epsilon)

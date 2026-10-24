"""MiniMax-M3.1 dense and shared expert SwiGLU-OAI."""

import torch
from torch import nn

from rtp_llm.models_py.model_desc.minimax_m31_activation import swiglu_oai
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

_cuda_swiglu_oai = torch.compile(swiglu_oai)


class M31OAIActivation(nn.Module):
    def __init__(self, alpha: float, limit: float):
        super().__init__()
        if alpha <= 0 or limit <= 0:
            raise ValueError("M3.1 SwiGLU-OAI requires positive alpha and limit")
        self.alpha = alpha
        self.limit = limit

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        if gate_up.is_cuda:
            return _cuda_swiglu_oai(gate_up, self.alpha, self.limit)
        return swiglu_oai(gate_up, self.alpha, self.limit)


class M31DenseMLP(DenseMLP):
    """Use main's projections and collectives with M3.1's activation math."""

    def __init__(
        self,
        activation_type,
        parallelism_config,
        weights,
        quant_config,
        hw_kernel_config=None,
        *,
        alpha: float,
        limit: float,
    ):
        super().__init__(
            activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        if not self.is_gated:
            raise ValueError("M3.1 dense MLP requires gated projections")
        self.act_fn = M31OAIActivation(alpha, limit)

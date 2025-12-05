"""CUDA F16 (non-quantized) Linear implementation"""

from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.linear import LinearBase


class CudaF16Linear(LinearBase):
    """CUDA F16 (non-quantized) Linear"""

    @classmethod
    def can_handle(
        cls,
        config: Optional[GptInitModelParameters],
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
    ) -> bool:
        """Handle non-FP8 cases (no weight_scales)"""
        return weight_scales is None

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        config: Optional[GptInitModelParameters] = None,
    ):
        super().__init__(weight, weight_scales, input_scales, bias, config)
        self.weight = weight.T
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

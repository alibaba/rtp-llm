"""ROCm F16 (non-quantized) Linear implementation"""

from typing import Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops.compute_ops import rtp_llm_ops


class RocmF16Linear(LinearBase):
    """ROCm F16 (non-quantized) Linear"""

    @classmethod
    def can_handle(
        cls,
        config: Optional[GptInitModelParameters],
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
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
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(weight, weight_scales, input_scales, bias, config, weight_scale_2)
        self.weight = weight
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            *input.shape[:-1],
            self.weight.shape[1],
            dtype=input.dtype,
            device=input.device
        )
        rtp_llm_ops.gemm(output, input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

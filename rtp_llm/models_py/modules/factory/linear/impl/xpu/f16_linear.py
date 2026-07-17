"""XPU F16/BF16 (non-quantized) Linear implementation.

Uses PyTorch F.linear on Intel XPU. Identical to the CUDA F16 implementation
since PyTorch handles device dispatch internally.
"""

from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig


class XpuF16Linear(LinearBase):
    """XPU F16/BF16 (non-quantized) Linear using PyTorch ops."""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional['HWKernelConfig'] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        """Handle non-quantized weights (weight_scales is None), matching the
        CUDA F16Linear contract.  A model-level quant_config does not by itself
        make a given weight quantized; whether a quantized weight is handled is
        decided by weight_scales / the specific quant strategy, so gating on a
        non-empty quant_config here would starve unquantized layers (lm_head,
        unquantized blocks) of any candidate strategy.
        """
        return weight_scales is None

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(weight, weight_scales, input_scales,
                         bias, quant_config, weight_scale_2)
        self.weight = weight.T
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

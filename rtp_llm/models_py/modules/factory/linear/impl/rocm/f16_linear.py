"""ROCm F16 (non-quantized) Linear implementation"""

import os
from typing import Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.linear import LinearBase
from aiter import hipb_mm, hipb_create_extension
from functools import lru_cache

class RocmF16Linear(LinearBase):
    """ROCm F16 (non-quantized) Linear"""

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
        self.weight = weight
        self.bias = bias
        
    @staticmethod    
    @lru_cache(maxsize=1)
    def init_hipblas():
        hipb_create_extension()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.init_hipblas()
        has_swizzle = os.environ.get("USE_SWIZZLEA", None) == "1"
        return hipb_mm(
            input,
            self.weight,
            solution_index=-1,
            bias=self.bias,
            out_dtype=input.dtype,
            scaleA=None,
            scaleB=None,
            scaleOut=None,
            bpreshuffle=has_swizzle,
        )

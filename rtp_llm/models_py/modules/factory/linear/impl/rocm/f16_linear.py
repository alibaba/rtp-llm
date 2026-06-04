"""ROCm F16 (non-quantized) Linear implementation"""

import os
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase
from aiter import hipb_mm, hipb_create_extension
from functools import lru_cache
from rtp_llm.ops import HWKernelConfig

class RocmF16LinearBase(LinearBase):
    """ROCm F16 (non-quantized) Linear"""

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
        raise NotImplementedError("Subclasses must implement `can_handle`.")

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
        self.weight = weight
        self.bias = bias
        
    @staticmethod    
    @lru_cache(maxsize=1)
    def init_hipblas():
        hipb_create_extension()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement `forward`.")

        
class RocmF16LinearWithSwizzle(RocmF16LinearBase):

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional['HWKernelConfig'],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        # FP16/BF16 (unquantized) weights are never swizzled: hipBLASLt has no
        # bpreshuffle heuristic for them on gfx942 (HIPBLAS_STATUS_INTERNAL_ERROR).
        # Only FP8 keeps swizzleA; unquantized always uses the no-swizzle path.
        return False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.init_hipblas()
        return hipb_mm(
            input,
            self.weight,
            solution_index=-1,
            bias=self.bias,
            out_dtype=input.dtype,
            scaleA=None,
            scaleB=None,
            scaleOut=None,
            bpreshuffle=True,
        )

class RocmF16LinearNoSwizzle(RocmF16LinearBase):

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional['HWKernelConfig'],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        # Unquantized FP16/BF16 always uses the no-swizzle path; swizzleA is FP8-only
        # (hipBLASLt has no bpreshuffle heuristic for FP16 on gfx942).
        return weight_scales is None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.init_hipblas()
        return hipb_mm(
            input,
            self.weight,
            solution_index=-1,
            bias=self.bias,
            out_dtype=input.dtype,
            scaleA=None,
            scaleB=None,
            scaleOut=None,
            bpreshuffle=False,
        )

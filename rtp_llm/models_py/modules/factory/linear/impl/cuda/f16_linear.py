"""CUDA F16 (non-quantized) Linear implementation.

When USE_ONLINE_FP4GEMM=1, layers with K%128==0 are deferred to
CudaOnlineMxfp4Linear for online MXFP4 quantization + mm_fp4 GEMM.
"""

import os
from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig

_MXFP4_ONLINE = os.environ.get("USE_ONLINE_FP4GEMM", "0") == "1"


class CudaF16Linear(LinearBase):
    """CUDA F16 (non-quantized) Linear"""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if weight_scales is not None:
            return False
        if _MXFP4_ONLINE and weight.dtype in (torch.bfloat16, torch.float16):
            if weight.dim() == 2 and weight.shape[0] % 128 == 0:
                return False
        return True

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self.weight = weight.T
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

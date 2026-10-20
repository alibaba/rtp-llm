"""CUDA F16 (non-quantized) Linear implementation"""

from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.models_py.modules.factory.linear.triton_bias_gelu import linear_bias_gelu
from rtp_llm.ops import HWKernelConfig

class CudaF16Linear(LinearBase):
    """CUDA F16 (non-quantized) Linear"""

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
        """Handle non-FP8 and non-FP4 cases (no weight_scales)"""
        return weight_scales is None

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None
    ):
        super().__init__(weight, weight_scales, input_scales,
                         bias, quant_config, weight_scale_2)
        self.weight = weight.T
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def forward_with_bias_gelu(self, input: torch.Tensor) -> torch.Tensor:
        """Fused GEMM + bias + GELU on CUDA; LinearBase default elsewhere.

        ``forward`` folds the bias into the cuBLASLt GEMM epilogue, so the
        base implementation would run ATen ``F.gelu`` on the biased GEMM
        (one native elementwise kernel per MLP per step on the decode
        timeline). Instead the GEMM is issued without the bias and one
        Triton kernel applies ``gelu(x @ W^T + bias)`` with the bias add
        and the exact erf GELU fused into a single launch; see
        ``triton_bias_gelu.py`` for the numerics contract, shape handling
        and the bit-exact fallback conditions (non-CUDA tensors, missing
        triton, ``RTP_LLM_LINEAR_BIAS_GELU_TRITON=0``).
        """
        return linear_bias_gelu(input, self.weight, self.bias)

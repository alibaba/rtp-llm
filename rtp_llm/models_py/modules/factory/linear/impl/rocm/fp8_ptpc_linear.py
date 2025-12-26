"""ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear implementation"""

import logging
from typing import Optional

import aiter
import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
from rtp_llm.ops import HWKernelConfig

class RocmFp8PTPCLinear(LinearBase):
    """ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear"""

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
        """Handle FP8_PER_CHANNEL_COMPRESSED"""
        if weight_scales is None or quant_config is None:
            return False

        # Check if weight is FP8 format
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False

        # Check quantization method
        quant_method = quant_config.get_method()
        return quant_method == "FP8_PER_CHANNEL_COMPRESSED"

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
        self.hidden_size = weight.shape[0]  # k
        self.output_size = weight.shape[1]  # n
        # Reshape weight from [k, n] to [n, k] as done in C++ code
        self.weight = weight.reshape([weight.shape[1], weight.shape[0]])
        self.weight_scales = weight_scales.reshape(
            [weight_scales.shape[1], weight_scales.shape[0]]
        )
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_dtype = input.dtype
        # Convert to BF16 if needed
        if input.dtype != torch.bfloat16:
            input_bf16 = input.to(torch.bfloat16)
        else:
            input_bf16 = input

        # Get input dimensions
        M = input_bf16.shape[0]
        N = self.output_size

        # Pre-allocate output tensor
        output = torch.empty((M, N), dtype=torch.bfloat16, device=input_bf16.device)

        quantization_eps = 1e-10
        # Use per-token quantization (not per-token-block)
        input_fp8, input_scales = rocm_per_token_quant_fp8(
            input_bf16,
            eps=quantization_eps,
        )

        input_scales = input_scales.to(torch.float32)

        # Use per-token scales (M, 1)
        x_scales = input_scales
        w_scales = self.weight_scales

        output = aiter.gemm_a8w8_bpreshuffle(
            input_fp8,  # A_quant_tensor
            self.weight,  # W_kernel_tensor (already reshaped to [n, k])
            x_scales,  # A_quant_scale_tensor (M, 1)
            w_scales,  # W_scale_tensor (reshaped)
            None,
            input_bf16.dtype,
        )

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        # Convert back to original dtype if needed
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output

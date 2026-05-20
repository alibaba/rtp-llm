"""ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear implementation"""

import logging
from typing import Optional

import aiter
import torch
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_cktile

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
        """Handle FP8_PER_CHANNEL_COMPRESSED and FP8_PER_CHANNEL_QUARK"""
        if weight_scales is None or quant_config is None:
            return False

        # Check if weight is FP8 format
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False

        # Check quantization method
        quant_method = quant_config.get_method()
        return quant_method in ("FP8_PER_CHANNEL_COMPRESSED", "FP8_PER_CHANNEL_QUARK")

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
        if weight_scales is None:
            raise ValueError("weight_scales is required for RocmFp8PTPCLinear")

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

        K = input_fp8.shape[-1]
        # Dispatch rules (validated on MI308X, sweep M=[1..16384], N={1024,2816}, K=1024):
        # - K < 192                     : aiter cktile (small-K kernel)
        # - K >= 192, M >= 1536         : aiter cktile FlatmmKernel (small-N crossover at M~1536)
        # - K >= 192, M >= 512, N > 1536: aiter cktile (large-N crossover at M~512, +22%)
        # - otherwise                   : aiter default (decode-friendly, protects M<=256)
        #
        # Rationale: cktile's 128x128x128 tile amortizes LDS setup cost only when
        # there are enough output elements. The crossover M depends on N:
        #   - N <= 1536 (small): cktile wins at M~1536 (+23% at N=1024)
        #   - N >  1536 (large): cktile wins at M~512  (+22% at N=2816)
        # M < 512 always stays on default to protect decode (M<=256, +40~97% regression).
        use_cktile = K < 192 or M >= 1536 or (M >= 512 and N > 1536)
        if use_cktile:
            output = torch.empty(
                (M, N), dtype=input_bf16.dtype, device=input_bf16.device
            )
            gemm_a8w8_bpreshuffle_cktile(
                input_fp8, self.weight, x_scales, w_scales, output
            )
        else:
            output = aiter.gemm_a8w8_bpreshuffle(
                input_fp8,
                self.weight,
                x_scales,
                w_scales,
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

    def forward_prequantized(
        self,
        input_fp8: torch.Tensor,
        input_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Run FP8 PTPC linear with pre-quantized input.

        ROCm fused RMSNorm kernels can produce per-token FP8 activations and
        scales directly. This path reuses those tensors and skips the duplicate
        per-token quantization performed by forward().
        """
        token_num = input_fp8.shape[0]
        output_size = self.output_size
        hidden_size = input_fp8.shape[-1]

        if input_scales.dtype != torch.float32:
            input_scales = input_scales.to(torch.float32)

        x_scales = input_scales
        w_scales = self.weight_scales

        use_cktile = (
            hidden_size < 192
            or token_num >= 1536
            or (token_num >= 512 and output_size > 1536)
        )
        if use_cktile:
            output = torch.empty(
                (token_num, output_size), dtype=torch.bfloat16, device=input_fp8.device
            )
            gemm_a8w8_bpreshuffle_cktile(
                input_fp8, self.weight, x_scales, w_scales, output
            )
        else:
            output = aiter.gemm_a8w8_bpreshuffle(
                input_fp8,
                self.weight,
                x_scales,
                w_scales,
                None,
                torch.bfloat16,
            )

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        return output

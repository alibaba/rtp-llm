import logging
from typing import Optional

import aiter
import torch
from torch import nn

from rtp_llm.config.quant_config import QuantizationConfig

logger = logging.getLogger(__name__)

try:
    from rtp_llm.models_py.modules.rocm.fp8_kernel import (
        rocm_per_token_group_quant_fp8,
        rocm_per_token_quant_fp8,
    )

    FP8_AVAILABLE = True
    AITER_GEMM_AVAILABLE = True
except ImportError:
    AITER_GEMM_AVAILABLE = False


class Fp8DeepGEMMLinear(nn.Module):
    """FP8 Linear layer with DeepGEMM quantized matrix multiplication."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        config=None,
    ) -> None:
        super().__init__()
        self.hidden_size = weight.shape[0]  # k
        self.output_size = weight.shape[1]  # n
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

        # Check FP8 availability
        if not FP8_AVAILABLE:
            error_msg = (
                "FP8 quantization is not available but required for Fp8DeepGEMMLinear. "
                "Please ensure FP8 kernel is properly installed and imported."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not AITER_GEMM_AVAILABLE:
            error_msg = (
                "aiter GEMM is not available but required for FP8 computation. "
                "Please ensure aiter is properly installed and imported."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Get input dimensions
        M = input_bf16.shape[0]
        N = self.output_size

        # DeepGEMM quantization: per-token-block quantization + gemm_a8w8_blockscale
        logger.debug("Fp8DeepGEMMLinear: Using gemm_a8w8_blockscale (DeepGEMM)")

        quantization_eps = 1e-4
        input_fp8, input_scales = rocm_per_token_group_quant_fp8(
            input_bf16,
            group_size=128,
            eps=quantization_eps,
            column_major_scales=False,
            scale_tma_aligned=False,
        )

        # Clamp scales to avoid numerical issues
        FP8_E4M3_MAX = 448.0
        min_scale_threshold = 1e-4 / FP8_E4M3_MAX
        input_scales = torch.clamp(input_scales, min=min_scale_threshold)
        input_scales = input_scales.to(torch.float32)

        # Use per-token-block scales directly (M, K/128)
        x_scales = input_scales
        w_scales = self.weight_scales

        try:
            output = aiter.gemm_a8w8_blockscale(
                input_fp8,  # XQ
                self.weight,  # WQ
                x_scales,  # x_scale
                w_scales,  # w_scale
            )
        except Exception as e:
            error_msg = (
                f"aiter gemm_a8w8_blockscale call failed: {type(e).__name__}: {e}"
            )
            logger.error(error_msg)
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        # Convert back to original dtype if needed
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output


class Fp8PTPCLinear(nn.Module):
    """
    FP8 Linear layer with Per-Token Per-Channel (PTPC) quantization.
    Mimics InvokeROCmPTPCGemm from ROCmGemmOp.cc
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        config=None,
    ) -> None:
        super().__init__()
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

        # Check FP8 availability
        if not FP8_AVAILABLE:
            error_msg = (
                "FP8 quantization is not available but required for Fp8PTPCLinear. "
                "Please ensure FP8 kernel is properly installed and imported."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not AITER_GEMM_AVAILABLE:
            error_msg = (
                "aiter GEMM is not available but required for FP8 PTPC computation. "
                "Please ensure aiter is properly installed and imported."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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

        # Clamp scales to avoid numerical issues
        # FP8_E4M3_MAX = 448.0
        # min_scale_threshold = 1e-10 / FP8_E4M3_MAX
        # input_scales = torch.clamp(input_scales, min=min_scale_threshold)
        input_scales = input_scales.to(torch.float32)

        # Use per-token scales (M, 1)
        x_scales = input_scales
        w_scales = self.weight_scales
        try:
            output = aiter.gemm_a8w8_bpreshuffle(
                input_fp8,  # A_quant_tensor
                self.weight,  # W_kernel_tensor (already reshaped to [n, k])
                x_scales,  # A_quant_scale_tensor (M, 1)
                w_scales,  # W_scale_tensor (reshaped)
                None,
                input_bf16.dtype,
            )
        except Exception as e:
            error_msg = (
                f"aiter gemm_a8w8_bpreshuffle call failed: {type(e).__name__}: {e}"
            )
            logger.error(error_msg)
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        # Convert back to original dtype if needed
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output

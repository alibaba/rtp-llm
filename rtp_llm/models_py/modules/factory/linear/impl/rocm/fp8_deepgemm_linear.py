"""ROCm FP8 DeepGEMM quantized Linear implementation"""

import logging
from typing import Optional

import aiter
import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_group_quant_fp8


class RocmFp8DeepGEMMLinear(LinearBase):
    """ROCm FP8 DeepGEMM quantized Linear"""

    @classmethod
    def can_handle(
        cls,
        config: Optional[GptInitModelParameters],
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
    ) -> bool:
        """Handle other FP8 methods (not PTPC)"""
        if weight_scales is None or config is None:
            return False

        if not hasattr(config, "quant_config") or config.quant_config is None:
            return False

        # Check if weight is FP8 format
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False

        # Check quantization method - handle FP8 methods that are NOT PTPC
        if hasattr(config.quant_config, "get_method"):
            quant_method = config.quant_config.get_method()
            return quant_method == "FP8_PER_BLOCK"
        return False

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        config: Optional[GptInitModelParameters] = None,
    ):
        super().__init__(weight, weight_scales, input_scales, bias, config)
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

        output = aiter.gemm_a8w8_blockscale(
            input_fp8,  # XQ
            self.weight,  # WQ
            x_scales,  # x_scale
            w_scales,  # w_scale
        )

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        # Convert back to original dtype if needed
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output

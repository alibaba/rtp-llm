"""CUDA FP8 DeepGEMM quantized Linear implementation"""

import logging
from typing import Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    fp8_gemm_nt,
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)


class CudaFp8DeepGEMMLinear(LinearBase):
    """CUDA FP8 DeepGEMM quantized Linear"""

    @classmethod
    def can_handle(
        cls,
        config: Optional[GptInitModelParameters],
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
    ) -> bool:
        """Handle other FP8 methods (FP8, FP8_PER_BLOCK, etc.)"""
        if weight_scales is None or config is None:
            return False

        if not hasattr(config, "quant_config") or config.quant_config is None:
            return False

        # Check if weight is FP8 format
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False

        # Check quantization method - handle all other FP8 methods
        if hasattr(config.quant_config, "get_method"):
            quant_method = config.quant_config.get_method()
            return quant_method == "FP8_PER_BLOCK"
        return False

    @torch.inference_mode()
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        config: Optional[GptInitModelParameters] = None,
    ):
        super().__init__(weight, weight_scales, input_scales, bias, config)
        if not has_deep_gemm():
            raise RuntimeError(
                "DeepGEMM is not available. Please install the `deep_gemm` package to enable DeepGEMM kernels."
            )
        self.hidden_size = weight.shape[0]  # k
        self.output_size = weight.shape[1]  # n
        self.weight = weight.reshape([weight.shape[1], weight.shape[0]])
        self.weight_scales = weight_scales.reshape(
            [weight_scales.shape[1], weight_scales.shape[0]]
        )
        self.bias = bias
        self.scale_ue8m0 = is_deep_gemm_e8m0_used()
        # Disable UE8M0 for small tensors due to performance/accuracy trade-offs.
        # TODO: Re-evaluate this threshold after further optimization of UE8M0 kernels.
        if self.hidden_size < 512 or self.output_size < 128:
            self.scale_ue8m0 = False
        if self.scale_ue8m0:
            w_tmp, self.weight_scales = requant_weight_ue8m0(
                self.weight, self.weight_scales
            )
            self.weight.copy_(w_tmp)
            del w_tmp

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get input dimensions
        input_m = input.shape[0]
        input_k = input.shape[1]
        output_n = self.output_size

        # Check input dtype - only accept BF16
        if input.dtype != torch.bfloat16:
            error_msg = (
                f"CudaFp8DeepGEMMLinear only accepts bfloat16 input, but got {input.dtype}. "
                "Please convert input to bfloat16 before calling Fp8DeepGEMMLinear."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        input_bf16 = input

        # Quantize input to FP8
        alignment = self._get_padding_size(input_m)
        target_m = (input_m + alignment - 1) // alignment * alignment
        need_padding = target_m > input_m

        if need_padding:
            input_for_quant = torch.zeros(
                target_m, input_k, dtype=torch.bfloat16, device=input.device
            )
            input_for_quant[:input_m, :] = input_bf16
        else:
            input_for_quant = input_bf16

        # Quantize using sgl_per_token_group_quant_fp8
        quantization_eps = 1e-4
        if self.scale_ue8m0:
            input_fp8, input_scales = sgl_per_token_group_quant_fp8(
                input_for_quant,
                128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        else:
            input_fp8, input_scales = sgl_per_token_group_quant_fp8(
                input_for_quant,
                group_size=128,
                eps=quantization_eps,
                column_major_scales=self.scale_ue8m0,
                scale_tma_aligned=self.scale_ue8m0,
                scale_ue8m0=self.scale_ue8m0,
            )
            input_fp8 = input_fp8.contiguous()
            input_scales = input_scales.contiguous()
        if not self.scale_ue8m0:
            FP8_E4M3_MAX = 448.0
            min_scale_threshold = 1e-4 / FP8_E4M3_MAX
            input_scales = torch.clamp(input_scales, min=min_scale_threshold)
            input_scales = input_scales.to(torch.float32).contiguous()
        output_m = input_for_quant.shape[0]
        output = torch.zeros(
            output_m, output_n, dtype=torch.bfloat16, device=input.device
        )

        # Call DeepGEMM
        weight = self.weight
        weight_scales = self.weight_scales
        output = output.contiguous()

        fp8_gemm_nt(
            (input_fp8, input_scales),
            (weight, weight_scales),
            output,
            c=None,
            disable_ue8m0_cast=not self.scale_ue8m0,
        )

        if need_padding:
            output = output[:input_m, :]
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    def _get_padding_size(self, m):
        """Calculate padding size based on DeepGEMM requirements."""
        if self._gemm_swap_ab_heuristic(m):
            if m < 16:
                return 16
            else:
                return 8
        else:
            return 64

    def _gemm_swap_ab_heuristic(self, m):
        return False

"""CUDA FP4 quantized Linear implementation using flashinfer mm_fp4 operator"""

import logging
import os
from typing import Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.cuda.fp4_kernel import (
    cutlass_scaled_fp4_mm_wrapper,
    # scaled_fp4_quant_wrapper,
)

try:
    from flashinfer import (
        mm_fp4,
        fp4_quantize,
    )
    enable_flashinfer_fp4_gemm = True
except ImportError:
    enable_flashinfer_fp4_gemm = False
    logger.warning("flashinfer with FP4 support is not available")


def has_flashinfer_fp4() -> bool:
    """Check if flashinfer FP4 support is available."""
    return enable_flashinfer_fp4_gemm


class CudaFp4GEMMLinear(LinearBase):
    """CUDA FP4 quantized Linear"""

    @classmethod
    def can_handle(
        cls,
        config: Optional[GptInitModelParameters],
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        """Check if this strategy can handle the given configuration"""
        if weight_scales is None or config is None or \
            weight_scale_2 is None or input_scale is None:
            return False

        if not hasattr(config, "quant_config") or config.quant_config is None:
            return False

        if weight.dtype not in (torch.uint8) or \
            weight_scales.dtype not in (torch.float8_e4m3fn):
            return False

        # Check quantization method
        if hasattr(config.quant_config, "get_method"):
            quant_method = config.quant_config.get_method()
            return quant_method == "modelopt_fp4"

        return False

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        config: Optional[GptInitModelParameters] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(weight, weight_scales, input_scales, bias, config, weight_scale_2)

        if not has_flashinfer_fp4():
            raise RuntimeError(
                "flashinfer with FP4 support is not available. "
                "Please install flashinfer with FP4 support."
            )

        self.hidden_size = weight.shape[0] * 2  # k
        self.output_size = weight.shape[1]  # n
        self.weight = weight.T
        self.weight_scales = weight_scales.T
        self.weight_scale_2 = weight_scale_2
        self.input_scale = input_scales
        self.bias = bias

        self.backend = os.getenv("RTP_LLM_FP4_GEMM_BACKEND", "cutlass")

        if self.backend == "trtllm" and enable_flashinfer_fp4_gemm:
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a
            epilogue_tile_m = 128
            self.weight = shuffle_matrix_a(self.weight.view(torch.uint8), epilogue_tile_m)
            self.weight_scales = (
                shuffle_matrix_sf_a(self.weight_scales.view(torch.uint8), epilogue_tile_m)
                .reshape(self.weight_scales.shape)
                .view(torch.float8_e4m3fn)
            )
        else:
            # Pad and blockwise interleave weight_scales
            scales = self.weight_scales
            scale_ndim = scales.ndim
            if scale_ndim == 2:
                scales = scales.unsqueeze(0)
            assert scales.ndim == 3
            B, M, K = scales.shape
            round_up_multiple = lambda x, m: (x + m - 1) // m * m
            M_padded = round_up_multiple(M, 128)
            K_padded = round_up_multiple(K, 4)
            padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
            padded_scales[:B, :M, :K] = scales
            batches, rows, cols = padded_scales.shape
            assert rows % 128 == 0
            assert cols % 4 == 0
            padded_scales = padded_scales.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
            padded_scales = padded_scales.permute((0, 1, 4, 3, 2, 5))
            padded_scales = padded_scales.contiguous().cuda()
            padded_scales = (
                padded_scales.reshape(M_padded, K_padded)
                if scale_ndim == 2
                else padded_scales.reshape(B, M_padded, K_padded)
            )
            self.weight_scales = padded_scales

        self.alpha = self.weight_scale_2 * self.input_scale
        self.input_scale_inv = 1 / self.input_scale
        self.weight = self.weight.T
        self.weight_scales = self.weight_scales.T

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get input dimensions
        input_m = input.shape[0]
        output_n = self.output_size
        output_dtype = input.dtype
        output_shape = [input_m, output_n]

        # Check input dtype - accept BF16 and FP16
        if input.dtype not in (torch.bfloat16, torch.float16):
            error_msg = (
                f"CudaFp4GEMMLinear accepts bfloat16 and float16 input, but got {input.dtype}. "
                "Please convert input to bfloat16 or float16 before calling CudaFp4GEMMLinear."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.backend == "trtllm" and input.dtype == torch.float16:
            error_msg = (
                "CudaFp4GEMMLinear with trtllm backend only supoorts bfloat16 input, "
                f"but got {input.dtype}. Please convert input to bfloat16 with trtllm backend."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Quantize BF16 or FP16 input to (FP4 and interleaved block scale)
        input_fp4, input_scale_interleaved = fp4_quantize(input, self.input_scale_inv)
        assert input_fp4.dtype == torch.uint8
         
        if self.backend == "sgl_cutlass":
            output = cutlass_scaled_fp4_mm_wrapper(
                input_fp4,
                self.weight.T,
                input_scale_interleaved.view(torch.float8_e4m3fn),
                self.weight_scales.T.view(torch.float8_e4m3fn),
                self.alpha,
                output_dtype
            ).view(*output_shape)
        else:
            output = mm_fp4(
                input_fp4,
                self.weight,
                input_scale_interleaved,
                self.weight_scales,
                self.alpha,
                output_dtype,
                backend=self.backend
            ).view(*output_shape)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output


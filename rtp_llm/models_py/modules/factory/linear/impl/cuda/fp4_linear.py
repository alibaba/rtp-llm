"""CUDA FP4 quantized Linear implementation using flashinfer mm_fp4 operator"""

import logging
import os
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.cuda.fp4_kernel import (
    cutlass_scaled_fp4_mm_wrapper,
)
from rtp_llm.ops import HWKernelConfig

from flashinfer import (
    mm_fp4,
    fp4_quantize,
)


class CudaFp4GEMMLinear(LinearBase):
    """CUDA FP4 quantized Linear"""

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
        """Check if this strategy can handle the given configuration"""
        if weight_scales is None or quant_config is None or \
            weight_scale_2 is None or input_scale is None:
            return False

        # Check quantization method
        quant_method = quant_config.get_method()
        return quant_method == "modelopt_fp4"

    @torch.inference_mode()
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
        # [n, k // 2]
        self.hidden_size = weight.shape[1] * 2  # k
        self.output_size = weight.shape[0]  # n
        self.weight = weight
        self.weight_scales = weight_scales
        self.weight_scale_2 = weight_scale_2
        self.input_scale = input_scales
        self.bias = bias
        self.backend = os.getenv("RTP_LLM_FP4_GEMM_BACKEND", "cutlass")
        self.alpha = self.weight_scale_2 * self.input_scale
        self.input_scale_inv = 1 / self.input_scale

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
                self.weight,
                input_scale_interleaved.view(torch.float8_e4m3fn),
                self.weight_scales.view(torch.float8_e4m3fn),
                self.alpha,
                output_dtype
            ).view(*output_shape)
        else:
            output = mm_fp4(
                input_fp4,
                self.weight.T,
                input_scale_interleaved,
                self.weight_scales.T,
                self.alpha,
                output_dtype,
                backend=self.backend
            ).view(*output_shape)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output


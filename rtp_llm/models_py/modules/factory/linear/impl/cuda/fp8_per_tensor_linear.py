"""CUDA FP8 Per-Tensor quantized Linear implementation"""

import logging
from typing import Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)


class CudaFp8PerTensorLinear(LinearBase):
    """CUDA FP8 Per-Tensor quantized Linear"""

    @classmethod
    def can_handle(
        cls,
        config: Optional[GptInitModelParameters],
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        """Handle FP8_PER_TENSOR_COMPRESSED and FP8_DYNAMIC_PER_TENSOR"""
        if weight_scales is None or config is None:
            return False

        if not hasattr(config, "quant_config") or config.quant_config is None:
            return False

        # Check if weight is FP8 format
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False

        # Check quantization method
        if hasattr(config.quant_config, "get_method"):
            quant_method = config.quant_config.get_method()
            return quant_method in [
                "FP8_PER_TENSOR_COMPRESSED",
                "FP8_DYNAMIC_PER_TENSOR",
            ]

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
        self.weight = weight.T
        self.weight_scale = weight_scales
        self.input_scale = input_scales
        self.bias = bias
        self.block_quant = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], self.weight.shape[1]]

        qinput, x_scale = scaled_fp8_per_tensor_quant(input_2d, self.input_scale)

        output = torch._scaled_mm(
            qinput,
            self.weight,
            out_dtype=input.dtype,
            scale_a=x_scale,
            scale_b=self.weight_scale,
            bias=self.bias,
        )

        if type(output) is tuple and len(output) == 2:
            output = output[0]

        return torch.narrow(output, 0, 0, input_2d.shape[0]).view(*output_shape)

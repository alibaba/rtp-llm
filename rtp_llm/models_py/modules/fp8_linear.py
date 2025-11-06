import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.models_py.modules.fp8_kernel import (
    scaled_fp8_per_tensor_quant,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.quantization.deepgemm_wrapper import fp8_gemm_nt

logger = logging.getLogger(__name__)


class Fp8PerBlockLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Initialize attributes
        self.weight = weight
        self.weight_scale = weight_scale
        self.bias = bias
        # Check weight and weight scale dimensions
        if self.weight.dim() != 2 or self.weight_scale.dim() != 2:
            error_msg = f"Weight and weight scale must be 2D tensors, but got weight dim: {self.weight.dim()} and weight scale dim: {self.weight_scale.dim()}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Reshape weight and weight scale
        self.K, self.N = self.weight.shape
        self.scale_K, self.scale_N = self.weight_scale.shape
        self.weight = self.weight.reshape(self.N, self.K)
        self.weight_scale = self.weight_scale.reshape(self.scale_N, self.scale_K)
        # Check weight scale sizes
        if self.scale_N * 128 != self.N or self.scale_K * 128 != self.K:
            error_msg = f"Weight scale dimension mismatch! Expected N: {self.N}, got {self.scale_N * 128}, expected K: {self.K}, got {self.scale_K * 128}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Check weight and weight scale dtypes
        if self.weight.dtype != torch.float8_e4m3fn:
            error_msg = f"Weight dtype must be float8_e4m3fn, got {self.weight.dtype}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.weight_scale.dtype != torch.float32:
            error_msg = (
                f"Weight scale dtype must be float32, got {self.weight_scale.dtype}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Check bias
        if self.bias is not None:
            if self.bias.dim() != 1 and self.bias.dim() != 2:
                error_msg = f"Bias dimension must be 1 or 2, got {self.bias.dim()}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if self.bias.shape[-1] != self.N:
                error_msg = (
                    f"Bias last dimension must be {self.N}, got {self.bias.shape[-1]}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            if self.bias.dim() == 2 and self.bias.shape[0] != 1:
                error_msg = f"Bias first dimension must be 1, got {self.bias.shape[0]}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if self.bias.dtype != torch.bfloat16:
                error_msg = f"Bias dtype must be bfloat16, got {self.bias.dtype}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check input dtype - only accept bfloat16
        if x.dtype != torch.bfloat16:
            error_msg = f"Input tensor dtype must be bfloat16, got {x.dtype}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Check input tensor dimension
        if x.dim() != 2:
            error_msg = f"Input tensor dimension must be 2, got {x.dim()}D tensor"
            logger.error(error_msg)
            raise ValueError(error_msg)
        M, K = x.shape
        # Check input tensor inner dimension expected to be K
        if K != self.K:
            error_msg = f"Input tensor inner dimension expected to be {self.K}, got {K}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Quantize x to FP8
        x_fp8, x_scales = sgl_per_token_group_quant_fp8(
            x,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
        )
        # Prepare output tensor
        output = torch.empty(M, self.N, dtype=torch.bfloat16, device=x.device)
        # Invoke DeepGEMM
        fp8_gemm_nt(
            (x_fp8, x_scales),
            (self.weight, self.weight_scale),
            output,
            c=None,
            disable_ue8m0_cast=True,
        )
        if self.bias is not None:
            output = output + self.bias
        return output


class Fp8PerTensorLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.weight = weight.T
        self.weight_scale = weight_scale
        self.input_scale = input_scale
        self.bias = bias
        self.quant_config = quant_config

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], self.weight.shape[1]]

        qinput, x_scale = scaled_fp8_per_tensor_quant(input_2d, self.input_scale)

        # TODO(serina.wzq): Use high performance kernel
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

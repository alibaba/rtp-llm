"""CUDA F16 (non-quantized) Linear implementation"""

from math import prod
from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig


def linear_add(
    input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Apply addmm to a same-shaped residual using PyTorch's math policy."""
    shape = (*input.shape[:-1], weight.shape[0])
    if tuple(residual.shape) != shape:
        raise ValueError("Projection residual must match the output shape")
    rows = prod(input.shape[:-1])
    x = input.reshape(rows, input.shape[-1])
    residual_2d = residual.reshape(rows, weight.shape[0])
    result = torch.addmm(residual_2d, x, weight.t())
    return result.reshape(shape)


def linear_add_inplace(
    input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Consume an owned residual in an addmm epilogue."""
    shape = (*input.shape[:-1], weight.shape[0])
    if tuple(residual.shape) != shape:
        raise ValueError("Projection residual must match the output shape")
    if not residual.is_contiguous():
        raise ValueError("in-place projection residual must be contiguous")
    rows = prod(input.shape[:-1])
    result = residual.reshape(rows, weight.shape[0]).addmm_(
        input.reshape(rows, input.shape[-1]), weight.t()
    )
    return result.reshape(shape)


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

    def forward(
        self, input: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if residual is not None:
            if self.bias is not None:
                raise ValueError("Residual-fused projections require bias-free weights")
            return linear_add(input, self.weight, residual)
        return F.linear(input, self.weight, self.bias)

    def supports_skip_head_mid(self, input: torch.Tensor,
                               head_splits: tuple[int, int, int]) -> bool:
        if len(head_splits) != 3 or min(head_splits) <= 0:
            return False
        left, middle, right = head_splits
        if (any(part % 64 for part in head_splits)
                or self.weight.shape[0] % (left + right)):
            return False
        if (self.bias is not None or input.ndim != 2 or not input.is_cuda
                or input.dtype != self.weight.dtype or input.dtype != torch.bfloat16
                or input.device != self.weight.device
                or input.shape[1] != self.weight.shape[1]
                or not input.is_contiguous() or not self.weight.T.is_contiguous()
                or torch.cuda.get_device_capability(input.device)[0] != 10):
            return False
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            has_bf16_gemm_nt_skip_head_mid,
        )

        return has_bf16_gemm_nt_skip_head_mid()

    def forward_skip_head_mid(
        self, input: torch.Tensor, head_splits: tuple[int, int, int],
        *, output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project each head around an untouched middle gap in a reusable buffer."""
        if not self.supports_skip_head_mid(input, head_splits):
            raise RuntimeError("BF16 skip-head-mid projection is unavailable")
        left, middle, right = head_splits
        heads = self.weight.shape[0] // (left + right)
        shape = (input.shape[0], heads * (left + middle + right))
        if output is None:
            output = torch.empty(shape, dtype=input.dtype, device=input.device)
        if (tuple(output.shape) != shape or output.dtype != input.dtype
                or output.device != input.device or not output.is_contiguous()
                or output.untyped_storage().data_ptr() in (
                    input.untyped_storage().data_ptr(),
                    self.weight.untyped_storage().data_ptr(),
                )):
            raise ValueError("BF16 skip-head-mid output buffer mismatch")
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            bf16_gemm_nt_skip_head_mid,
        )

        bf16_gemm_nt_skip_head_mid(input, self.weight, output, head_splits)
        return output

"""K3's BF16 projections with FP32 intermediate reductions."""

from math import prod

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.ops.compute_ops import rtp_llm_ops


def bf16_linear(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply an [out, in] weight without changing PyTorch's global math policy."""
    if input.is_cuda and input.dtype == weight.dtype == torch.bfloat16:
        shape = (*input.shape[:-1], weight.shape[0])
        rows = prod(input.shape[:-1])
        result = rtp_llm_ops.cublas_gemm_bf16_fp32_accum(
            input.reshape(rows, input.shape[-1]), weight
        )
        return result.reshape(shape)
    return F.linear(input, weight)


def bf16_linear_add(input: torch.Tensor, weight: torch.Tensor,
                    residual: torch.Tensor) -> torch.Tensor:
    """Apply native BF16 addmm to a same-shaped residual without changing math policy."""
    shape = (*input.shape[:-1], weight.shape[0])
    if tuple(residual.shape) != shape:
        raise ValueError("K3 projection residual must match the output shape")
    rows = prod(input.shape[:-1])
    x = input.reshape(rows, input.shape[-1])
    residual_2d = residual.reshape(rows, weight.shape[0])
    if input.is_cuda and input.dtype == weight.dtype == residual.dtype == torch.bfloat16:
        result = rtp_llm_ops.cublas_gemm_bf16_fp32_accum_add(x, weight, residual_2d)
    else:
        result = torch.addmm(residual_2d, x, weight.t())
    return result.reshape(shape)


class KimiK3Bf16Linear(CudaF16Linear):
    """Explicit K3 selection; deliberately absent from the public registry."""

    def forward(self, input: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if self.bias is not None:
            raise ValueError("K3 BF16 projections require bias-free weights")
        if residual is not None:
            return bf16_linear_add(input, self.weight, residual)
        return bf16_linear(input, self.weight)

class KimiK3LatentDownLinear(KimiK3Bf16Linear):
    """Native K3 SM103/SM107 latent-down plan, prepared before Graph capture."""

    def __init__(self, weight):
        super().__init__(weight)
        self._native = (
            self.weight.is_cuda
            and self.weight.dtype == torch.bfloat16
            and tuple(self.weight.shape) == (3584, 7168)
            and torch.cuda.get_device_capability(self.weight.device) in ((10, 3), (10, 7))
        )
        if self._native:
            from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path
            setup_cutlass_import_path()
            from .native_linear.skinny_gemm import (
                SkinnyGemmConfig, shape_dynamic_skinny_gemm,
            )
            if not hasattr(rtp_llm_ops, "kimi_k3_fused_a_gemm"):
                raise RuntimeError("K3 native latent-down requires CUDA13 fused-A binding")
            self.weight = self.weight.contiguous()
            self._skinny = shape_dynamic_skinny_gemm
            self._skinny_config = SkinnyGemmConfig(1, 224, 2, 4)
            self._skinny.request_warmup_configs(
                torch.bfloat16, (self._skinny_config,)
            )

    def forward(self, input, residual=None):
        if (
            self._native and residual is None and input.ndim == 2
            and input.is_contiguous() and input.dtype == torch.bfloat16
            and 1 <= input.shape[0] <= 8
        ):
            if input.shape[0] == 1:
                return self._skinny(input, self.weight, self._skinny_config)
            output = torch.empty(
                (input.shape[0], self.weight.shape[0]),
                dtype=input.dtype, device=input.device,
            )
            rtp_llm_ops.kimi_k3_fused_a_gemm(
                output, input, self.weight.t(), True
            )
            return output
        return super().forward(input, residual)

class KimiK3MlaLinear(KimiK3Bf16Linear):
    """Pinned native SM103/107 MLA projection plan for M=1..16."""

    def __init__(self, weight):
        super().__init__(weight)
        self._native = (
            self.weight.is_cuda and self.weight.dtype == torch.bfloat16
            and tuple(self.weight.shape) in (
                (2112, 7168), (1536, 7168), (2304, 1536), (4608, 1536)
            )
            and torch.cuda.get_device_capability(self.weight.device) in ((10, 3), (10, 7))
        )
        if self._native:
            if not hasattr(rtp_llm_ops, "kimi_k3_fused_a_gemm"):
                raise RuntimeError("K3 native MLA requires the CUDA13 fused-A binding")
            self.weight = self.weight.contiguous()

    def forward(self, input, residual=None):
        if (
            self._native and residual is None and input.ndim == 2
            and input.is_contiguous() and input.dtype == torch.bfloat16
            and 1 <= input.shape[0] <= 16
        ):
            output = torch.empty(
                (input.shape[0], self.weight.shape[0]),
                dtype=input.dtype, device=input.device,
            )
            rtp_llm_ops.kimi_k3_fused_a_gemm(output, input, self.weight.t(), True)
            return output
        return super().forward(input, residual)

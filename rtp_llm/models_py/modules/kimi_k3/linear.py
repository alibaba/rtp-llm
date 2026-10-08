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
        return result if input.ndim == 2 else result.reshape(shape)
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


def bf16_linear_add_inplace(input: torch.Tensor, weight: torch.Tensor,
                            residual: torch.Tensor) -> torch.Tensor:
    """Consume an owned BF16 residual in an addmm epilogue."""
    shape = (*input.shape[:-1], weight.shape[0])
    if tuple(residual.shape) != shape:
        raise ValueError("K3 projection residual must match the output shape")
    if not residual.is_contiguous():
        raise ValueError("in-place projection residual must be contiguous")
    rows = prod(input.shape[:-1])
    result = residual.reshape(rows, weight.shape[0]).addmm_(
        input.reshape(rows, input.shape[-1]), weight.t()
    )
    return result.reshape(shape)


class KimiK3Bf16Linear(CudaF16Linear):
    """Explicit K3 selection; deliberately absent from the public registry."""

    def forward(self, input: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if self.bias is not None:
            raise ValueError("K3 BF16 projections require bias-free weights")
        if residual is not None:
            return bf16_linear_add(input, self.weight, residual)
        return bf16_linear(input, self.weight)

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
        try:
            from rtp_llm.models_py.modules.kimi_k3.moe_backend import _load_native
            backend = _load_native()
        except (ImportError, RuntimeError):
            return False
        return callable(getattr(backend, "bf16_gemm_nt_skip_head_mid", None))

    def forward_skip_head_mid(
        self, input: torch.Tensor, head_splits: tuple[int, int, int],
        *, output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write the per-head RoPE gap directly into a reusable BF16 buffer."""
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
        from rtp_llm.models_py.modules.kimi_k3.moe_backend import _load_native
        _load_native().bf16_gemm_nt_skip_head_mid(
            input, self.weight, output, head_splits, compiled_dims="nk"
        )
        return output

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

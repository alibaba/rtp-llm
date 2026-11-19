"""Keep GLM's CUDA RMS reductions independent of FlashInfer's default backend."""

import functools
from typing import Optional, Tuple

import torch

from rtp_llm.models_py.modules.base.common.norm import BaseNorm, BaseResNorm


@functools.cache
def _cuda_norm_module():
    # Import and build during model initialization, before CUDA Graph capture.
    # Do not change FlashInfer's process-wide backend selector.
    from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

    setup_cutlass_import_path()
    from flashinfer.jit.norm import gen_norm_module

    return gen_norm_module().build_and_load()


def _resolve_pdl(device: torch.device, enable_pdl: Optional[bool]) -> bool:
    if enable_pdl is False:
        return False
    from flashinfer.norm import device_support_pdl

    return bool(device_support_pdl(device))


def _validate_rows(x: torch.Tensor, weight: torch.Tensor) -> None:
    if (
        x.ndim != 2
        or x.shape[1] == 0
        or x.stride(1) != 1
        or x.stride(0) < x.shape[1]
        or not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float16)
    ):
        raise ValueError("GLM CUDA norm requires nonoverlapping CUDA rows")
    if (
        weight.ndim != 1
        or weight.shape[0] != x.shape[1]
        or not weight.is_contiguous()
        or weight.device != x.device
        or weight.dtype != x.dtype
    ):
        raise ValueError(
            "GLM CUDA norm weight must match the input columns, dtype and device"
        )


def _span(x: torch.Tensor) -> Tuple[int, int]:
    start = x.data_ptr()
    elements = (
        0
        if x.numel() == 0
        else 1 + sum((size - 1) * stride for size, stride in zip(x.shape, x.stride()))
    )
    return start, start + elements * x.element_size()


def _overlaps(x: torch.Tensor, y: torch.Tensor) -> bool:
    x_begin, x_end = _span(x)
    y_begin, y_end = _span(y)
    return max(x_begin, y_begin) < min(x_end, y_end)


def _rows_per_launch(hidden_size: int, *row_strides: int) -> int:
    # FlashInfer CUDA's row/stride arithmetic is uint32. Slicing changes the
    # base pointer, so each launch's last addressed element must fit uint32.
    limit = 1 << 32
    if not 0 < hidden_size < limit or any(
        stride < hidden_size or stride >= limit for stride in row_strides
    ):
        raise ValueError("GLM CUDA norm row geometry exceeds uint32 addressing")
    return min((limit - hidden_size) // stride + 1 for stride in row_strides)


class Glm53RMSNorm(BaseNorm):
    def __init__(
        self,
        weight: torch.Tensor,
        eps: float = 1e-6,
        *,
        enable_pdl: Optional[bool] = None
    ):
        super().__init__(weight, eps)
        self._enable_pdl = enable_pdl
        if weight.is_cuda:
            _cuda_norm_module()

    def forward(
        self, hidden_states: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _validate_rows(hidden_states, self.weight)
        if output is None:
            output = torch.empty_like(hidden_states)
        _validate_rows(output, self.weight)
        if output.shape != hidden_states.shape:
            raise ValueError("GLM CUDA norm output shape must match the input")
        exact_alias = (
            output.data_ptr() == hidden_states.data_ptr()
            and output.stride() == hidden_states.stride()
        )
        if _overlaps(output, self.weight) or (
            _overlaps(hidden_states, output) and not exact_alias
        ):
            raise ValueError(
                "GLM CUDA norm only supports disjoint output or exact out=input"
            )
        rows = _rows_per_launch(
            hidden_states.shape[1], hidden_states.stride(0), output.stride(0)
        )
        if hidden_states.shape[0] == 0:
            return output
        module = _cuda_norm_module()
        pdl = _resolve_pdl(hidden_states.device, self._enable_pdl)
        for begin in range(0, hidden_states.shape[0], rows):
            module.rmsnorm(
                output[begin : begin + rows],
                hidden_states[begin : begin + rows],
                self.weight.data,
                self.variance_epsilon,
                pdl,
            )
        return output


class Glm53RMSResNorm(BaseResNorm):
    def __init__(
        self,
        weight: torch.Tensor,
        eps: float = 1e-6,
        *,
        enable_pdl: Optional[bool] = None
    ):
        super().__init__(weight, eps)
        self._enable_pdl = enable_pdl
        if weight.is_cuda:
            _cuda_norm_module()

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_rows(hidden_states, self.weight)
        _validate_rows(residual, self.weight)
        if hidden_states.shape != residual.shape:
            raise ValueError("GLM CUDA fused norm input and residual shapes must match")
        if (
            _overlaps(hidden_states, residual)
            or _overlaps(hidden_states, self.weight)
            or _overlaps(residual, self.weight)
        ):
            raise ValueError(
                "GLM CUDA fused norm requires disjoint writable inputs and weight"
            )
        rows = _rows_per_launch(
            hidden_states.shape[1], hidden_states.stride(0), residual.stride(0)
        )
        if hidden_states.shape[0] == 0:
            return hidden_states, residual
        module = _cuda_norm_module()
        pdl = _resolve_pdl(hidden_states.device, self._enable_pdl)
        for begin in range(0, hidden_states.shape[0], rows):
            module.fused_add_rmsnorm(
                hidden_states[begin : begin + rows],
                residual[begin : begin + rows],
                self.weight.data,
                self.variance_epsilon,
                pdl,
            )
        return hidden_states, residual

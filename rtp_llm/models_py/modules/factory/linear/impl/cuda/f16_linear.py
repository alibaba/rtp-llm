"""CUDA F16 (non-quantized) Linear implementation"""

import functools
from typing import Optional, Tuple

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig


@functools.cache
def _has_bf16_gemm_nt_skip_head_mid() -> bool:
    """Resolve the optional CUDA capability without affecting CPU imports."""

    try:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            has_bf16_gemm_nt_skip_head_mid,
        )
    except (ImportError, RuntimeError):
        return False
    return has_bf16_gemm_nt_skip_head_mid()


class CudaF16Linear(LinearBase):
    """CUDA F16 (non-quantized) Linear"""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
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
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight,
            weight_scales,
            input_scales,
            bias,
            quant_config,
            weight_scale_2,
        )
        self.weight = weight.T
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def _valid_skip_head_mid_splits(
        self,
        head_splits: Tuple[int, int, int],
    ) -> bool:
        if len(head_splits) != 3:
            return False
        left, _, right = head_splits
        return (
            min(head_splits) > 0
            and all(split % 64 == 0 for split in head_splits)
            and self.weight.shape[0] % (left + right) == 0
        )

    def _validate_skip_head_mid_input(
        self, input: torch.Tensor, head_splits: Tuple[int, int, int]
    ) -> None:
        if not self._valid_skip_head_mid_splits(head_splits):
            raise ValueError(
                f"invalid head_splits={head_splits} for output_features="
                f"{self.weight.shape[0]}"
            )
        if (
            self.bias is not None
            or input.ndim != 2
            or self.weight.ndim != 2
            or input.dtype != torch.bfloat16
            or self.weight.dtype != torch.bfloat16
            or not input.is_cuda
            or not self.weight.is_cuda
            or input.device != self.weight.device
            or input.shape[1] != self.weight.shape[1]
            or not input.is_contiguous()
            or not self.weight.T.is_contiguous()
        ):
            raise RuntimeError(
                "bf16_gemm_nt_skip_head_mid requires bias-free, two-dimensional "
                "CUDA BF16 tensors in canonical layouts: row-major contiguous input "
                "and transpose-contiguous weight"
            )
        if torch.cuda.get_device_capability(input.device)[0] != 10:
            raise RuntimeError("bf16_gemm_nt_skip_head_mid requires SM100 or SM103")
        if input.data_ptr() % 16 or self.weight.data_ptr() % 16:
            raise RuntimeError(
                "bf16_gemm_nt_skip_head_mid requires 16-byte-aligned input and weight"
            )
        if not _has_bf16_gemm_nt_skip_head_mid():
            raise RuntimeError(
                "the installed DeepGEMM package does not provide "
                "bf16_gemm_nt_skip_head_mid"
            )

    @staticmethod
    def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
        if left.numel() == 0 or right.numel() == 0:
            return False
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()

    def supports_skip_head_mid(
        self,
        input: torch.Tensor,
        head_splits: Tuple[int, int, int],
    ) -> bool:
        try:
            self._validate_skip_head_mid_input(input, head_splits)
        except (RuntimeError, ValueError):
            return False
        return True

    def forward_skip_head_mid(
        self,
        input: torch.Tensor,
        head_splits: Tuple[int, int, int],
        *,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one BF16 GEMM into ``[left | mid gap | right]`` per head.

        ``output`` may be supplied by a serialized chunk loop to reuse its
        storage.  Omitting it preserves the regular allocating interface.
        """
        self._validate_skip_head_mid_input(input, head_splits)
        expected_shape = self._skip_head_mid_output_shape(input, head_splits)
        if output is None:
            output = input.new_empty(expected_shape)
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != input.dtype
            or output.device != input.device
            or not output.is_contiguous()
        ):
            raise RuntimeError(
                "bf16_gemm_nt_skip_head_mid output buffer mismatch: "
                f"expected shape={expected_shape}, dtype={input.dtype}, "
                f"device={input.device}, contiguous=True; got "
                f"shape={tuple(output.shape)}, dtype={output.dtype}, "
                f"device={output.device}, contiguous={output.is_contiguous()}"
            )
        if output.data_ptr() % 16:
            raise RuntimeError(
                "bf16_gemm_nt_skip_head_mid output must be 16-byte aligned"
            )
        if self._shares_storage(output, input) or self._shares_storage(
            output, self.weight
        ):
            raise RuntimeError(
                "bf16_gemm_nt_skip_head_mid output must use storage independent "
                "from input and weight"
            )
        return self._run_skip_head_mid(input, output, head_splits)

    def forward_skip_head_mid_out(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        head_splits: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Run skip-head-mid BF16 GEMM into a caller-provided output."""
        return self.forward_skip_head_mid(input, head_splits, output=output)

    def _skip_head_mid_output_shape(
        self,
        input: torch.Tensor,
        head_splits: Tuple[int, int, int],
    ) -> tuple[int, int]:
        left, mid, right = head_splits
        num_heads = self.weight.shape[0] // (left + right)
        return input.shape[0], num_heads * (left + mid + right)

    def _run_skip_head_mid(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        head_splits: Tuple[int, int, int],
    ) -> torch.Tensor:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            bf16_gemm_nt_skip_head_mid,
        )

        bf16_gemm_nt_skip_head_mid(input, self.weight, output, head_splits)
        return output

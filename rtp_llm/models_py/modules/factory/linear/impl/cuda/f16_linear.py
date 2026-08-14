"""CUDA F16 (non-quantized) Linear implementation"""

from typing import Optional, Tuple

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig


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
            or input.stride(1) != 1
            or 1 not in self.weight.stride()
        ):
            raise RuntimeError(
                "bf16_gemm_nt_skip_head_mid requires bias-free, two-dimensional "
                "CUDA BF16 input and weight tensors with a contiguous inner dimension"
            )
        if torch.cuda.get_device_capability(input.device)[0] != 10:
            raise RuntimeError("bf16_gemm_nt_skip_head_mid requires SM100 or SM103")

    def forward_skip_head_mid(
        self,
        input: torch.Tensor,
        head_splits: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Run one BF16 GEMM into ``[left | mid gap | right]`` per head."""
        self._validate_skip_head_mid_input(input, head_splits)
        output_features = self.weight.shape[0]
        left, mid, right = head_splits
        logical_head_dim = left + right
        num_heads = output_features // logical_head_dim
        physical_features = num_heads * (left + mid + right)
        output = input.new_empty((input.shape[0], physical_features))

        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            bf16_gemm_nt_skip_head_mid,
        )

        bf16_gemm_nt_skip_head_mid(input, self.weight, output, head_splits)
        return output

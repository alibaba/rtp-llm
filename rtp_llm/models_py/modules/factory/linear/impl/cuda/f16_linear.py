"""CUDA F16 (non-quantized) Linear implementation"""

import os
from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.models_py.modules.factory.linear.impl.cuda.router_linear import (
    maybe_bf16_gdn_linear,
    maybe_bf16_router_linear,
)
from rtp_llm.models_py.triton_kernels.common.bf16_gate_linear import (
    maybe_bf16_gate_linear,
)
from rtp_llm.models_py.triton_kernels.common.scalar_linear import (
    maybe_bf16_scalar_linear,
    maybe_fp32_scalar_linear,
)
from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import is_decode_phase
from rtp_llm.ops import HWKernelConfig


class CudaF16Linear(LinearBase):
    """CUDA F16 (non-quantized) Linear"""

    _shared_scalar_weight_fp32: Optional[torch.Tensor] = None

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
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self.weight = weight.T
        self.bias = bias
        # One option controls all three BF16 gate repairs; snapshot before capture.
        self.batch_invariant_gates = os.environ.get("RTP_BF16_GATE_KERNEL", "0") == "1"

    def enable_fp32_shared_scalar_decode(self) -> bool:
        """Opt in only this loaded, bias-free CUDA BF16 shared scalar gate."""
        if not (
            self.weight.is_cuda
            and self.weight.dtype == torch.bfloat16
            and self.weight.shape == (1, 4096)
            and self.bias is None
        ):
            return False
        self._shared_scalar_weight_fp32 = self.weight.detach().float().contiguous()
        self._shared_scalar_weight_source = self.weight
        return True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (
            self._shared_scalar_weight_fp32 is not None
            and self.weight is self._shared_scalar_weight_source
            and is_decode_phase()
        ):
            output = maybe_fp32_scalar_linear(input, self._shared_scalar_weight_fp32)
            if output is not None:
                return output
        if not self.batch_invariant_gates:
            return F.linear(input, self.weight, self.bias)

        # Decode covers every M. Keep the validated small-prefill limit fixed.
        decode = is_decode_phase()
        if input.ndim == 2 and (input.shape[0] <= 256 or decode):
            output = maybe_bf16_scalar_linear(input, self.weight, self.bias)
            if output is not None:
                return output
        if decode:
            output = maybe_bf16_gate_linear(input, self.weight, self.bias)
            if output is not None:
                return output
        output = maybe_bf16_router_linear(input, self.weight, self.bias)
        if output is not None:
            return output
        output = maybe_bf16_gdn_linear(input, self.weight, self.bias)
        if output is not None:
            return output
        return F.linear(input, self.weight, self.bias)

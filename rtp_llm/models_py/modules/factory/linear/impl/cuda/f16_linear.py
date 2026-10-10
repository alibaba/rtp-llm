"""CUDA F16 (non-quantized) Linear implementation.

When USE_ONLINE_FP4GEMM=1, layers with K%128==0 are deferred to
CudaOnlineMxfp4Linear for online MXFP4 quantization + mm_fp4 GEMM.
RTP_BF16_LINEAR_BACKEND=deepgemm opts supported SM100 BF16 projections into
DeepGEMM, whose reduction order is independent of the number of token rows.
"""

import os
from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig

_MXFP4_ONLINE = os.environ.get("USE_ONLINE_FP4GEMM", "0") == "1"


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
        if weight_scales is not None:
            return False
        if _MXFP4_ONLINE and weight.dtype in (torch.bfloat16, torch.float16):
            if weight.dim() == 2 and weight.shape[0] % 128 == 0:
                return False
        return True

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
        backend = os.environ.get("RTP_BF16_LINEAR_BACKEND", "torch")
        if backend not in ("torch", "deepgemm"):
            raise ValueError(f"Unsupported RTP_BF16_LINEAR_BACKEND: {backend}")
        self._bf16_gemm = None
        self._bf16_weight = None
        if (
            backend == "deepgemm"
            and self.weight.is_cuda
            and self.weight.dtype == torch.bfloat16
            and self.bias is None
            and self.weight.shape[0] % 8 == 0
            and self.weight.shape[1] % 128 == 0
            and torch.cuda.get_device_capability(self.weight.device)[0] == 10
        ):
            import deep_gemm

            self._bf16_gemm = getattr(deep_gemm, "bf16_gemm_nt", None)
            if self._bf16_gemm is not None:
                # Keep the original weight view for callers that inspect its
                # layout to decide whether collective GEMM fusion is supported.
                self._bf16_weight = self.weight.contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (
            self._bf16_gemm is not None
            and input.dtype == torch.bfloat16
            and input.device == self.weight.device
        ):
            flat = input.reshape(-1, input.shape[-1]).contiguous()
            output = torch.empty(
                (flat.shape[0], self.weight.shape[0]),
                dtype=input.dtype,
                device=input.device,
            )
            if flat.shape[0]:
                self._bf16_gemm(flat, self._bf16_weight, output)
            return output.view(*input.shape[:-1], self.weight.shape[0])
        return F.linear(input, self.weight, self.bias)

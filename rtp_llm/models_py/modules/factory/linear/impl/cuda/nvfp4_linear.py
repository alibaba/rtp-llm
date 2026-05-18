"""Online MXFP4 linear layer.

Converts BF16/FP16 weights to MXFP4 (block_size=32, UE8M0 scales) at init
time, then uses flashinfer mm_fp4 for inference.
Activated by env USE_ONLINE_FP4GEMM=1.

Constraint: K must be divisible by 128 (required by swizzled scale factor layout).
N has no alignment requirement.
"""

import os
from typing import Optional

import torch
from flashinfer import autotune, mxfp4_quantize, mm_fp4

from rtp_llm.models_py.modules.factory.linear import LinearBase

_ENABLED = os.environ.get("USE_ONLINE_FP4GEMM", "0") == "1"


class CudaOnlineMxfp4Linear(LinearBase):
    """Online MXFP4 quantized linear using flashinfer mm_fp4 (cute-dsl backend)."""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config=None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if not _ENABLED or weight_scales is not None:
            return False
        if weight.dtype not in (torch.bfloat16, torch.float16):
            return False
        if weight.dim() != 2:
            return False
        K, N = weight.shape
        return K % 128 == 0

    @torch.inference_mode()
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(weight, weight_scales, input_scales, bias, quant_config, weight_scale_2)

        K, N = weight.shape
        w = weight.T.to(torch.bfloat16).contiguous()  # [N, K]

        w_fp4, w_sf = mxfp4_quantize(w, backend="cute-dsl")
        del w, weight

        self.register_buffer("w_fp4_t", w_fp4.T)
        self.register_buffer("w_sf_t", w_sf.T)
        self.N = N
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        shape = input.shape
        x = input.reshape(-1, shape[-1]) if input.dim() > 2 else input

        a_fp4, a_sf = mxfp4_quantize(x, backend="cute-dsl")

        with autotune():
            out = mm_fp4(
                a_fp4, self.w_fp4_t,
                a_sf, self.w_sf_t,
                None, torch.bfloat16,
                block_size=32, use_nvfp4=False, backend="cute-dsl",
            )

        if self.bias is not None:
            out = out + self.bias

        if input.dim() > 2:
            out = out.reshape(*shape[:-1], self.N)

        return out


# Backward-compatible alias
CudaOnlineNvfp4Linear = CudaOnlineMxfp4Linear

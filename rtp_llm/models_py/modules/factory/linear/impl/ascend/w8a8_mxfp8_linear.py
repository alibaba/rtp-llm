"""Ascend NPU W8A8_MXFP8 Linear (flow aligned with vllm-ascend w8a8_mxfp8.py).

Flow: ``npu_dynamic_mx_quant`` (activation, 1x32 groups along K) → ``npu_quant_matmul``.

Weight layouts (after ``PerBlockFp8Weight._postprocess``, NPU branch):
  * ``weight``: fp8_e4m3fn ``[K, N]``
  * ``weight_scales``: uint8 (E8M0) ``[kp // 2, N, 2]`` swizzled pair-split,
    ``kp = K // 32``.
"""

from typing import Optional

import torch
import torch_npu

from rtp_llm.models_py.kernels.ascend.w8a8_mx_layout import (
    MXFP8_GROUP_SIZE,
    get_e8m0_dtype,
)
from rtp_llm.models_py.modules.factory.linear import LinearBase

# Logical dtype of E8M0 scales (prefer torch_npu, fall back to torch).
FLOAT8_E8M0FNU_DTYPE = get_e8m0_dtype()


class AscendW8A8MXFP8Linear(LinearBase):
    """NPU W8A8_MXFP8 Linear layer (activation is dynamically MX-quantized)."""

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
        if weight_scales is None or quant_config is None:
            return False
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False
        # Static (ModelSlim) path only in this stage; the dynamic load-quant
        # path (FP8_PER_BLOCK) is enabled by a follow-up TODO once the weight
        # loader produces MXFP8 layouts for it as well.
        return quant_config.get_method() == "ASCEND_W8A8_MXFP8"

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
        assert weight.dim() == 2, (
            f"AscendW8A8MXFP8Linear expects 2D weight [K, N], got {tuple(weight.shape)}"
        )
        self.weight = weight  # fp8_e4m3fn [K, N]
        self.weight_scale = weight_scales  # uint8 (E8M0) [kp//2, N, 2]
        self.bias = bias

    # y = x * W + b
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 2, (
            f"AscendW8A8MXFP8Linear expects 2D input [M, K], got {tuple(input.shape)}"
        )
        if input.dtype in (torch.bfloat16, torch.float16):
            # W8A8: activation is dynamically MX-quantized online (not weight-only)
            x_fp8, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                input, dst_type=torch.float8_e4m3fn
            )
            output_dtype = input.dtype
        elif input.dtype == torch.float8_e4m3fn:
            x_fp8 = input
            # Unity scale: E8M0 0x7F == 2^0 == 1.0 (torch.ones would be
            # 0x01 == 2^-126). npu_dynamic_mx_quant on torch_npu 2.9.0.post3
            # returns [M, kp//2, 2] (pair-split 3D), so build the same layout.
            pertoken_scale = torch.full(
                (input.shape[0], input.shape[1] // MXFP8_GROUP_SIZE // 2, 2),
                0x7F,
                dtype=torch.uint8,
                device=input.device,
            )
            output_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported input dtype: {input.dtype}")

        bias = self.bias.to(torch.float32) if self.bias is not None else None
        # pertoken_scale stays 2D [M, K//32] (no swizzle) — same as vllm-ascend.
        return torch_npu.npu_quant_matmul(
            x_fp8,
            self.weight,
            self.weight_scale,
            scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            bias=bias,
            output_dtype=output_dtype,
            group_sizes=[1, 1, MXFP8_GROUP_SIZE],
        )

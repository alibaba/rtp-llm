"""CUDA MXFP8 (1x32 microscaling FP8) Linear implementation.

Weight is e4m3 ``[N, K]`` with a prepacked int32 UE8M0 ``[1, 32]`` block scale
(produced by the MXFP8 weight loader). Activations are dynamically quantized to
e4m3 with a per-(row, 32-col) UE8M0 scale, then the GEMM goes through DeepGEMM's
``fp8_fp4_gemm_nt`` with ``recipe=(1, 32)``. SM100 only.
"""

from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_linear
from rtp_llm.ops import HWKernelConfig


class CudaMxfp8Linear(LinearBase):
    # Read by fused norm+quant gates in generic_moe / mla_attention; MXFP8
    # always emits UE8M0-packed scales.
    scale_ue8m0: bool = True

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
        if quant_config is None or not hasattr(quant_config, "get_method"):
            return False
        if quant_config.get_method() != "MXFP8":
            return False
        if weight_scales is None:
            return False
        return weight.dtype == torch.float8_e4m3fn

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
        self.weight = weight
        self.weight_scale = weight_scales
        self.bias = bias
        self.N = weight.shape[0]
        self._weight_scale_packed = None

    def _packed_weight_scale(self) -> torch.Tensor:
        if self._weight_scale_packed is None:
            from rtp_llm.models_py.kernels.cuda.mxfp8_ops import pack_mxfp8_scale
            n, k = self.weight.shape
            self._weight_scale_packed = pack_mxfp8_scale(self.weight_scale, mn=n, k=k)
        return self._weight_scale_packed

    def forward(
        self,
        input: torch.Tensor,
        input_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = input.shape
        x = input.reshape(-1, orig_shape[-1])
        if input_scales is not None:
            from rtp_llm.models_py.kernels.cuda.mxfp8_ops import MX_BLOCK
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_fp4_gemm_nt
            M = x.shape[0]
            out = torch.empty(M, self.N, device=x.device, dtype=torch.bfloat16)
            with torch.cuda.device(x.device):
                fp8_fp4_gemm_nt(
                    (x, input_scales),
                    (self.weight, self._packed_weight_scale()),
                    out,
                    recipe_a=(1, MX_BLOCK),
                    recipe_b=(1, MX_BLOCK),
                    disable_ue8m0_cast=True,
                )
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
        else:
            out = mxfp8_linear(
                x, self.weight, self._packed_weight_scale(), self.bias,
                out_dtype=torch.bfloat16,
            )
        return out.reshape(*orig_shape[:-1], out.shape[-1])

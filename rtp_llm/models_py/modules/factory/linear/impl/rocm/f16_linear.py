"""ROCm F16 (non-quantized) Linear implementation"""

import os
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase
from aiter import hipb_mm, hipb_create_extension
from functools import lru_cache
from rtp_llm.ops import HWKernelConfig
from rtp_llm.ops.compute_ops import rtp_llm_ops


class RocmF16LinearBase(LinearBase):
    """ROCm F16 (non-quantized) Linear"""

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
        raise NotImplementedError("Subclasses must implement `can_handle`.")

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
        self.bias = bias
        self._cached_gemm = (
            getattr(rtp_llm_ops, "rocm_cached_f16_gemm", None)
            if os.environ.get("RTP_ROCM_F16_GEMM_PLAN_CACHE", "1") == "1"
            and os.environ.get("HIP_ONLINE_TUNING", "") not in ("1", "true")
            else None
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def init_hipblas():
        hipb_create_extension()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement `forward`.")


class RocmF16LinearWithSwizzle(RocmF16LinearBase):

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        return (
            weight_scales is None
            and hw_kernel_config is not None
            and hw_kernel_config.use_swizzleA
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Initialize the legacy graph/fallback resources during eager warmup,
        # even when all eager calls hit the native cache. The initializer is cached.
        self.init_hipblas()
        if self._cached_gemm is not None:
            output = self._cached_gemm(input, self.weight, self.bias, True)
            if output is not None:
                return output
        return hipb_mm(
            input,
            self.weight,
            solution_index=-1,
            bias=self.bias,
            out_dtype=input.dtype,
            scaleA=None,
            scaleB=None,
            scaleOut=None,
            bpreshuffle=True,
        )


class RocmF16LinearNoSwizzle(RocmF16LinearBase):

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if weight_scales is not None:
            return False
        if hw_kernel_config is None:
            return True
        elif not hw_kernel_config.use_swizzleA:
            return True
        else:
            return False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.init_hipblas()
        if self._cached_gemm is not None:
            output = self._cached_gemm(input, self.weight, self.bias, False)
            if output is not None:
                return output
        return hipb_mm(
            input,
            self.weight,
            solution_index=-1,
            bias=self.bias,
            out_dtype=input.dtype,
            scaleA=None,
            scaleB=None,
            scaleOut=None,
            bpreshuffle=False,
        )

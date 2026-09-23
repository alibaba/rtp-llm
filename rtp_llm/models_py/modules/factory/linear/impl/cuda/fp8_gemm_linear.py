"""CUDA FP8 GEMM wrapper that dispatches between flashinfer and DeepGEMM."""

import logging
import os
from functools import cache
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_flashinfer_linear import (
    CudaFp8FlashinferLinear,
)
from rtp_llm.ops import HWKernelConfig

logger = logging.getLogger(__name__)

_ENABLE_FP8_FLASHINFER_GEMM_ENV = "RTP_LLM_ENABLE_FP8_FLASHINFER_GEMM"
_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))


def _fp8_flashinfer_gemm_enabled() -> bool:
    """Return whether the small-M FlashInfer FP8 GEMM optimization is enabled."""
    raw_value = os.environ.get(_ENABLE_FP8_FLASHINFER_GEMM_ENV)
    if raw_value is None:
        return True
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{_ENABLE_FP8_FLASHINFER_GEMM_ENV} must be a boolean "
        f"(1/0, true/false, yes/no, on/off), got {raw_value!r}"
    )


@cache
def _log_fp8_flashinfer_gemm_switch(enabled: bool) -> None:
    logger.info(
        "Small-M FlashInfer FP8 GEMM is %s (%s=%d); " "disabled calls use DeepGEMM",
        "enabled" if enabled else "disabled",
        _ENABLE_FP8_FLASHINFER_GEMM_ENV,
        int(enabled),
    )


class CudaFp8GEMMLinear(LinearBase):
    """CUDA FP8 GEMM wrapper."""

    FLASHINFER_M_THRESHOLD = CudaFp8FlashinferLinear.FLASHINFER_M_THRESHOLD

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
        if weight_scales is None or quant_config is None:
            return False
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False
        return quant_config.get_method() == "FP8_PER_BLOCK"

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
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self._deepgemm_linear = CudaFp8DeepGEMMLinear(
            weight=weight,
            weight_scales=weight_scales,
            input_scales=input_scales,
            bias=bias,
            quant_config=quant_config,
            weight_scale_2=weight_scale_2,
        )
        self._enable_flashinfer_gemm = _fp8_flashinfer_gemm_enabled()
        _log_fp8_flashinfer_gemm_switch(self._enable_flashinfer_gemm)
        self._flashinfer_linear = (
            self._create_flashinfer_backend(
                weight=weight,
                weight_scales=weight_scales,
                input_scales=input_scales,
                bias=bias,
                quant_config=quant_config,
                weight_scale_2=weight_scale_2,
            )
            if self._enable_flashinfer_gemm
            else None
        )

        self.weight = self._deepgemm_linear.weight
        self.weight_scales = self._deepgemm_linear.weight_scales
        self.input_scales = input_scales
        self.bias = self._deepgemm_linear.bias
        self.K = self._deepgemm_linear.K
        self.N = self._deepgemm_linear.N
        self.scale_ue8m0 = getattr(self._deepgemm_linear, "scale_ue8m0", False)
        self.cached_scales = getattr(self._deepgemm_linear, "cached_scales", None)
        self.cached_scales_max_len = getattr(
            self._deepgemm_linear, "cached_scales_max_len", 0
        )

    def _create_flashinfer_backend(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        input_scales: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        quant_config: object,
        weight_scale_2: Optional[torch.Tensor],
    ) -> Optional[CudaFp8FlashinferLinear]:
        if not CudaFp8FlashinferLinear.can_handle(
            quant_config,
            weight,
            weight_scales,
            None,
            weight_scale_2,
            input_scales,
        ):
            return None
        return CudaFp8FlashinferLinear(
            weight=weight,
            weight_scales=weight_scales,
            input_scales=input_scales,
            bias=bias,
            quant_config=quant_config,
            weight_scale_2=weight_scale_2,
        )

    def maybe_cache_quant_scale(self, max_len: int) -> None:
        self._deepgemm_linear.maybe_cache_quant_scale(max_len)
        self.cached_scales = getattr(self._deepgemm_linear, "cached_scales", None)
        self.cached_scales_max_len = getattr(
            self._deepgemm_linear, "cached_scales_max_len", 0
        )

    def _should_use_flashinfer(self, input: torch.Tensor) -> bool:
        if not self._enable_flashinfer_gemm:
            return False
        if self._flashinfer_linear is None:
            return False
        if input.dim() != 2:
            return False

        m, k = input.shape
        if m >= self.FLASHINFER_M_THRESHOLD or k != self.K:
            return False
        if input.dtype == torch.bfloat16:
            return True
        if (
            input.dtype == torch.float8_e4m3fn
            and self.input_scales is not None
            and self.input_scales.dtype == torch.float32
        ):
            return True
        return False

    def forward(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # DeepGEMM supports caller-provided output buffers; the small-M
        # FlashInfer path currently does not.
        if out is not None:
            return self._deepgemm_linear(input, out=out)
        if not self._should_use_flashinfer(input):
            return self._deepgemm_linear(input)
        return self._flashinfer_linear(input)

"""CUDA FP8 GEMM wrapper that dispatches between flashinfer and DeepGEMM."""

from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_flashinfer_linear import (
    CudaFp8FlashinferLinear,
)
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.ops import HWKernelConfig


class CudaFp8GEMMLinear(LinearBase):
    """CUDA FP8 GEMM wrapper."""

    FLASHINFER_M_THRESHOLD = CudaFp8FlashinferLinear.FLASHINFER_M_THRESHOLD

    @classmethod
    def _is_fp8_per_block_candidate(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
    ) -> bool:
        return (
            weight_scales is not None
            and quant_config is not None
            and weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            and quant_config.get_method() == "FP8_PER_BLOCK"
        )

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
        if not cls._is_fp8_per_block_candidate(quant_config, weight, weight_scales):
            return False
        # Preserve the sm90/sm100 behavior where construction reports the
        # actionable DeepGEMM installation error. Only sm12x is excluded from
        # this strategy because it uses the SM120 CUTLASS backend instead.
        if not is_sm12x(weight.device):
            return True

        return False

    @classmethod
    def rejection_reason(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> Optional[str]:
        if not is_sm12x(weight.device) or not cls._is_fp8_per_block_candidate(
            quant_config, weight, weight_scales
        ):
            return None
        # The SM120 strategy can be absent from the registry when its binding
        # was not compiled. Preserve its public diagnostic in that case.
        try:
            from .fp8_vllm_blockwise_sm120_linear import CudaFp8VllmBlockwiseLinear
        except ImportError as error:
            return (
                "SM120 FP8_PER_BLOCK backend is unavailable; rebuild on x86 "
                f"with --config=cuda12_9 (ENABLE_FP8_SM120): {error}"
            )

        return CudaFp8VllmBlockwiseLinear.rejection_reason(
            quant_config, weight, weight_scales
        )

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
        self._flashinfer_linear = self._create_flashinfer_backend(
            weight=weight,
            weight_scales=weight_scales,
            input_scales=input_scales,
            bias=bias,
            quant_config=quant_config,
            weight_scale_2=weight_scale_2,
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._should_use_flashinfer(input):
            return self._deepgemm_linear(input)
        return self._flashinfer_linear(input)

"""CUDA FP8 flashinfer Linear implementation for SM90."""

from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.ops import HWKernelConfig


def is_sm90() -> bool:
    try:
        major, _ = get_sm()
    except Exception:
        return False
    return major == 9


def _has_flashinfer_sm90_fp8_gemm() -> bool:
    return _get_flashinfer_sm90_fp8_gemm() is not None


def _get_flashinfer_sm90_fp8_gemm():
    try:
        from flashinfer.gemm import fp8_blockscale_gemm_sm90

        return fp8_blockscale_gemm_sm90
    except Exception:
        return None


class CudaFp8FlashinferLinear(LinearBase):
    """CUDA FP8 block-scaled Linear using flashinfer's SM90 kernel."""

    FLASHINFER_M_THRESHOLD = 32

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
        if weight.dtype not in (torch.float8_e4m3fn,):
            return False
        if quant_config.get_method() != "FP8_PER_BLOCK":
            return False
        if not is_sm90():
            return False
        return _has_flashinfer_sm90_fp8_gemm()

    @torch.inference_mode()
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        activation_type: Optional[str] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2,
            activation_type,
        )
        self._fi_gemm = _get_flashinfer_sm90_fp8_gemm()
        if self._fi_gemm is None:
            raise RuntimeError(
                "flashinfer.gemm.fp8_blockscale_gemm_sm90 is unavailable. "
                "Install flashinfer with the SM90 FP8 block-scale GEMM module."
            )

        self.weight = weight
        self.weight_scales = weight_scales
        self.input_scales = input_scales
        self.bias = bias
        if self.weight.dim() != 2 or self.weight_scales.dim() != 2:
            raise ValueError(
                "Weight and weight scale must be 2D tensors, but got "
                f"weight dim: {self.weight.dim()} and weight scale dim: {self.weight_scales.dim()}"
            )

        self.K, self.N = self.weight.shape
        self.scale_K, self.scale_N = self.weight_scales.shape
        self.weight = self.weight.reshape(self.N, self.K)
        self.weight_scales = self.weight_scales.reshape(self.scale_N, self.scale_K)

        if (self.N + 127) // 128 != self.scale_N or (
            self.K + 127
        ) // 128 != self.scale_K:
            raise ValueError(
                "Weight scale dimension mismatch! "
                f"N: {self.N}, scale_N: {self.scale_N}, "
                f"K: {self.K}, scale_K: {self.scale_K}"
            )
        if self.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"Weight dtype must be float8_e4m3fn, got {self.weight.dtype}"
            )
        if self.weight_scales.dtype != torch.float32:
            raise ValueError(
                f"Weight scale dtype must be float32, got {self.weight_scales.dtype}"
            )

        if self.bias is not None:
            if self.bias.dim() not in (1, 2):
                raise ValueError(
                    f"Bias dimension must be 1 or 2, got {self.bias.dim()}"
                )
            if self.bias.shape[-1] != self.N:
                raise ValueError(
                    f"Bias last dimension must be {self.N}, got {self.bias.shape[-1]}"
                )
            if self.bias.dim() == 2 and self.bias.shape[0] != 1:
                raise ValueError(
                    f"Bias first dimension must be 1, got {self.bias.shape[0]}"
                )
            if self.bias.dtype != torch.bfloat16:
                raise ValueError(f"Bias dtype must be bfloat16, got {self.bias.dtype}")

        if not self.weight_scales.is_contiguous():
            self.weight_scales = self.weight_scales.contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        M = input.shape[0]
        if input.dtype == torch.bfloat16:
            input_fp8, input_scales = sgl_per_token_group_quant_fp8(
                input,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
        else:
            input_fp8 = input
            input_scales = self.input_scales

        output = torch.empty(M, self.N, dtype=torch.bfloat16, device=input.device)
        self._fi_gemm(
            input_fp8,
            self.weight,
            input_scales,
            self.weight_scales,
            out=output,
        )
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

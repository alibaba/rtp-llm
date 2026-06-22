"""CUDA FP8 PER_BLOCK GEMM for sm_120 family (consumer Blackwell).

Backend wraps the vLLM-ported `cutlass_scaled_mm_blockwise_sm120_fp8`
kernel (see `models_py/bindings/cuda/cutlass/cutlass_kernels/fp8_blockwise_sm120/`).
Selected by LinearFactory only when `is_sm12x()` is true; sm_9x / sm_10x
keep using DeepGEMM via `CudaFp8GEMMLinear`.
"""

from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.models_py.utils.arch import is_cuda, is_sm12x
from rtp_llm.ops import HWKernelConfig


def _get_cutlass_scaled_mm_blockwise_sm120_fp8():
    if not (is_cuda() and is_sm12x()):
        return None
    try:
        from rtp_llm.ops.compute_ops import cutlass_scaled_mm_blockwise_sm120_fp8

        return cutlass_scaled_mm_blockwise_sm120_fp8
    except ImportError:
        return None


def _has_cutlass_scaled_mm_blockwise_sm120_fp8() -> bool:
    return _get_cutlass_scaled_mm_blockwise_sm120_fp8() is not None


cutlass_scaled_mm_blockwise_sm120_fp8 = _get_cutlass_scaled_mm_blockwise_sm120_fp8()


class CudaFp8VllmBlockwiseLinear(LinearBase):
    """CUDA FP8 PER_BLOCK Linear for sm_120 (RTX PRO 5000 / 5090).

    Only BF16 activations, output, and bias are currently supported. K and N
    must both be multiples of the 128-element block size.

    Scale layout (matches CUTLASS Sm120BlockwiseScaleConfig<1, 128, 128, MN, K>):
      - input_scales : (M, K//128), MN-major (M-stride=1, K-group-stride=M)
      - weight_scales: (N//128, K//128), K-major  (K-stride = 1)
    Input scales use column_major_scales=True, scale_tma_aligned=False
    because CUTLASS tile_atom_to_shape_SFA computes K-group stride as exactly
    M (no alignment padding).  scale_tma_aligned=True would pad to ceil4(M),
    causing a stride mismatch for non-multiple-of-4 M values.
    """

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
        if not is_sm12x():
            return False
        if not _has_cutlass_scaled_mm_blockwise_sm120_fp8():
            return False
        if weight.dtype != torch.float8_e4m3fn:
            return False
        # vLLM kernel wants float32 PER_BLOCK scales — UE8M0 (int32) is a
        # DeepGEMM-only encoding and is not supported here.
        if weight_scales.dtype != torch.float32:
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
        self._gemm_op = _get_cutlass_scaled_mm_blockwise_sm120_fp8()
        if self._gemm_op is None:
            raise RuntimeError(
                "cutlass_scaled_mm_blockwise_sm120_fp8 op is not available; "
                "this backend requires a cuda12_9_x86 build with -DENABLE_FP8_SM120."
            )

        self.weight = weight
        self.weight_scales = weight_scales
        self.input_scales = input_scales
        self.bias = bias

        if self.weight.dim() != 2 or self.weight_scales.dim() != 2:
            raise ValueError(
                f"Weight and weight scale must be 2D tensors, got weight dim "
                f"{self.weight.dim()} and weight scale dim {self.weight_scales.dim()}"
            )

        self.K, self.N = self.weight.shape
        self.scale_K, self.scale_N = self.weight_scales.shape
        if self.K % 128 != 0 or self.N % 128 != 0:
            raise ValueError(
                f"SM120 FP8 blockwise GEMM requires K and N to be multiples of "
                f"128, got K={self.K} and N={self.N}"
            )
        self.weight = self.weight.reshape(self.N, self.K).contiguous()
        self.weight_scales = self.weight_scales.reshape(
            self.scale_N, self.scale_K
        ).contiguous()

        if (self.N + 127) // 128 != self.scale_N or (
            self.K + 127
        ) // 128 != self.scale_K:
            raise ValueError(
                f"Weight scale dim mismatch: N={self.N} scale_N={self.scale_N}, "
                f"K={self.K} scale_K={self.scale_K} (expected ceil_div by 128)"
            )

        if self.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"Weight dtype must be float8_e4m3fn, got {self.weight.dtype}"
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
            self._bias_flat = self.bias.reshape(-1).contiguous()
            if self._bias_flat.device != self.weight.device:
                self._bias_flat = self._bias_flat.to(self.weight.device)
        else:
            self._bias_flat = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype != torch.bfloat16:
            raise ValueError(f"Input tensor dtype must be bfloat16, got {input.dtype}")
        if input.dim() != 2:
            raise ValueError(
                f"Input tensor dimension must be 2, got {input.dim()}D tensor"
            )
        M, K = input.shape
        if K != self.K:
            raise ValueError(
                f"Input tensor inner dimension expected to be {self.K}, got {K}"
            )

        input_fp8, input_scales = sgl_per_token_group_quant_fp8(
            input,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )

        output = torch.empty(M, self.N, dtype=torch.bfloat16, device=input.device)
        self._gemm_op(
            output,
            input_fp8,
            self.weight,
            input_scales,
            self.weight_scales,
            self._bias_flat,
        )
        return output

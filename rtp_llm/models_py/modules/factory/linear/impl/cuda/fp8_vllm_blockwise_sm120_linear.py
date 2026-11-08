"""CUDA FP8 PER_BLOCK GEMM for sm_120 family (consumer Blackwell).

Backend wraps the vLLM-ported `cutlass_scaled_mm_blockwise_sm120_fp8`
kernel (see `models_py/bindings/cuda/cutlass/cutlass_kernels/fp8_blockwise_sm120/`).
Selected by LinearFactory only on the compiled sm_120 architecture; sm_9x / sm_10x
keep using DeepGEMM via `CudaFp8GEMMLinear`.
"""

from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.kernels.cuda.sm120_fp8_blockwise import (
    sm120_blockwise_fp8_available,
    sm120_blockwise_fp8_op,
)
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.ops import HWKernelConfig


class CudaFp8VllmBlockwiseLinear(LinearBase):
    """CUDA FP8 PER_BLOCK Linear for SM120 (consumer Blackwell).

    Only BF16 activations, output, and bias are currently supported. Partial
    128-element blocks are zero-padded for the kernel. ``weight_scales`` is
    required; its Optional annotation only preserves the LinearBase/factory
    constructor signature.

    Scale layout (matches CUTLASS Sm120BlockwiseScaleConfig<1, 128, 128, MN, K>):
      - input_scales : (M, K//128), MN-major (M-stride=1, K-group-stride=M)
      - weight_scales: (N//128, K//128), K-major  (K-stride = 1)
    Input scales use column_major_scales=True, scale_tma_aligned=False
    because CUTLASS tile_atom_to_shape_SFA computes K-group stride as exactly
    M (no alignment padding).  scale_tma_aligned=True would pad to ceil4(M),
    causing a stride mismatch for non-multiple-of-4 M values.
    """

    @staticmethod
    def _restore_blockwise_weight_layout(
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        block_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        """Normalize the loader tensors to CUTLASS's physical ``(N, K)``.

        V4 checkpoints and the SM120 per-block loader expose contiguous
        weights in output-major ``(N, K)`` order and scales in
        ``(N/128, K/128)`` order.  The previous implementation interpreted
        those tensors as logical ``(K, N)`` and reshaped them, silently
        swapping the GEMM dimensions (e.g. expecting K=1024 for a K=4096
        projection).  CUTLASS consumes the tensors as-is, so only validate and
        return the canonical layout here.
        """
        if weight.device.type != "cuda" or weight_scales.device.type != "cuda":
            raise ValueError("SM120 FP8 blockwise weights must be on a CUDA device")
        if not weight.is_contiguous():
            raise ValueError("SM120 FP8 blockwise weight must be contiguous")
        if not weight_scales.is_contiguous():
            raise ValueError("SM120 FP8 blockwise weight scales must be contiguous")
        N, K = weight.shape
        scale_N, scale_K = weight_scales.shape
        if (N + block_size - 1) // block_size != scale_N or (
            K + block_size - 1
        ) // block_size != scale_K:
            raise ValueError(
                "SM120 FP8 blockwise weight scale dimension mismatch: "
                f"N={N}, scale_N={scale_N}, K={K}, scale_K={scale_K}"
            )
        return (
            weight,
            weight_scales,
            K,
            N,
            scale_K,
            scale_N,
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
        return (
            weight_scales is not None
            and quant_config is not None
            and quant_config.get_method() == "FP8_PER_BLOCK"
            and weight.dtype == torch.float8_e4m3fn
            and is_sm120(weight.device)
            and weight_scales.dtype in (torch.float32, torch.float8_e8m0fnu)
            and sm120_blockwise_fp8_available(weight.device)
            and weight.dim() == 2
            and weight_scales.dim() == 2
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
        if weight_scales is None:
            raise ValueError("SM120 FP8 blockwise GEMM requires weight_scales")
        self._gemm_op = sm120_blockwise_fp8_op(weight.device)
        if self._gemm_op is None:
            raise RuntimeError(
                "cutlass_scaled_mm_blockwise_sm120_fp8 op is not available; "
                "this backend requires a cuda13_sm120 build with -DENABLE_FP8_SM120."
            )

        self.weight = weight
        # V4 checkpoints store UE8M0 as float8. CUTLASS consumes the decoded
        # power-of-two values as float32, while preserving the compact weight
        # storage (no eager replicated/int32 scale expansion).
        self.weight_scales = weight_scales.float().contiguous()
        self.input_scales = input_scales
        self.bias = bias

        if self.weight.dim() != 2 or self.weight_scales.dim() != 2:
            raise ValueError(
                f"Weight and weight scale must be 2D tensors, got weight dim "
                f"{self.weight.dim()} and weight scale dim {self.weight_scales.dim()}"
            )

        (
            self.weight,
            self.weight_scales,
            self.K,
            self.N,
            self.scale_K,
            self.scale_N,
        ) = self._restore_blockwise_weight_layout(self.weight, self.weight_scales)

        if self.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"Weight dtype must be float8_e4m3fn, got {self.weight.dtype}"
            )

        self.kernel_K = self.scale_K * 128
        self.kernel_N = self.scale_N * 128
        if (self.N, self.K) != (self.kernel_N, self.kernel_K):
            padded_weight = self.weight.new_zeros((self.kernel_N, self.kernel_K))
            padded_weight[: self.N, : self.K].copy_(self.weight)
            self.weight = padded_weight

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
            self.bias = self.bias.to(device=self.weight.device)

    def forward(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
        if not input.is_contiguous():
            raise ValueError("SM120 FP8 blockwise GEMM input must be contiguous")
        if out is not None:
            if out.dtype != torch.bfloat16:
                raise ValueError(
                    f"SM120 FP8 blockwise GEMM output must be bfloat16, got {out.dtype}"
                )
            if out.shape != (M, self.N):
                raise ValueError(
                    f"SM120 FP8 blockwise GEMM output shape must be {(M, self.N)}, "
                    f"got {tuple(out.shape)}"
                )
            if out.device != input.device:
                raise ValueError(
                    "SM120 FP8 blockwise GEMM output must share input device"
                )
            if not out.is_contiguous():
                raise ValueError("SM120 FP8 blockwise GEMM output must be contiguous")
        if M == 0:
            return (
                out
                if out is not None
                else torch.empty(0, self.N, dtype=torch.bfloat16, device=input.device)
            )

        if self.K != self.kernel_K:
            padded_input = input.new_zeros((M, self.kernel_K))
            padded_input[:, : self.K].copy_(input)
            input = padded_input

        input_fp8, input_scales = sgl_per_token_group_quant_fp8(
            input,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )

        output = (
            out
            if out is not None and self.N == self.kernel_N
            else torch.empty(
                M, self.kernel_N, dtype=torch.bfloat16, device=input.device
            )
        )
        self._gemm_op(
            output,
            input_fp8,
            self.weight,
            input_scales,
            self.weight_scales,
        )
        if self.N != self.kernel_N:
            output = output[:, : self.N]
            if out is not None:
                out.copy_(output)
                output = out
            else:
                output = output.contiguous()
        if self.bias is not None:
            output.add_(self.bias)
        return output

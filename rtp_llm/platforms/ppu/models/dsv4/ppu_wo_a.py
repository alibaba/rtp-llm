"""DeepSeek V4 grouped output projection using the common PPU FP8 ABI."""

import os
from typing import Optional

import torch
from rtp_llm.platforms.ppu.modules.linear.fp8_linear import (
    FP8_BLOCK_SIZE,
    _require_dtype,
    _require_m890p,
    _resolve_deep_gemm_symbol,
    _validate_fp8_quantization,
    _validate_out,
    checkpoint_ue8m0_scale_to_fp32,
    quantize_ppu_fp8_activation,
)
from torch import nn


class PpuWoAFp8Linear(nn.Module):
    """M890P grouped ``wo_a`` FP8 projection.

    ``rank`` is derived from the checkpoint weight and therefore is not an API
    parameter.  DeepSeek V4 has eight attention groups, so TP1/2/4/8 map to
    ``groups=8/4/2/1`` and share the same validation and launch path.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        checkpoint_scale: torch.Tensor,
        *,
        groups: int,
        k_local: int,
        sglang_layout: Optional[bool] = None,
        quantization: str = "auto",
    ):
        super().__init__()
        _validate_fp8_quantization(quantization)
        if quantization == "v2_column":
            raise ValueError("PPU wo_a fused permutation requires row-major scales")
        self.quantization = quantization
        self.sglang_layout = (
            (os.environ.get("DSV4_PPU_SGLANG_WO_A", "0") == "1")
            if sglang_layout is None
            else bool(sglang_layout)
        )
        if weight.ndim != 2:
            raise ValueError(f"wo_a weight must be 2D, got {weight.ndim}D")
        if weight.dtype != _require_dtype("float8_e4m3fn"):
            raise TypeError(
                f"wo_a weight must be torch.float8_e4m3fn, got {weight.dtype}"
            )
        _require_m890p(weight, "wo_a weight")
        if groups not in (1, 2, 4, 8):
            raise ValueError(
                "wo_a groups must match a supported TP split "
                f"(TP1=8, TP2=4, TP4=2, TP8=1), got {groups}"
            )
        if k_local <= 0 or k_local % FP8_BLOCK_SIZE:
            raise ValueError(
                f"wo_a k_local must be a positive multiple of 128, got {k_local}"
            )
        if int(weight.shape[1]) != k_local:
            raise ValueError(
                f"wo_a weight K must equal k_local={k_local}, " f"got {weight.shape[1]}"
            )
        if int(weight.shape[0]) % groups:
            raise ValueError(
                f"wo_a weight N={weight.shape[0]} must be divisible by groups={groups}"
            )
        rank = int(weight.shape[0]) // groups
        if rank <= 0 or rank % FP8_BLOCK_SIZE:
            raise ValueError(
                f"derived wo_a rank must be a positive multiple of 128, got {rank}"
            )

        scale_fp32 = checkpoint_ue8m0_scale_to_fp32(checkpoint_scale, weight.shape)
        if scale_fp32.device != weight.device:
            raise ValueError(
                f"wo_a weight and checkpoint scale must share a device, got "
                f"{weight.device} and {scale_fp32.device}"
            )

        self.groups = groups
        self.rank = rank
        self.k_local = k_local
        self.register_buffer(
            "weight",
            weight.view(groups, rank, k_local),
            persistent=False,
        )
        self.register_buffer(
            "weight_scale",
            scale_fp32.view(
                groups,
                rank // FP8_BLOCK_SIZE,
                k_local // FP8_BLOCK_SIZE,
            ),
            persistent=False,
        )
        self._einsum = _resolve_deep_gemm_symbol("fp8_einsum")

    def forward(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim < 3 or tuple(x.shape[-2:]) != (self.groups, self.k_local):
            raise ValueError(
                "wo_a activation shape must end in "
                f"(groups, k_local)=({self.groups}, {self.k_local}), "
                f"got {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16:
            raise TypeError(f"wo_a activation must be torch.bfloat16, got {x.dtype}")
        _require_m890p(x, "wo_a activation")
        if x.device != self.weight.device:
            raise ValueError(
                f"wo_a activation device must be {self.weight.device}, got {x.device}"
            )

        output_shape = tuple(x.shape[:-2]) + (self.groups, self.rank)
        output = _validate_out(
            out,
            output_shape,
            x.device,
            (x, self.weight, self.weight_scale),
        )
        m = x.numel() // (self.groups * self.k_local)
        if m == 0:
            return output

        # Quantization retains the token-major arithmetic and block scales.
        x_2d = x.view(m * self.groups, self.k_local)
        x_fp8, x_scale = quantize_ppu_fp8_activation(
            x_2d, quantization=self.quantization
        )
        x_fp8 = x_fp8.view(m, self.groups, self.k_local)
        x_scale = x_scale.view(m, self.groups, self.k_local // FP8_BLOCK_SIZE)
        output_3d = output.view(m, self.groups, self.rank)
        if self.sglang_layout:
            from deep_gemm.jit_kernels.einsum import fp8_bmm
            from rtp_llm.platforms.ppu.kernels.cuda.ppu_sglang_permute import (
                fused_permute,
            )

            perm_a, perm_scale = fused_permute(x_fp8, x_scale)
            fp8_bmm(perm_a, perm_scale, self.weight, self.weight_scale, output_3d)
            return output
        try:
            self._einsum(
                "bhr,hdr->bhd",
                (x_fp8, x_scale),
                (self.weight, self.weight_scale),
                output_3d,
                recipe=(1, 1, FP8_BLOCK_SIZE),
            )
        except TypeError as exc:
            raise RuntimeError(
                "deep_gemm.fp8_einsum ABI mismatch for the M890P DSV4 "
                "contract; expected equation, lhs_pair, rhs_pair, out, recipe"
            ) from exc
        return output


__all__ = ["PpuWoAFp8Linear"]

"""M890P routed-expert packed-FP4 linear adapter."""

from __future__ import annotations

import torch
from rtp_llm.platforms.ppu.kernels.ppu_mxfp4 import (
    downcast_to_mxfp4,
    gemm_fp4_fp4_bf16_nt,
)
from torch import nn


class PpuFp4Linear(nn.Module):
    """Keep checkpoint FP4 payloads packed and launch the PPU W4A4 GEMM."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        scale_gemm: torch.Tensor,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        if weight.dtype not in (torch.int8, torch.uint8):
            raise TypeError(
                f"PPU FP4 weight must be packed int8/uint8, got {weight.dtype}"
            )
        if tuple(weight.shape) != (out_features, in_features // 2):
            raise ValueError(
                "PPU FP4 weight shape mismatch: "
                f"got {tuple(weight.shape)}, expected {(out_features, in_features // 2)}"
            )
        if scale_gemm is None or scale_gemm.dtype != torch.uint16:
            raise TypeError(
                "PPU FP4 scale_gemm must be platform-prepared uint16, got "
                f"{getattr(scale_gemm, 'dtype', None)}"
            )
        if not weight.is_cuda or not scale_gemm.is_cuda:
            raise ValueError("PPU FP4 weight and prepared scale must be device tensors")
        if weight.device != scale_gemm.device:
            raise ValueError("PPU FP4 weight and prepared scale must share a device")
        if not weight.is_contiguous():
            raise ValueError("PPU FP4 weight must be contiguous")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = weight.view(torch.uint8) if weight.dtype == torch.int8 else weight
        self.scale = scale
        self.scale_gemm = scale_gemm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"PPU FP4 input K must be {self.in_features}, got {x.shape[-1]}"
            )
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        if x_2d.dtype not in (torch.bfloat16, torch.float32):
            x_2d = x_2d.to(torch.bfloat16)
        activation = downcast_to_mxfp4(x_2d)
        output = gemm_fp4_fp4_bf16_nt(
            activation,
            (self.weight, self.scale_gemm),
        )
        return output.reshape(*original_shape[:-1], self.out_features)


__all__ = ["PpuFp4Linear"]

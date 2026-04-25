"""Quantization-aware Linear for DeepSeek-V4.

Stores weights in their native checkpoint dtype (FP4 e2m1 packed in int8,
FP8 e4m3fn, or BF16) + companion scale parameters. Dequantizes on-the-fly
during forward pass — avoids the 4x memory blowup from eagerly dequantizing
to BF16 at load time.

Memory footprint:
  FP4 weight:  [out, in//2] int8 + [out, in//32] UE8M0  ≈ original + 1/32 scale
  FP8 weight:  [out, in] e4m3fn + [out//128, in//128] UE8M0  ≈ original + tiny scale
  BF16 weight: [out, in] bf16  — no scale

For now, forward uses PyTorch reference dequant (slow but memory-correct).
M6 will swap in TileLang fp4_gemm / fp8_gemm / act_quant kernels.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FP8_BLOCK = 128
FP4_BLOCK = 32

# FP4 e2m1 lookup: 4-bit raw -> fp32 value
_FP4_LUT = torch.tensor([
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def _fp4_unpack_to_fp32(weight_int8: torch.Tensor, scale_ue8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize packed FP4 [out, in/2] + UE8M0 scale [out, in/32] -> fp32 [out, in]."""
    out_dim, packed_in = weight_int8.shape
    in_dim = packed_in * 2
    w_uint = weight_int8.to(torch.int32) & 0xFF
    low = w_uint & 0x0F
    high = (w_uint >> 4) & 0x0F
    interleaved = torch.empty(out_dim, in_dim, dtype=torch.int64, device=weight_int8.device)
    interleaved[:, 0::2] = low.long()
    interleaved[:, 1::2] = high.long()
    lut = _FP4_LUT.to(weight_int8.device)
    w_f = lut[interleaved]
    scale_f = scale_ue8m0.to(torch.float32).repeat_interleave(FP4_BLOCK, 1)[:, :in_dim]
    return w_f * scale_f


def _fp8_dequant_to_fp32(weight_fp8: torch.Tensor, scale_ue8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 e4m3fn [out, in] + UE8M0 scale [out/128, in/128] -> fp32 [out, in]."""
    out_dim, in_dim = weight_fp8.shape
    w_f = weight_fp8.to(torch.float32)
    scale_full = scale_ue8m0.to(torch.float32)
    scale_full = scale_full.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    scale_full = scale_full[:out_dim, :in_dim]
    return w_f * scale_full


class QuantizedLinear(nn.Module):
    """Linear layer holding native-dtype weight + scale; dequants in forward.

    Three modes selected at construction via `storage`:
      - "fp4":  weight int8 [out, in//2], scale UE8M0 [out, in//32]
      - "fp8":  weight float8_e4m3fn [out, in], scale UE8M0 [out//128, in//128]
      - "bf16": plain bf16 weight [out, in], no scale

    Checkpoint loading populates `.weight` and `.scale` directly without
    dequantizing. M6 will swap the forward impl for TileLang fp{4,8}_gemm.
    """

    def __init__(self, in_features: int, out_features: int, storage: str = "bf16",
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.storage = storage
        assert bias is False, "V4 linears have no bias"
        if storage == "fp4":
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features // 2, dtype=torch.int8),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(out_features, in_features // FP4_BLOCK, dtype=torch.float8_e8m0fnu),
                requires_grad=False,
            )
        elif storage == "fp8":
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(
                    (out_features + FP8_BLOCK - 1) // FP8_BLOCK,
                    (in_features + FP8_BLOCK - 1) // FP8_BLOCK,
                    dtype=torch.float8_e8m0fnu,
                ),
                requires_grad=False,
            )
        elif storage == "bf16":
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))
            self.register_parameter("scale", None)
        else:
            raise ValueError(f"unknown storage {storage!r}")

    def dequant_weight(self, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Return dequantized [out, in] weight in `out_dtype`.

        PyTorch reference; slow but correct. For perf, M6 replaces Linear
        with fused fp{4,8}_gemm that never materializes the dequantized weight.
        """
        if self.storage == "fp4":
            return _fp4_unpack_to_fp32(self.weight, self.scale).to(out_dtype)
        if self.storage == "fp8":
            return _fp8_dequant_to_fp32(self.weight, self.scale).to(out_dtype)
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.storage == "bf16":
            return F.linear(x, self.weight)
        # FP4/FP8: dequant to x's dtype on the fly.
        w = self.dequant_weight(out_dtype=x.dtype)
        return F.linear(x, w)

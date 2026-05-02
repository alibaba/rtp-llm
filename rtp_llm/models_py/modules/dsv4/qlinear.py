"""Quantization-aware Linear for DeepSeek-V4.

Stores weights in their native checkpoint dtype (FP4 e2m1 packed in int8,
FP8 e4m3fn, or BF16) + companion scale parameters. Dequantizes on-the-fly
during forward pass — avoids the 4x memory blowup from eagerly dequantizing
to BF16 at load time.

Memory footprint:
  FP4 weight:  [out, in//2] int8 + [out, in//32] UE8M0  ≈ original + 1/32 scale
  FP8 weight:  [out, in] e4m3fn + [out//128, in//128] UE8M0  ≈ original + tiny scale
  BF16 weight: [out, in] bf16  — no scale

FP4 forward goes through ``deep_gemm.fp8_fp4_gemm_nt`` (K2 in
``docs/dsv4/kernel_audit.md``) when DeepGEMM ≥ 2.4 is installed and the
input is on CUDA; otherwise it falls back to the PyTorch dequant path
(slow but memory-correct, useful for CPU-only unit tests).  Matches V4
official ``inference/model.py``'s per-expert ``Linear.forward`` which
calls V4's own TileLang ``fp4_gemm`` — same math, different kernel.

FP8 forward stays on the PyTorch dequant path for now; the factory-mode
FP8 linears already migrated in S2 to ``CudaFp8DeepGEMMLinear`` and don't
go through this class for attention/indexer/shared-expert linears.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FP8_BLOCK = 128
FP4_BLOCK = 32

_WARNED_FP4_FALLBACK: bool = False

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
    """Dequantize FP8 e4m3fn [out, in] + scale -> fp32 [out, in].

    Accepts the scale in either of two layouts:
      * raw UE8M0 ``float8_e8m0fnu`` shape ``[out/128, in/128]``
      * DeepGEMM TMA-aligned packed ``int32`` shape ``[out_pad, in/128/4]`` where
        each int32 holds 4 UE8M0 bytes along K and rows are replicated by 128
        (this is what ``get_mn_major_tma_aligned_packed_ue8m0_tensor`` produces;
        ``rtp_llm/model_loader/per_block_fp8_quant_weight.py::_postprocess``
        applies it via ``requant_weight_ue8m0`` on SM100+ devices).
    """
    out_dim, in_dim = weight_fp8.shape
    w_f = weight_fp8.to(torch.float32)
    if scale_ue8m0.dtype == torch.int32:
        n_pad, k_blk_div_4 = scale_ue8m0.shape
        k_blk = k_blk_div_4 * 4
        scale_bytes = scale_ue8m0.contiguous().view(torch.uint8).reshape(n_pad, k_blk)
        scale_per_row = (scale_bytes.to(torch.int32) - 127).to(torch.float32).exp2()
        scale_per_row = scale_per_row[:out_dim]
        scale_full = scale_per_row.repeat_interleave(FP8_BLOCK, 1)[:, :in_dim]
    else:
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

    def _fp4_forward_deepgemm(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Try the native FP4 kernel.  Returns None to signal fall-back.

        Matches V4 official ``inference/model.py``'s per-expert linear:
        quant ``x`` to FP8 e4m3fn with UE8M0 block-128 scale along K,
        then run FP8 × packed-FP4 GEMM against our stored weight/scale.
        """
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            _fp8_fp4_gemm_nt_impl, fp8_fp4_gemm_nt,
        )
        if _fp8_fp4_gemm_nt_impl is None or not x.is_cuda:
            return None
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)
        M = x_2d.shape[0]
        if M == 0:
            return x.new_empty(*orig_shape[:-1], self.out_features)

        x_fp8, x_scale = sgl_per_token_group_quant_fp8(
            x_2d, group_size=FP8_BLOCK, eps=1e-4,
            column_major_scales=True, scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        out = torch.empty(M, self.out_features, dtype=torch.bfloat16, device=x.device)
        fp8_fp4_gemm_nt(
            (x_fp8, x_scale),
            (self.weight, self.scale.float()),
            out,
            recipe_a=(1, FP8_BLOCK), recipe_b=(1, FP4_BLOCK),
        )
        return out.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.storage == "bf16":
            return F.linear(x, self.weight)
        if self.storage == "fp4":
            y = self._fp4_forward_deepgemm(x)
            if y is not None:
                return y
            global _WARNED_FP4_FALLBACK
            if not _WARNED_FP4_FALLBACK:
                _WARNED_FP4_FALLBACK = True
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "[dsv4] deep_gemm.fp8_fp4_gemm_nt unavailable; FP4 "
                    "linears fall back to PyTorch dequant (slow)",
                )
        # FP8, or FP4 fallback: dequant to x's dtype on the fly.
        w = self.dequant_weight(out_dtype=x.dtype)
        return F.linear(x, w)

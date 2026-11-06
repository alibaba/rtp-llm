"""Quantization-aware Linear for FP8/FP4 MoE experts.

Stores weights in their native checkpoint dtype (FP4 e2m1 packed in int8,
FP8 e4m3fn, or BF16) plus companion scale parameters. FP4 executes through
DeepGEMM without materializing BF16 weights; FP8 retains an eager dequant
fallback. Native checkpoint storage avoids an eager load-time memory blowup.

Memory footprint:
  FP4 weight:  [out, in//2] int8 + [out, in//32] UE8M0  ≈ original + 1/32 scale
  FP8 weight:  [out, in] e4m3fn + [out//128, in//128] UE8M0  ≈ original + tiny scale
  MXFP8 weight: [out, in] e4m3fn + [out//32, in//32] UE8M0  ≈ original + 1/32 scale
  BF16 weight: [out, in] bf16  — no scale

FP4 forward goes through ``deep_gemm.fp8_fp4_gemm_nt`` and requires a
compatible DeepGEMM build on CUDA.

FP8 forward stays on the PyTorch dequant path for now; the factory-mode
FP8 linears already migrated in S2 to ``CudaFp8DeepGEMMLinear`` and don't
go through this class for attention/indexer/shared-expert linears.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
    FP8_BLOCK,
    dequantize_fp4_weight,
    dequantize_fp8_weight,
    prepare_fp4_weight_scale_for_deepgemm,
)

_FP8_BLOCK_CONFIG = Fp8BlockWiseQuantConfig()

MXFP8_BLOCK = 32


def is_mxfp8_scale(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    return (
        scale.dim() == 2
        and scale.dtype == torch.float8_e8m0fnu
        and weight.dim() == 2
        and tuple(scale.shape)
        == (
            (weight.shape[0] + MXFP8_BLOCK - 1) // MXFP8_BLOCK,
            (weight.shape[1] + MXFP8_BLOCK - 1) // MXFP8_BLOCK,
        )
    )


def create_fp8_linear(weight: torch.Tensor, scale: torch.Tensor):
    """Build the CUDA FP8 linear used by FP8 shared experts."""
    from rtp_llm.models_py.modules.factory.linear import LinearFactory

    if is_mxfp8_scale(weight, scale):
        return Mxfp8Linear(weight, scale)

    if scale.dtype == torch.float8_e8m0fnu:
        from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

        if scale.dim() != 2:
            raise ValueError(f"FP8 scale must be 2D, got {scale.dim()}D")
        row_count = scale.shape[0] * FP8_BLOCK
        row_index = torch.arange(row_count, device=scale.device) // FP8_BLOCK
        scale = get_mn_major_tma_aligned_packed_ue8m0_tensor(
            scale.float().index_select(-2, row_index)
        )
    local_weights = {"weight": weight, "scale": scale}
    return LinearFactory.create_linear_from_weights(
        local_weights,
        "weight",
        "scale",
        quant_config=_FP8_BLOCK_CONFIG,
    )


class QuantizedLinear(nn.Module):
    """Linear layer holding native-dtype weight and scale.

    Three modes selected at construction via `storage`:
      - "fp4":  weight int8 [out, in//2], scale UE8M0 [out, in//32]
      - "fp8":  weight float8_e4m3fn [out, in], scale UE8M0 [out//128, in//128]
      - "bf16": plain bf16 weight [out, in], no scale

    Checkpoint loading binds `.weight` and `.scale` directly. FP4 uses the
    native DeepGEMM kernel; FP8 keeps a reference dequant path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        storage: str = "bf16",
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.storage = storage
        assert bias is False, "FP8/FP4 expert linears do not support bias"
        if storage not in {"fp4", "fp8", "bf16"}:
            raise ValueError(f"unknown storage {storage!r}")
        # Weight and scale are bound directly from the framework's
        # ``ModelWeights`` tensors — no
        # ``nn.Parameter`` wrapping, no ``torch.empty`` placeholder.  Plain
        # attributes avoid double-holding the storage in ``module._parameters``
        # on top of the tensor that already lives in the loader's hands.
        self.weight = None
        self.scale = None
        self.scale_gemm = None

    def bind_fp4_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        scale_gemm: Optional[torch.Tensor] = None,
    ) -> None:
        """Bind FP4 weight plus a prepacked DeepGEMM scale."""
        self.weight = weight
        self.scale = scale
        self.scale_gemm = scale_gemm
        if self.scale_gemm is None:
            self.scale_gemm = prepare_fp4_weight_scale_for_deepgemm(
                scale, self.out_features, self.in_features
            )
        if self.scale_gemm.dtype != torch.int32:
            raise TypeError(
                f"expected packed FP4 scale int32, got {self.scale_gemm.dtype}"
            )

    def dequant_weight(self, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Return dequantized [out, in] weight in `out_dtype`.

        PyTorch reference; slow but correct. For perf, M6 replaces Linear
        with fused fp{4,8}_gemm that never materializes the dequantized weight.
        """
        if self.storage == "fp4":
            return dequantize_fp4_weight(self.weight, self.scale).to(out_dtype)
        if self.storage == "fp8":
            return dequantize_fp8_weight(self.weight, self.scale).to(out_dtype)
        return self.weight

    def _fp4_forward_deepgemm(self, x: torch.Tensor) -> torch.Tensor:
        """Run the native FP4 kernel.

        Quantize ``x`` to FP8 e4m3fn with UE8M0 block-128 scale along K,
        then run FP8 × packed-FP4 GEMM against the stored weight and scale.
        """
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            _fp8_fp4_gemm_nt_impl,
            fp8_fp4_gemm_nt,
        )

        if _fp8_fp4_gemm_nt_impl is None or not x.is_cuda:
            raise RuntimeError(
                "FP4 QuantizedLinear requires deep_gemm fp8_fp4_gemm_nt "
                f"on CUDA; got device={x.device}, impl={_fp8_fp4_gemm_nt_impl}"
            )
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)
        M = x_2d.shape[0]
        if M == 0:
            return x.new_empty(*orig_shape[:-1], self.out_features)

        x_fp8, x_scale = sgl_per_token_group_quant_fp8(
            x_2d,
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        out = torch.empty(M, self.out_features, dtype=torch.bfloat16, device=x.device)
        if self.scale_gemm is None or self.scale_gemm.dtype != torch.int32:
            raise RuntimeError(
                "FP4 QuantizedLinear requires init-time packed int32 scale"
            )
        fp8_fp4_gemm_nt(
            (x_fp8, x_scale),
            (self.weight, self.scale_gemm),
            out,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        return out.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.storage == "bf16":
            return F.linear(x, self.weight)
        if self.storage == "fp4":
            return self._fp4_forward_deepgemm(x)
        # FP8: dequant to x's dtype on the fly.
        w = self.dequant_weight(out_dtype=x.dtype)
        return F.linear(x, w)


class Mxfp8Linear(nn.Module):
    """Linear with native 32x32 checkpoint quantization blocks.

    Checkpoints store one UE8M0 scale per 32x32 weight tile. DeepGEMM needs one
    scale per 32-element group along K, so the leading dimension is expanded by
    a pure row broadcast (``row // 32``) and packed into the same TMA-aligned
    int32 form used by the block-128 path.

    The activation contract is group-32 FP8 as well: ``forward`` quantizes the
    input with ``sgl_per_token_group_quant_fp8(group_size=32)`` and runs
    ``fp8_fp4_gemm_nt`` with ``recipe=(1, 1, 32)``.
    """

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.weight = weight
        self.weight_scales = self.pack_scales(scale)

    def pack_scales(self, scale: torch.Tensor) -> torch.Tensor:
        if not is_mxfp8_scale(self.weight, scale):
            raise ValueError(
                f"invalid MXFP8 scale {tuple(scale.shape)} for weight "
                f"{tuple(self.weight.shape)}"
            )
        from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

        rows = torch.arange(self.out_features, device=scale.device) // MXFP8_BLOCK
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(
            scale.float().index_select(0, rows)
        )

    def _quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        flat = x.reshape(-1, self.in_features).contiguous()
        quantized, scales = sgl_per_token_group_quant_fp8(
            flat,
            group_size=MXFP8_BLOCK,
            eps=torch.finfo(torch.float32).tiny,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        return quantized.view(x.shape), scales

    def forward_quantized(
        self,
        x_q: torch.Tensor,
        x_s: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Consume group-32 E4M3/packed UE8M0 without another quantization.

        Used by the fused shared-expert activation kernel, which hands over the
        quantized SwiGLU output directly. Input layout and disjoint BF16 output
        storage are caller preconditions.
        """
        m = x_q.numel() // self.in_features
        shape = (*x_q.shape[:-1], self.out_features)
        output = (
            out
            if out is not None
            else torch.empty(shape, device=x_q.device, dtype=torch.bfloat16)
        )
        if m:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_fp4_gemm_nt

            fp8_fp4_gemm_nt(
                (x_q.reshape(m, self.in_features), x_s),
                (self.weight, self.weight_scales),
                output.reshape(m, self.out_features),
                recipe=(1, 1, MXFP8_BLOCK),
            )
        return output

    def forward(
        self, x: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.numel() == 0:
            return (
                out
                if out is not None
                else x.new_empty((*x.shape[:-1], self.out_features))
            )
        return self.forward_quantized(*self._quantize_input(x), out=out)

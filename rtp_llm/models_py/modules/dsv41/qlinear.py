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
``docs/dsv4/kernel_audit.md``) and requires DeepGEMM ≥ 2.4 on CUDA.  Matches V4
official ``inference/model.py``'s per-expert ``Linear.forward`` which
calls V4's own TileLang ``fp4_gemm`` — same math, different kernel.

Pre-SM100 devices (e.g. Ada sm89) have no packed-UE8M0 DeepGEMM FP4 kernel;
there FP4 forward falls back to the exact ``dequant_weight`` bf16 GEMM. FP4
e2m1 values times power-of-two UE8M0 scales are precisely representable in
bf16, so the fallback introduces no additional quantization.

FP8 forward stays on the PyTorch dequant path for now; the factory-mode
FP8 linears already migrated in S2 to ``CudaFp8DeepGEMMLinear`` and don't
go through this class for attention/indexer/shared-expert linears.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.quant_layouts import (
    prepare_fp4_weight_scale_for_deepgemm,
)

FP8_BLOCK = 128
FP4_BLOCK = 32
# Decode-sized batches dequantize FP4 experts block-by-block instead of
# materializing the whole weight (graph-capture pools stay small).
_FP4_CHUNKED_GEMM_MAX_ROWS = int(
    os.environ.get("DSV41_FP4_CHUNKED_GEMM_MAX_ROWS", "256")
)
_FP4_DEQUANT_GEMM_CHUNK_ROWS = 128


def native_fp4_gemm_supported(device: Optional[torch.device] = None) -> bool:
    """Whether DeepGEMM's FP8xFP4 kernels (packed UE8M0 SF) can run here.

    The fp8_fp4 family is SM100+ only; the layout helpers reject other
    architectures ("Unknown SF transformation"). Non-CUDA and pre-SM100
    devices must use the exact bf16 dequant fallback instead.
    """
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device)[0] >= 10

# FP4 e2m1 lookup: 4-bit raw -> fp32 value
_FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


_FP4_LUT_CACHE: Dict[torch.device, torch.Tensor] = {}


def _fp4_lut(device: torch.device) -> torch.Tensor:
    """Device-resident FP4 code->value table, materialized once per device.

    A per-call ``_FP4_LUT.to(device)`` is a pageable host->device copy, which
    CUDA graph capture rejects ("Cannot copy between CPU and CUDA tensors
    during capture"). The eager warmup that precedes every capture populates
    this cache, so capture and replay only index a constant device tensor.
    """
    lut = _FP4_LUT_CACHE.get(device)
    if lut is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FP4 dequant LUT must be materialized before graph capture"
            )
        lut = _FP4_LUT.to(device)
        _FP4_LUT_CACHE[device] = lut
    return lut


def _fp4_unpack_to_fp32(
    weight_int8: torch.Tensor, scale_ue8m0: torch.Tensor
) -> torch.Tensor:
    """Dequantize packed FP4 [out, in/2] + UE8M0 scale [out, in/32] -> fp32 [out, in]."""
    out_dim, packed_in = weight_int8.shape
    in_dim = packed_in * 2
    w_uint = weight_int8.to(torch.int32) & 0xFF
    low = w_uint & 0x0F
    high = (w_uint >> 4) & 0x0F
    interleaved = torch.empty(
        out_dim, in_dim, dtype=torch.int64, device=weight_int8.device
    )
    interleaved[:, 0::2] = low.long()
    interleaved[:, 1::2] = high.long()
    lut = _fp4_lut(weight_int8.device)
    w_f = lut[interleaved]
    scale_f = scale_ue8m0.to(torch.float32).repeat_interleave(FP4_BLOCK, 1)[:, :in_dim]
    return w_f * scale_f


def _fp4_unpack_rows_to_bf16(
    weight_int8: torch.Tensor, scale_ue8m0: torch.Tensor
) -> torch.Tensor:
    """Exact bf16 dequant of packed FP4 rows for block-wise GEMMs.

    Same math as :func:`_fp4_unpack_to_fp32` without the int64 interleave
    temporary: the LUT gathers land directly in the strided halves. e2m1
    values carry <=4 significand bits, so the fp32 product with the
    power-of-two scale casts to bf16 losslessly.
    """
    out_dim, packed_in = weight_int8.shape
    in_dim = packed_in * 2
    w_uint = weight_int8.to(torch.int32) & 0xFF
    low = w_uint & 0x0F
    high = (w_uint >> 4) & 0x0F
    lut = _fp4_lut(weight_int8.device)
    w_f = torch.empty(out_dim, in_dim, dtype=torch.float32, device=weight_int8.device)
    w_f[:, 0::2] = lut[low.long()]
    w_f[:, 1::2] = lut[high.long()]
    scale_f = scale_ue8m0.to(torch.float32).repeat_interleave(FP4_BLOCK, 1)[:, :in_dim]
    return (w_f * scale_f).to(torch.bfloat16)


def _fp8_dequant_to_fp32(
    weight_fp8: torch.Tensor, scale_ue8m0: torch.Tensor
) -> torch.Tensor:
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
        scale_full = scale_full.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(
            FP8_BLOCK, 1
        )
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
        assert bias is False, "V4 linears have no bias"
        if storage not in {"fp4", "fp8", "bf16"}:
            raise ValueError(f"unknown storage {storage!r}")
        # weight / scale are bound externally by V4Transformer factory mode
        # directly from the framework's ``ModelWeights`` tensors — no
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
        """Bind FP4 weight plus (on SM100+) a prepacked DeepGEMM scale.

        Pre-SM100 devices keep ``scale_gemm=None`` and forward through the
        exact bf16 dequant path; the DeepGEMM SF layout helper rejects
        those architectures at bind time.
        """
        self.weight = weight
        self.scale = scale
        self.scale_gemm = scale_gemm
        if self.scale_gemm is None and native_fp4_gemm_supported(
            weight.device if weight.is_cuda else None
        ):
            self.scale_gemm = prepare_fp4_weight_scale_for_deepgemm(
                scale, self.out_features, self.in_features
            )
        if self.scale_gemm is not None and self.scale_gemm.dtype != torch.int32:
            raise TypeError(
                f"expected packed FP4 scale int32, got {self.scale_gemm.dtype}"
            )

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

    def _fp4_forward_deepgemm(self, x: torch.Tensor) -> torch.Tensor:
        """Run the native FP4 kernel.

        Matches V4 official ``inference/model.py``'s per-expert linear:
        quant ``x`` to FP8 e4m3fn with UE8M0 block-128 scale along K,
        then run FP8 × packed-FP4 GEMM against our stored weight/scale.
        """
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_fp4_gemm_nt
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
        fp8_fp4_gemm_nt(
            (x_fp8, x_scale),
            (self.weight, self.scale_gemm),
            out,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        return out.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)

    def _forward_fp4_chunked(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Bounded-transient exact FP4 GEMM for decode-sized batches.

        Dequantizes one 128-row block of the packed expert weight at a time
        and GEMMs it into the output slice, so per-call transients stay at a
        few MB. Whole-weight fp32 dequants (~90MB per expert call) pinned a
        fresh transient per captured batch size and OOM'd bs4 decode-graph
        capture on 44.5/46GB cards.
        """
        x_bf16 = x_2d.to(torch.bfloat16)
        output = torch.empty(
            (x_2d.shape[0], self.out_features),
            dtype=torch.bfloat16,
            device=x_2d.device,
        )
        for begin in range(0, self.out_features, _FP4_DEQUANT_GEMM_CHUNK_ROWS):
            end = min(begin + _FP4_DEQUANT_GEMM_CHUNK_ROWS, self.out_features)
            w_chunk = _fp4_unpack_rows_to_bf16(
                self.weight[begin:end], self.scale[begin:end]
            )
            output[:, begin:end] = F.linear(x_bf16, w_chunk)
            del w_chunk
        return output if output.dtype == x_2d.dtype else output.to(x_2d.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.storage == "bf16":
            return F.linear(x, self.weight)
        if self.storage == "fp4":
            if self.scale_gemm is not None:
                return self._fp4_forward_deepgemm(x)
            # Pre-SM100 exact fallback: FP4 e2m1 × 2^k dequantizes losslessly
            # to bf16, then a plain GEMM keeps the checkpoint semantics.
            # Decode-sized batches take the block-wise path; large prefill
            # batches keep the faster whole-weight dequant.
            rows = x.numel() // self.in_features if x.numel() else 0
            if 0 < rows <= _FP4_CHUNKED_GEMM_MAX_ROWS:
                y = self._forward_fp4_chunked(x.reshape(-1, self.in_features))
                return y.reshape(*x.shape[:-1], self.out_features)
            w = self.dequant_weight(out_dtype=x.dtype)
            return F.linear(x, w)
        # FP8: dequant to x's dtype on the fly.
        w = self.dequant_weight(out_dtype=x.dtype)
        return F.linear(x, w)

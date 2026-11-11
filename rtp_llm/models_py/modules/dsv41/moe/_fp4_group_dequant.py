"""Fused Triton FP4→bf16 dequantization for stacked (grouped) expert weights.

The reference ``qlinear._fp4_unpack_rows_to_bf16`` materializes three fp32
[R, K] temporaries (LUT gather, broadcast scale, product) — ~1.1GB peak for
one 8-expert w1 group, which OOMs tight-HBM ranks during decode-graph
warmup (load38g: rank0 died on the 360MiB product with 359.88MiB free), and
row-slicing it back to bounded transients multiplies kernel launches, which
is exactly what CUDA-graph capture cannot afford (~44K nodes => >1.2GB
cudaGraphInstantiate, load38f).

This kernel decodes both nibbles of each packed byte directly against the
device LUT and applies the UE8M0 scale in-register, writing bf16 output in a
single launch per weight group. Values are bitwise identical to the
reference: e2m1 significands are <=4 bits and scales are powers of two, so
the fp32 product is exact and the bf16 cast is the same round-to-nearest-even
(as is -0.0, which lives in the LUT). Scale byte 0 decodes to the 2^-127
denormal via explicit bits; byte 255 (NaN in torch's e8m0 conversion) decodes
to +inf here — unreachable for valid checkpoints.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

_BLOCK_R = 32
_BLOCK_K = 256


if _HAS_TRITON:

    @triton.jit
    def _fp4_group_dequant_kernel(
        w_ptr,  # [R, K/2] int8, packed nibbles (low -> even cols)
        s_ptr,  # [R, K/32] uint8, UE8M0 exponent bytes
        lut_ptr,  # [16] fp32 e2m1 values
        out_ptr,  # [R, K] bf16
        R,
        K,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        rmask = rows < R
        kmask = cols < K
        mask = rmask[:, None] & kmask[None, :]

        half_k = K // 2
        byte = tl.load(
            w_ptr + rows[:, None] * half_k + (cols // 2)[None, :],
            mask=mask,
            other=0,
        ).to(tl.int32)
        is_high = (cols % 2) == 1
        nibble = tl.where(is_high[None, :], (byte >> 4) & 0x0F, byte & 0x0F)
        value = tl.load(lut_ptr + nibble, mask=mask, other=0.0)

        scale_byte = tl.load(
            s_ptr + rows[:, None] * (K // 32) + (cols // 32)[None, :],
            mask=mask,
            other=0,
        ).to(tl.int32)
        # 2^(x-127) by IEEE exponent-field construction; x==0 is the fp32
        # denormal 2^-127 (0x00400000).
        scale_bits = tl.where(scale_byte > 0, scale_byte << 23, 0x00400000)
        scale = scale_bits.to(tl.float32, bitcast=True)

        tl.store(out_ptr + rows[:, None] * K + cols[None, :], (value * scale).to(tl.bfloat16), mask=mask)


def fp4_group_dequant_available(weight: torch.Tensor) -> bool:
    return (
        _HAS_TRITON
        and weight.is_cuda
        and weight.dtype == torch.int8
        and weight.is_contiguous()
    )


def dequant_fp4_group_to_bf16(
    weight_int8: torch.Tensor, scale, out: torch.Tensor = None
) -> torch.Tensor:
    """Exact bf16 dequant of packed FP4 rows in one kernel launch.

    Falls back to the reference unpack when Triton or the layout is
    unavailable, so callers can use it unconditionally.
    """
    rows, half_k = weight_int8.shape
    if not fp4_group_dequant_available(weight_int8):
        from rtp_llm.models_py.modules.dsv41.qlinear import (
            _fp4_unpack_rows_to_bf16,
        )

        return _fp4_unpack_rows_to_bf16(weight_int8, scale)
    from rtp_llm.models_py.modules.dsv41.qlinear import _fp4_lut

    k = half_k * 2
    if out is None:
        out = torch.empty(
            (rows, k), dtype=torch.bfloat16, device=weight_int8.device
        )
    scale_u8 = (
        scale.view(torch.uint8)
        if scale.dtype != torch.uint8
        else scale
    )
    if not scale_u8.is_contiguous():
        scale_u8 = scale_u8.contiguous()
    lut = _fp4_lut(weight_int8.device)
    grid = (
        triton.cdiv(rows, _BLOCK_R),
        triton.cdiv(k, _BLOCK_K),
    )
    _fp4_group_dequant_kernel[grid](
        weight_int8,
        scale_u8,
        lut,
        out,
        rows,
        k,
        BLOCK_R=_BLOCK_R,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
    )
    return out

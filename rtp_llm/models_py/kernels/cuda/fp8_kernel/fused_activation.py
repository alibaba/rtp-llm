"""Fused pointwise producers for grouped E4M3 activations and UE8M0 scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_group128(values, out, scales, row, rows, width: tl.constexpr, block: tl.constexpr):
    columns = tl.arange(0, block)
    # The existing quantizer observes the BF16 result of the pointwise op.
    values = values.to(tl.bfloat16).to(tl.float32)
    groups = tl.reshape(tl.where(columns < width, values, 0.0), (block // 128, 128))
    amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 1.0e-4)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1.0e-10))))
    quantized = tl.reshape(
        tl.minimum(tl.maximum(groups / scale[:, None], -448.0), 448.0), (block,)
    )
    tl.store(out + row * width + columns, quantized, columns < width)

    exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 255
    exponent = tl.where(tl.arange(0, block // 128) < width // 128, exponent, 127)
    packed = tl.sum(
        tl.reshape(exponent, (block // 512, 4)) << (tl.arange(0, 4)[None, :] * 8),
        axis=1,
    )
    packs = tl.arange(0, block // 512)
    aligned_rows = tl.cdiv(rows, 4) * 4
    tl.store(scales + packs * aligned_rows + row, packed, packs < triton.cdiv(width, 512))
    if row == 0:
        padding = rows + tl.arange(0, 4)
        tl.store(
            scales + packs[:, None] * aligned_rows + padding[None, :],
            0x7F7F7F7F,
            (packs[:, None] < triton.cdiv(width, 512))
            & (padding[None, :] < aligned_rows),
        )


@triton.jit(do_not_specialize=["rows"])
def _sigmoid_mul_group128_kernel(
    x,
    gate,
    out,
    scales,
    rows,
    width: tl.constexpr,
    x_row_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    a = tl.load(x + row * x_row_stride + columns, columns < width, other=0.0).to(
        tl.float32
    )
    b = tl.load(
        gate + row * gate_row_stride + columns, columns < width, other=0.0
    ).to(tl.float32)
    # Match the current compiled BF16 gate path: sigmoid and multiply in FP32,
    # then one BF16 rounding before group128 quantization.
    _store_group128(a * tl.sigmoid(b), out, scales, row, rows, width, block)


def sigmoid_mul_per_token_group_quant_fp8(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return grouped E4M3 values and packed UE8M0 scales for ``x*sigmoid(gate)``.

    The scale layout matches ``sgl_per_token_group_quant_fp8`` with group 128,
    column-major TMA-aligned scales, and UE8M0 packing.
    """
    if x.ndim != 2 or gate.shape != x.shape:
        raise ValueError("FP8 sigmoid-mul requires matching [rows, width] inputs")
    if x.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise ValueError("FP8 sigmoid-mul requires BF16 inputs")
    if not x.is_cuda or gate.device != x.device:
        raise ValueError("FP8 sigmoid-mul inputs must share a CUDA device")
    if x.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("FP8 sigmoid-mul requires contiguous feature dimensions")
    rows, width = x.shape
    if width == 0 or width % 128:
        raise ValueError("FP8 sigmoid-mul width must be divisible by 128")

    out = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    scale_wire = torch.empty(
        ((width + 511) // 512, (rows + 3) // 4 * 4),
        device=x.device,
        dtype=torch.int32,
    )
    if rows:
        _sigmoid_mul_group128_kernel[(rows,)](
            x,
            gate,
            out,
            scale_wire,
            rows,
            width,
            x.stride(0),
            gate.stride(0),
            max(512, triton.next_power_of_2(width)),
        )
    return out, scale_wire.T[:rows, :]

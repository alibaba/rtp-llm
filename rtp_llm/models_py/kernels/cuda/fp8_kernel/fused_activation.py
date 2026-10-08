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
    # Match the existing quantizer's legacy/v2 padding at its 4M-element switch.
    pad_exponent = tl.where(rows * width >= 4 * 1024 * 1024, 0, 0x3F)
    exponent = tl.where(
        tl.arange(0, block // 128) < width // 128, exponent, pad_exponent
    )
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
    # Match BF16 eager gating: sigmoid rounds before the multiply, then the
    # grouped quantizer observes the BF16 product.
    sigmoid_bf16 = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)
    _store_group128(a * sigmoid_bf16, out, scales, row, rows, width, block)


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


@triton.jit(do_not_specialize=["rows"])
def _rmsnorm_sigmoid_gate_group128_kernel(
    x,
    gate,
    weight,
    out,
    scales,
    rows,
    heads: tl.constexpr,
    x_row_stride: tl.constexpr,
    x_head_stride: tl.constexpr,
    x_dim_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    gate_head_stride: tl.constexpr,
    gate_dim_stride: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    head = columns // 128
    dim = columns % 128
    valid = columns < heads * 128
    a = tl.load(
        x + row * x_row_stride + head * x_head_stride + dim * x_dim_stride,
        valid,
        other=0.0,
    ).to(tl.float32)
    b = tl.load(
        gate
        + row * gate_row_stride
        + head * gate_head_stride
        + dim * gate_dim_stride,
        valid,
        other=0.0,
    ).to(tl.float32)
    gamma = tl.load(weight + dim).to(tl.float32)
    groups = tl.reshape(a, (block // 128, 128))
    variance = tl.sum(groups * groups, axis=1) / 128.0
    # Preserve the native gated-norm kernel's BF16 rounding at quantization input.
    inverse = 1 / tl.sqrt(variance + eps)
    normalized = tl.reshape(groups * inverse[:, None], (block,))
    _store_group128(
        normalized * gamma * tl.sigmoid(b),
        out,
        scales,
        row,
        rows,
        heads * 128,
        block,
    )


def rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-head RMSNorm, sigmoid gate and group128 E4M3 production.

    Inputs are BF16 ``[tokens, heads, 128]``. Scales use the same packed UE8M0
    layout as ``sigmoid_mul_per_token_group_quant_fp8``.
    """
    if x.ndim != 3 or gate.shape != x.shape or x.shape[-1] != 128:
        raise ValueError("FP8 gated RMSNorm requires matching [tokens, heads, 128] inputs")
    if weight.shape != (128,):
        raise ValueError("FP8 gated RMSNorm requires a 128-element weight")
    if x.dtype != torch.bfloat16 or gate.dtype != x.dtype or weight.dtype != x.dtype:
        raise ValueError("FP8 gated RMSNorm requires BF16 inputs and weight")
    if not x.is_cuda or gate.device != x.device or weight.device != x.device:
        raise ValueError("FP8 gated RMSNorm tensors must share a CUDA device")
    rows, heads, _ = x.shape
    if heads <= 0:
        raise ValueError("FP8 gated RMSNorm requires at least one head")
    width = heads * 128
    out = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    scale_wire = torch.empty(
        ((width + 511) // 512, (rows + 3) // 4 * 4),
        device=x.device,
        dtype=torch.int32,
    )
    if rows:
        _rmsnorm_sigmoid_gate_group128_kernel[(rows,)](
            x,
            gate,
            weight,
            out,
            scale_wire,
            rows,
            heads,
            *x.stride(),
            *gate.stride(),
            eps,
            max(512, triton.next_power_of_2(width)),
        )
    return out, scale_wire.T[:rows, :]


@triton.jit(do_not_specialize=["rows"])
def _quantize_strided_group128_kernel(
    x,
    out,
    scales,
    rows,
    row_stride: tl.constexpr,
    block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    column = tl.arange(0, 128)
    values = tl.load(
        x + row[:, None] * row_stride + column[None, :],
        row[:, None] < rows,
        other=0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), 1.0e-4)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1.0e-10))))
    quantized = tl.minimum(tl.maximum(values / scale[:, None], -448.0), 448.0)
    tl.store(out + row[:, None] * 128 + column[None, :], quantized, row[:, None] < rows)

    exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 255
    packed = tl.where(row < rows, exponent | 0x7F7F7F00, 0x7F7F7F7F)
    tl.store(scales + row, packed, row < ((rows + 3) // 4) * 4)


def quantize_strided_group128_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a row-strided BF16 [M,128] view without a staging copy."""
    if x.ndim != 2 or x.shape[1] != 128 or x.stride(1) != 1 or x.stride(0) < 128:
        raise ValueError("FP8 strided group128 input must be row-strided [M,128]")
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("FP8 strided group128 input must be CUDA BF16")
    rows = x.shape[0]
    out = torch.empty((rows, 128), dtype=torch.float8_e4m3fn, device=x.device)
    scale_wire = torch.empty(
        (1, triton.cdiv(rows, 4) * 4), dtype=torch.int32, device=x.device
    ).T
    if rows:
        _quantize_strided_group128_kernel[(triton.cdiv(rows, 4),)](
            x, out, scale_wire, rows, x.stride(0), 4, num_warps=4
        )
    return out, scale_wire[:rows]

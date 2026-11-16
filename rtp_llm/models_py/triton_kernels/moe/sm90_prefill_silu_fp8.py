"""SiLU and grouped FP8 with the native BF16 quantizer's rounding contract."""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_native_fp8(
    gate_up,
    quantized,
    scales,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group = tl.program_id(1)
    cols = group * 128 + tl.arange(0, 128)
    offsets = rows[:, None] * (2 * WIDTH) + cols[None, :]
    value = tl.load(gate_up + offsets, rows[:, None] < ROWS, other=0).to(tl.float32)
    gate = tl.load(gate_up + offsets + WIDTH, rows[:, None] < ROWS, other=0).to(
        tl.float32
    )
    # Match silu_and_mul's BF16 store before the native quantizer reads it.
    activated = ((gate * tl.sigmoid(gate)) * value).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(activated), axis=1), 1.0e-4)
    scale = tl.div_rn(maximum, 448.0)
    # The native quantizer rounds the inverse and multiplication to BF16.
    inverse = tl.div_rn(1.0, scale).to(tl.bfloat16).to(tl.float32)
    scaled = (activated * inverse[:, None]).to(tl.bfloat16)
    tl.store(
        quantized + rows[:, None] * WIDTH + cols[None, :],
        scaled.to(tl.float8e4nv),
        rows[:, None] < ROWS,
    )
    tl.store(scales + group * ROWS + rows, scale, rows < ROWS)


def silu_mul_native_fp8(gate_up, *, output=None, output_scale=None, block_m=16):
    """Produce FP8 values and MN-major FP32 scales for qualified SM90 MoE."""
    rows, doubled_width = gate_up.shape
    width = doubled_width // 2
    assert gate_up.dtype == torch.bfloat16 and gate_up.is_contiguous()
    assert doubled_width == 1280 and rows % 128 == 0
    if output is None:
        output = torch.empty(
            (rows, width), device=gate_up.device, dtype=torch.float8_e4m3fn
        )
    if output_scale is None:
        output_scale = torch.empty_strided(
            (rows, width // 128), (1, rows), device=gate_up.device, dtype=torch.float32
        )
    assert output.shape == (rows, width) and output.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn and output.device == gate_up.device
    assert output_scale.shape == (rows, width // 128)
    assert output_scale.stride() == (1, rows)
    assert output_scale.dtype == torch.float32 and output_scale.device == gate_up.device
    _silu_mul_native_fp8[(triton.cdiv(rows, block_m), width // 128)](
        gate_up,
        output,
        output_scale,
        rows,
        width,
        block_m,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, output_scale

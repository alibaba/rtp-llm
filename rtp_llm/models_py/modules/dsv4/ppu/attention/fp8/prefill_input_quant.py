"""UE8M0 activation quantization with FP32 row scales for PPU Prefill GEMM."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["group_count"])
def _quantize(
    x_ptr,
    payload_ptr,
    scale_ptr,
    group_count,
    groups_per_cta: tl.constexpr,
):
    groups = tl.program_id(0) * groups_per_cta + tl.arange(0, groups_per_cta)
    offsets = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    valid = groups < group_count
    x = tl.load(x_ptr + offsets, valid[:, None], other=0).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-4)

    # Round the positive normal FP32 scale up to a power of two. The
    # reciprocal of a power of two negates its unbiased IEEE exponent.
    raw = (maximum * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    rounded = (raw + 0x007FFFFF) & 0x7F800000
    scale = rounded.to(tl.float32, bitcast=True)
    inverse = (0x7F000000 - rounded).to(tl.float32, bitcast=True)
    payload = tl.clamp(x * inverse[:, None], -448.0, 448.0)
    tl.store(
        payload_ptr + offsets, payload.to(payload_ptr.dtype.element_ty), valid[:, None]
    )
    tl.store(scale_ptr + groups, scale, valid)


def quantize_prefill_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = x.shape
    payload = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, width // 128), device=x.device, dtype=torch.float32)
    if rows:
        groups_per_cta = 16
        _quantize[(triton.cdiv(scales.numel(), groups_per_cta),)](
            x,
            payload,
            scales,
            scales.numel(),
            groups_per_cta,
            num_warps=4,
        )
    return payload, scales


@triton.jit(do_not_specialize=["rows"])
def _quantize_dequant_kv_nope(
    x_ptr,
    rows,
    width: tl.constexpr,
    nope_width: tl.constexpr,
    group_size: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = group * group_size + tl.arange(0, group_size)
    valid = (row < rows) & (offsets < nope_width)
    x = tl.load(x_ptr + row * width + offsets, valid, other=0).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-4)

    raw = (maximum * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    rounded = (raw + 0x007FFFFF) & 0x7F800000
    scale = rounded.to(tl.float32, bitcast=True)
    inverse = (0x7F000000 - rounded).to(tl.float32, bitcast=True)
    quantized = tl.clamp(x * inverse, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(x_ptr + row * width + offsets, quantized.to(tl.float32) * scale, valid)


def quantize_dequant_prefill_kv_nope_(
    x: torch.Tensor, rope_head_dim: int = 64
) -> torch.Tensor:
    """Apply the official block64 UE8M0 FP8 QAT boundary to KV NoPE in-place."""
    if x.dtype != torch.bfloat16 or x.shape[-1] != 512 or not x.is_contiguous():
        raise ValueError("prefill KV must be contiguous BF16 [..., 512]")
    nope_width = x.shape[-1] - rope_head_dim
    if rope_head_dim != 64 or nope_width % 64:
        raise ValueError("prefill KV requires 448 NoPE + 64 RoPE dimensions")
    rows = x.numel() // x.shape[-1]
    if rows:
        _quantize_dequant_kv_nope[(rows, nope_width // 64)](
            x, rows, x.shape[-1], nope_width, 64, num_warps=1
        )
    return x

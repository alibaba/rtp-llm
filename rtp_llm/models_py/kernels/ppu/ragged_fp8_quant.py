"""Quantize packed expert rows using GPU counts instead of padded capacity."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS", "SCALE_ROWS"])
def _ragged_silu_quant(
    x_ptr,
    counts_ptr,
    q_ptr,
    scales_ptr,
    ROWS,
    HIDDEN: tl.constexpr,
    SCALE_ROWS,
    EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    GROUPS: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    rows = tl.sum(tl.load(counts_ptr + experts, experts < EXPERTS, other=0), 0)
    rows = tl.minimum(rows, ROWS)
    total_groups = rows * (HIDDEN // 128)
    # A bounded persistent grid avoids launching one program for every unused
    # capacity tile. Both loop bounds and live-row masks come from GPU counts.
    for start in range(
        tl.program_id(0) * GROUPS, total_groups, tl.num_programs(0) * GROUPS
    ):
        groups = start + tl.arange(0, GROUPS)
        row = groups // (HIDDEN // 128)
        column = groups % (HIDDEN // 128)
        cols = column[:, None] * 128 + tl.arange(0, 128)[None, :]
        mask = row[:, None] < rows
        base = row[:, None] * (2 * HIDDEN) + cols
        up = tl.load(x_ptr + base, mask, other=0).to(tl.float32)
        gate = tl.load(x_ptr + base + HIDDEN, mask, other=0).to(tl.float32)
        value = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(value), 1), 1e-10)
        scales = amax / 448.0
        quantized = tl.clamp(value * (448.0 / amax[:, None]), -448.0, 448.0).to(
            q_ptr.dtype.element_ty
        )
        tl.store(q_ptr + row[:, None] * HIDDEN + cols, quantized, mask)
        # Produce the column-major layout consumed by the SDK directly.
        tl.store(scales_ptr + row + column * SCALE_ROWS, scales, row < rows)


def ragged_silu_mul_fp8_quantize_128(
    x: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize only sum(counts) live [up,gate] rows; leave spare rows unused.

    Counts and loop bounds stay on the device, including graph replay. The
    returned scale view is column-major with a 16-byte-aligned row capacity,
    so grouped GEMM does not need a separate scale-layout conversion.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("ragged SiLU FP8 quantization expects contiguous BF16 [M,2K]")
    if (
        counts.ndim != 1
        or counts.dtype != torch.int32
        or not counts.is_contiguous()
        or counts.device != x.device
        or counts.numel() == 0
    ):
        raise ValueError("ragged SiLU FP8 counts must be nonempty device INT32 [E]")
    rows, width = x.shape
    if width == 0 or width % 256:
        raise ValueError("SiLU activation width must be a positive multiple of 256")
    hidden = width // 2
    q = torch.empty((rows, hidden), device=x.device, dtype=torch.float8_e4m3fn)
    scale_rows = triton.cdiv(rows, 4) * 4
    scales = torch.empty(
        (hidden // 128, scale_rows), device=x.device, dtype=torch.float32
    ).T[:rows]
    if rows:
        _ragged_silu_quant[(min(triton.cdiv(rows * (hidden // 128), 4), 256),)](
            x,
            counts,
            q,
            scales,
            rows,
            hidden,
            scale_rows,
            counts.numel(),
            triton.next_power_of_2(counts.numel()),
            4,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, scales

"""Experimental GeGLU with the eager intermediate BF16 GELU result retained."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _geglu_kernel(
    GATE,
    UP,
    OUT,
    G_STRIDE: tl.constexpr,
    U_STRIDE: tl.constexpr,
    ROWS,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = index // WIDTH
    col = index % WIDTH
    g = tl.load(GATE + row * G_STRIDE + col, mask=row < ROWS, other=0).to(tl.float32)
    u = tl.load(UP + row * U_STRIDE + col, mask=row < ROWS, other=0).to(tl.float32)
    cube = (g * g) * g
    inner = 0.7978845608028654 * (g + 0.044715 * cube)
    activated = (0.5 * g) * (1.0 + libdevice.tanh(inner))
    activated = activated.to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + index, activated * u, mask=row < ROWS)


def geglu(gate, up, contraction=True, output=None):
    if (
        not gate.is_cuda
        or gate.dtype != torch.bfloat16
        or gate.dim() != 2
        or up.device != gate.device
        or up.dtype != gate.dtype
        or up.shape != gate.shape
        or gate.stride(1) != 1
        or up.stride(1) != 1
        or gate.numel() == 0
        or gate.stride(0) <= 0
        or up.stride(0) <= 0
    ):
        return None
    if output is None:
        output = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    elif (
        output.device != gate.device or output.dtype != gate.dtype
        or output.shape != gate.shape or not output.is_contiguous()
    ):
        return None
    _geglu_kernel[(triton.cdiv(gate.numel(), 1024),)](
        gate,
        up,
        output,
        gate.stride(0),
        up.stride(0),
        gate.shape[0],
        gate.shape[1],
        1024,
        num_warps=4,
        enable_fp_fusion=contraction,
    )
    return output

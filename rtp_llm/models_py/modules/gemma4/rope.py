"""Split-half Gemma4 RoPE with the eager BF16 multiplication boundaries."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from rtp_llm.models_py.modules.gemma4.elementwise import is_supported


@triton.jit
def _rope_kernel(
    X,
    POSITIONS,
    FREQS,
    Y,
    TOKEN_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    offset = row // HEADS * TOKEN_STRIDE + row % HEADS * HEAD_STRIDE
    half = D // 2
    pair = tl.where(col < half, col + half, col - half)
    x = tl.load(X + offset + col, mask=col < D, other=0).to(tl.float32)
    other = tl.load(X + offset + pair, mask=col < D, other=0).to(tl.float32)
    position = tl.load(POSITIONS + row // HEADS).to(tl.float32)
    freq = tl.load(FREQS + col % half, mask=col < D, other=0)
    angle = position * freq
    # CUDA libdevice preserves large-angle argument reduction, unlike the
    # fast approximate Triton cosine/sine instructions.
    cosine = libdevice.cos(angle).to(tl.bfloat16).to(tl.float32)
    sine = libdevice.sin(angle).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(col < half, -other, other)
    first = (x * cosine).to(tl.bfloat16).to(tl.float32)
    second = (rotated * sine).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, first + second, mask=col < D)


def rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor):
    if (
        not is_supported(x)
        or x.dim() != 3
        or x.shape[-1] % 2
        or positions.device != x.device
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.numel() != x.shape[0]
        or not positions.is_contiguous()
        or inv_freq.device != x.device
        or inv_freq.dtype != torch.float32
        or inv_freq.shape != (x.shape[-1] // 2,)
        or not inv_freq.is_contiguous()
    ):
        return None
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    heads = x.shape[1] if x.dim() == 3 else 1
    _rope_kernel[(x.numel() // x.shape[-1],)](
        x,
        positions,
        inv_freq,
        output,
        x.stride(0),
        x.stride(1) if x.dim() == 3 else 0,
        heads,
        x.shape[-1],
        triton.next_power_of_2(x.shape[-1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output

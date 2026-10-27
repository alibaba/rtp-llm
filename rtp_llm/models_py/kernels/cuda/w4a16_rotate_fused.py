"""Fused prologues for the SM120 W4A16 dense FFN (decode, small M).

rotate(x, signs) = Hadamard128(x * signs) * 128^-0.5 is two launches upstream
(elementwise mul, fast_hadamard_transform) and the down projection additionally
runs silu_and_mul before it. These Triton kernels do
  * sign-flip + Hadamard-128                       (up-proj input)
  * silu(gate) * up -> bf16 + sign-flip + Hadamard (down-proj input)
in a single launch each. The Hadamard matrix is generated in-kernel from the
Sylvester construction H[i, j] = (-1) ** popcount(i & j).
"""

import torch
import triton
import triton.language as tl

_BLOCK = 128


@triton.jit
def _rotate_kernel(
    x_ptr,
    signs_ptr,
    out_ptr,
    stride_x,
    stride_o,
    D,
    scale,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    x_row = x_ptr + row * stride_x
    if ACT:
        g = tl.load(x_row + offs).to(tl.float32)
        u = tl.load(x_row + D + offs).to(tl.float32)
        v = (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16).to(tl.float32)
    else:
        v = tl.load(x_row + offs).to(tl.float32)
    v = v * tl.load(signs_ptr + offs).to(tl.float32)
    i = tl.arange(0, BLOCK)[:, None]
    j = tl.arange(0, BLOCK)[None, :]
    p = i & j
    p = p ^ (p >> 4)
    p = p ^ (p >> 2)
    p = p ^ (p >> 1)
    h = 1.0 - 2.0 * (p & 1).to(tl.float32)
    y = tl.sum(v[:, None] * h, axis=0) * scale
    tl.store(out_ptr + row * stride_o + offs, y.to(out_ptr.dtype.element_ty))


def _launch(x, signs, D, act):
    assert x.dim() == 2 and x.stride(1) == 1 and D % _BLOCK == 0
    assert D < 1024 * 1024 and signs.numel() == D
    T = x.shape[0]
    out = torch.empty((T, D), dtype=x.dtype, device=x.device)
    if T > 0:
        _rotate_kernel[(T, D // _BLOCK)](
            x,
            signs,
            out,
            x.stride(0),
            out.stride(0),
            D,
            _BLOCK**-0.5,
            ACT=act,
            BLOCK=_BLOCK,
            num_warps=4,
        )
    return out


def fused_rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """== w4a16_sm120.rotate(x, signs) for a 2-D bf16 x."""
    return _launch(x, signs, x.shape[1], False)


def fused_silu_mul_rotate(gate_up: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """== w4a16_sm120.rotate(silu_and_mul(gate_up), signs) for 2-D bf16 gate_up."""
    return _launch(gate_up, signs, gate_up.shape[1] // 2, True)

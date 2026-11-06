# SPDX-License-Identifier: Apache-2.0
"""Build the final sparse-MLA query layout without a real-head copy."""
import torch
import triton
import triton.language as tl


@triton.jit
def _suffix_and_head_padding(Q, O, QS0: tl.constexpr, QS1: tl.constexpr,
                             QS2: tl.constexpr,
                             HEADS: tl.constexpr, KHEADS: tl.constexpr,
                             NOPE: tl.constexpr, LATENT: tl.constexpr,
                             SUFFIX: tl.constexpr, HEAD_BLOCK: tl.constexpr,
                             BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    col = tl.arange(0, BLOCK)
    width = LATENT + SUFFIX
    dst = O + (row * KHEADS + head[:, None]) * width
    real = (head[:, None] < HEADS) & (col[None, :] < SUFFIX)
    value = tl.load(Q + row * QS0 + head[:, None] * QS1 + (NOPE + col[None, :]) * QS2,
                    real, other=0)
    tl.store(dst + LATENT + col[None, :], value, real)
    pad = (head[:, None] >= HEADS) & (head[:, None] < KHEADS) & (col[None, :] < width)
    tl.store(dst + col[None, :], 0, pad)


def absorbed_query(q, weight, latent_dim, kernel_heads):
    """BF16 K-absorb once; fill suffix and only the kernel's extra heads.

    Q and O have independent allocations: Q-NoPE must remain intact while the
    strided BMM writes the first latent_dim columns of each real head.
    """
    rows, heads, qdim = q.shape
    if (q.dtype != torch.bfloat16 or weight.dtype != q.dtype
        or q.device != weight.device or not q.is_cuda or q.stride(-1) != 1
        or weight.ndim != 3 or weight.shape[0] != heads
        or weight.shape[2] != latent_dim or kernel_heads < heads):
        raise ValueError('Incompatible BF16 sparse absorbed-query layout')
    nope = weight.shape[1]
    suffix = qdim - nope
    if suffix < 0:
        raise ValueError('Sparse suffix dimension must be nonnegative')
    output = torch.empty((rows, kernel_heads, latent_dim + suffix),
                         device=q.device, dtype=q.dtype)
    if rows:
        torch.bmm(q[..., :nope].transpose(0, 1), weight,
                  out=output[:, :heads, :latent_dim].transpose(0, 1))
        _suffix_and_head_padding[(rows, triton.cdiv(kernel_heads, 8))](
            q, output, *q.stride(), heads, kernel_heads, nope, latent_dim,
            suffix, 8, triton.next_power_of_2(latent_dim + suffix), num_warps=4)
    return output

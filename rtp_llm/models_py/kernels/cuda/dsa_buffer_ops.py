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


@triton.jit(do_not_specialize=["ROWS"])
def _absorb_query_with_suffix(
    Q, W, O, ROWS,
    QS0: tl.constexpr, QS1: tl.constexpr,
    WS0: tl.constexpr, WS1: tl.constexpr, WS2: tl.constexpr,
    HEADS: tl.constexpr, KHEADS: tl.constexpr,
    NOPE: tl.constexpr, LATENT: tl.constexpr, SUFFIX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    tile = tl.program_id(1)
    head = tl.program_id(2)
    col = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_row = row[:, None] < ROWS
    query = Q + row[:, None] * QS0 + head * QS1
    output = O + (row[:, None] * KHEADS + head) * (LATENT + SUFFIX)

    if head < HEADS:
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in tl.static_range(0, NOPE, BLOCK_K):
            k = start + tl.arange(0, BLOCK_K)
            q = tl.load(query + k[None, :], valid_row, other=0)
            w = tl.load(W + head * WS0 + k[:, None] * WS1 + col[None, :] * WS2)
            acc = tl.dot(q, w, acc)
        tl.store(output + col[None, :], acc, valid_row)
    else:
        tl.store(output + col[None, :], 0, valid_row)

    # Only one tile per head owns the suffix columns.
    if tile == 0:
        suffix = tl.arange(0, SUFFIX)[None, :]
        value = tl.load(query + NOPE + suffix, valid_row & (head < HEADS), other=0)
        tl.store(output + LATENT + suffix, value, valid_row)


def absorbed_query(q, weight, latent_dim, kernel_heads, *, fuse_suffix=False):
    """Build the BF16 K-absorbed query, its suffix and any extra kernel heads."""
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
    if not rows:
        return output
    if fuse_suffix and (nope, latent_dim, suffix) == (128, 512, 64):
        # Prefill Query-CP uses the fixed K3.1 geometry and one launch.
        block_m, block_n, block_k = 64, 128, 64
        grid = (triton.cdiv(rows, block_m), latent_dim // block_n, kernel_heads)
        _absorb_query_with_suffix[grid](
            q, weight, output, rows, q.stride(0), q.stride(1), *weight.stride(),
            heads, kernel_heads, nope, latent_dim, suffix, block_m, block_n, block_k,
            num_warps=4,
        )
    else:
        torch.bmm(q[..., :nope].transpose(0, 1), weight,
                  out=output[:, :heads, :latent_dim].transpose(0, 1))
        _suffix_and_head_padding[(rows, triton.cdiv(kernel_heads, 8))](
            q, output, *q.stride(), heads, kernel_heads, nope, latent_dim,
            suffix, 8, triton.next_power_of_2(latent_dim + suffix), num_warps=4)
    return output

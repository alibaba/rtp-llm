"""FP32 normalization of the existing DeepGEMM mHC split-K partials."""

import triton
import triton.language as tl


@triton.jit
def hc_prenorm_reduce_kernel(
    Mul,
    Squares,
    Mixes,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SPLITS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    splits = tl.arange(0, BLOCK_S)
    mixes = tl.arange(0, 32)
    products = tl.load(
        Mul + splits[:, None] * ROWS * 24 + row * 24 + mixes[None, :],
        (splits[:, None] < SPLITS) & (mixes[None, :] < 24),
        0,
    )
    squares = tl.load(Squares + splits * ROWS + row, splits < SPLITS, 0)
    rstd = tl.rsqrt(tl.sum(squares, 0) / HIDDEN + NORM_EPS)
    tl.store(Mixes + row * 24 + mixes, tl.sum(products, 0) * rstd, mixes < 24)

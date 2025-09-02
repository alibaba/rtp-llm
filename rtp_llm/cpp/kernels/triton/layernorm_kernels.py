import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    Y1,
    W,  # pointer to the weights
    BETA,  # pointer to the beta
    RESIDUAL,  # pointer to the residual
    BIAS,
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_res_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    HAS_Y1,
    HAS_RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row

    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(BIAS + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += bias

    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual

    # Var[x] = E[(x - E[x])²]
    # mean = tl.sum(x, axis=0) / N
    # xbar = tl.where(cols < N, x - mean, 0.0)
    # var = tl.sum(xbar * xbar, axis=0) / N

    # Var[x] = E[x²] - E[x]²
    mean = tl.sum(x, axis=0) / N
    mean_of_x_square = tl.sum(x * x, axis=0) / N
    square_of_mean = mean * mean
    var = tl.where(cols < N, mean_of_x_square - square_of_mean, 0.0)

    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd

    b = tl.load(BETA + cols, mask=mask).to(tl.float32)
    y = x_hat * w + b

    # Write output
    tl.store(Y + cols, y, mask=mask)

    if HAS_Y1:
        Y1 += row * stride_y_row
        tl.store(Y1 + cols, y, mask=mask)

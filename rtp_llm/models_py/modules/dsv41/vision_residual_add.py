"""Out-of-place fused residual addition and RMSNorm for DeepSeek vision.

The BF16 residual-add rounding boundary matches the existing ``x + residual``
path. RMS reduction and weight multiplication use FP32 inside the kernel. Both
returned tensors own fresh storage; neither input is mutated.
"""

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton = None


if triton is not None:

    @triton.jit
    def _vision_residual_rmsnorm_kernel(
        X,
        R,
        W,
        S,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        EPS: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
    ):
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, COLS)
        offsets = rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        x = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)
        residual = tl.load(R + offsets, mask=mask, other=0).to(tl.float32)
        # Match the old two-op path: BF16 add is rounded before RMS reduction.
        summed = (x + residual).to(tl.bfloat16)
        value = summed.to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(value * value, axis=1) / N + EPS)
        weight = tl.load(W + cols, mask=cols < N, other=0).to(tl.float32)
        normalized = value * inv_rms[:, None] * weight[None, :]
        tl.store(S + offsets, summed, mask=mask)
        tl.store(Y + offsets, normalized.to(tl.bfloat16), mask=mask)


def apply_vision_residual_rmsnorm(x, residual, weight, eps):
    """Return ``(sum, norm)`` or ``None`` for unsupported inputs."""
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(residual, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and not torch.is_grad_enabled()
        and x.layout == residual.layout == weight.layout == torch.strided
        and x.is_cuda
        and x.device == residual.device == weight.device
        and x.dtype == residual.dtype == weight.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape == residual.shape
        and x.shape[1] == 1024
        and weight.shape == (1024,)
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and torch.cuda.get_device_capability(x.device)[0] == 10
    ):
        return None
    if triton is None:
        return None
    summed = torch.empty_like(x)
    normalized = torch.empty_like(x)
    # FlashInfer's H=1024 RMSNorm uses one 32-thread warp per row and four
    # rows per 128-thread block. Matching that row grouping reduces reduction
    # order drift through the 32-layer vision stack.
    rows = 4
    _vision_residual_rmsnorm_kernel[(triton.cdiv(x.shape[0], rows),)](
        x,
        residual,
        weight,
        summed,
        normalized,
        x.shape[0],
        x.shape[1],
        eps,
        rows,
        triton.next_power_of_2(x.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return summed, normalized

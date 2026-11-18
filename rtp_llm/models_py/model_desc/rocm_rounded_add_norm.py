"""BF16-rounded residual add + RTZ RMSNorm for opt-in ROCm Decode fusion.

Call only after the Qwen3 compatibility checks and outside grad mode. Outputs
are forward-owned (or owned by the native graph memory pool), never shared
between live requests. The reference AITER output conversion is runtime-specific.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(X, R, W, Y, S, H: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * H + col, col < H, 0).to(tl.float32)
    r = tl.load(R + row * H + col, col < H, 0).to(tl.float32)
    # Preserve the standalone torch.add BF16/FP16 rounding boundary.
    rounded = (x + r).to(S.dtype.element_ty)
    v = rounded.to(tl.float32)
    inv = tl.rsqrt(tl.sum(v * v, axis=0) / H + EPS)
    w = tl.load(W + col, col < H, 0).to(tl.float32)
    tl.store(S + row * H + col, rounded, col < H)
    # Inspected module_rmsnorm_quant converts FP32 output to BF16 with RTZ.
    # This is runtime-specific, not a generally interchangeable RMSNorm.
    result = (v * inv * w).to(Y.dtype.element_ty, fp_downcast_rounding="rtz")
    tl.store(Y + row * H + col, result, col < H)


def rounded_add_norm(x, residual, weight, eps):
    output = torch.empty_like(x)
    added = torch.empty_like(x)
    _kernel[(x.shape[0],)](
        x,
        residual,
        weight,
        output,
        added,
        x.shape[1],
        eps,
        triton.next_power_of_2(x.shape[1]),
        num_warps=4,
    )
    return output, added

"""Small-batch HC projections with FP8 weights and BF16 activations."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_hc_gemv_kernel(
    x,
    weight,
    scale,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Adjacent programs reuse the same weight rows across batch elements.
    program = tl.program_id(0)
    row = program % M
    columns = (program // M) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    weight_bytes = weight.to(tl.pointer_type(tl.uint8))
    accumulator = tl.full((BN,), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + offsets
        values = tl.load(x + row * K + k, k < K, other=0).to(tl.float32)
        weights = (
            tl.load(
                weight_bytes + columns[:, None] * K + k[None, :],
                (columns[:, None] < N) & (k[None, :] < K),
                other=0,
            )
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
        accumulator += tl.sum(weights * values[None, :], axis=1)
    tl.store(output + row * N + columns, accumulator * tl.load(scale), columns < N)


def is_supported(x, weight, scale):
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.ndim == weight.ndim == 2
        and 1 <= x.shape[0] <= 8
        and (weight.shape[0], weight.shape[1]) in ((320, 10240), (10240, 320))
        and x.shape[1] == weight.shape[1]
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.float8_e4m3fn
        and scale.dtype == torch.float32
        and scale.numel() == 1
        and x.device == weight.device == scale.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and scale.is_contiguous()
        and torch.cuda.get_device_capability(x.device)[0] == 12
    )


def fp8_hc_gemv(x, weight, scale):
    if not is_supported(x, weight, scale):
        raise ValueError("unsupported FP8 HC GEMV layout")
    m, k = x.shape
    n = weight.shape[0]
    output = x.new_empty((m, n))
    _fp8_hc_gemv_kernel[(triton.cdiv(n, 4) * m,)](
        x, weight, scale, output, M=m, N=n, K=k, BN=4, BK=512, num_warps=4
    )
    return output

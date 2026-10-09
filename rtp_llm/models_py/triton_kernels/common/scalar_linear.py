"""Batch-invariant BF16 scalar projection for shared-expert gates."""

from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.prefill_input_log import trace_triton


@triton.jit
def _scalar_linear(
    X,
    W,
    BIAS,
    Y,
    K: tl.constexpr,
    XS: tl.constexpr,
    XK: tl.constexpr,
    WK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * XS + k * XK, k < K, 0).to(tl.float32)
    w = tl.load(W + k * WK, k < K, 0).to(tl.float32)
    # The reduction geometry is independent of the number of input rows.
    value = tl.sum(x * w, 0)
    if HAS_BIAS:
        value += tl.load(BIAS).to(tl.float32)
    tl.store(Y + row, value)


def is_supported(x, weight, bias=None):
    return (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim == 2
        and weight.ndim == 2
        and weight.shape[0] == 1
        and x.shape[1] == weight.shape[1]
        and 0 < x.shape[1] <= 8192
        and (
            bias is None
            or (bias.device == x.device and bias.dtype == x.dtype and bias.numel() == 1)
        )
    )


def maybe_bf16_scalar_linear(x, weight, bias=None) -> Optional[torch.Tensor]:
    """Return None for unsupported inputs so the caller retains F.linear."""
    if not is_supported(x, weight, bias):
        return None
    output = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
    if x.shape[0]:
        trace_triton(
            "scalar_linear:fixed_row_reduction",
            _scalar_linear,
            (x.shape[0],),
            x,
            weight,
            bias,
            output,
            K=x.shape[1],
            XS=x.stride(0),
            XK=x.stride(1),
            WK=weight.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK=triton.next_power_of_2(x.shape[1]),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output

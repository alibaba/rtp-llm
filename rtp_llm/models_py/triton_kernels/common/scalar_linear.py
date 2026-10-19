"""BF16 scalar projections and FP32 shared-expert decode projection."""

from threading import RLock
from typing import Optional

import torch
import torch.nn.functional as F
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


# Serialize this path's scoped changes to PyTorch's process-wide TF32 setting.
_fp32_scalar_lock = RLock()


def maybe_fp32_scalar_linear(x, weight) -> Optional[torch.Tensor]:
    """FP32 projection with one BF16 output store; weight is a cold FP32 cache."""
    if not (
        x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.float32
        and x.ndim == 2
        and x.shape[1] == 4096
        and weight.shape == (1, 4096)
        and x.is_contiguous()
        and weight.is_contiguous()
    ):
        return None
    with _fp32_scalar_lock:
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = False
            return F.linear(x.float(), weight, None).to(torch.bfloat16)
        finally:
            if allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True

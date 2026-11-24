"""Opt-in V4.1 RMSNorm helper used by the production vision RMSNorm.forward.

The caller selects this helper with V41_VISION_RMSNORM_FUSED (default off).
Keep numerical and performance acceptance separate from that integration.

Only the input cast/square and output multiplies/cast are fused. The contiguous
FP32 square buffer is passed to the original ATen mean, epsilon addition, and
rsqrt in their original order. No residual is read or modified by this module.

Unsupported inputs return None so the caller can use its original norm. Once
the supported path is selected, dependency and execution errors propagate.
"""

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    # CPU callers can import this candidate without installing Triton. Do not
    # hide a broken Triton installation or a missing transitive dependency.
    if error.name != "triton":
        raise
    triton = None


_VISION_WIDTH = 1024
_BLOCK_ELEMENTS = 1024


def _supported(x: torch.Tensor, weight: torch.Tensor, eps: float) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and not torch.is_grad_enabled()
        and x.layout == torch.strided
        and weight.layout == torch.strided
        and x.device.type == "cuda"
        and weight.device == x.device
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.float32
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == _VISION_WIDTH
        and x.is_contiguous()
        and x.stride() == (_VISION_WIDTH, 1)
        and weight.shape == (_VISION_WIDTH,)
        and weight.is_contiguous()
        and weight.stride() == (1,)
        and isinstance(eps, (float, int))
        and not isinstance(eps, bool)
        and math.isfinite(eps)
        and eps > 0
        and torch.cuda.get_device_capability(x.device)[0] == 10
    )


if triton is not None:

    @triton.jit
    def _cast_square_kernel(X, SQUARED, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
        tl.store(SQUARED + offsets, values * values, offsets < N)

    @triton.jit
    def _scale_weight_cast_kernel(
        X,
        INVERSE,
        WEIGHT,
        OUT,
        N: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
        inverse = tl.load(INVERSE + offsets // WIDTH, offsets < N, other=0)
        weight = tl.load(WEIGHT + offsets % WIDTH, offsets < N, other=0)
        # Preserve both FP32 multiply boundaries and their operand order.
        normalized = values * inverse
        weighted = weight * normalized
        tl.store(OUT + offsets, weighted, offsets < N)


def apply_vision_rmsnorm(
    x_bf16: torch.Tensor, weight_fp32: torch.Tensor, eps: float
) -> Optional[torch.Tensor]:
    """Return a fresh BF16 norm output, or None for an unsupported contract.

    Target: no-grad, CUDA SM10x, dense [rows, 1024] BF16 input and contiguous
    [1024] FP32 weight on the same device, with a positive finite scalar eps.
    Every scratch/output allocation belongs to this call; there is no cache,
    shared workspace, input mutation, or change to the caller's grad mode.
    """
    if not _supported(x_bf16, weight_fp32, eps):
        return None
    if triton is None:
        raise RuntimeError("The supported V4.1 RMSNorm path requires Triton")

    # Triton uses the current CUDA device; preserve callers whose tensor lives
    # on a device other than their thread's current one.
    with torch.cuda.device(x_bf16.device):
        squared = torch.empty_like(x_bf16, dtype=torch.float32)
        grid = (triton.cdiv(x_bf16.numel(), _BLOCK_ELEMENTS),)
        _cast_square_kernel[grid](
            x_bf16,
            squared,
            x_bf16.numel(),
            _BLOCK_ELEMENTS,
            enable_fp_fusion=False,
        )
        # Keep ATen's input shape, dtype, layout, reduction order, and scalar
        # epsilon handling. In particular, do not replace this with tl.sum.
        inverse = torch.rsqrt(squared.mean(-1, keepdim=True) + eps)
        output = torch.empty_like(x_bf16)
        _scale_weight_cast_kernel[grid](
            x_bf16,
            inverse,
            weight_fp32,
            output,
            x_bf16.numel(),
            _VISION_WIDTH,
            _BLOCK_ELEMENTS,
            enable_fp_fusion=False,
        )
    return output

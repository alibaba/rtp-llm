"""Graph-safe grouped RMSNorm for the Qwen4 hyperconnection stream."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_rms_norm_kernel(
    x_ptr,
    gamma_ptr,
    out_ptr,
    amax_ptr,
    WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    COLLECT_AMAX: tl.constexpr = False,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    columns = group * GROUP_SIZE + offsets
    values = tl.load(
        x_ptr + row * WIDTH + columns, mask=offsets < GROUP_SIZE, other=0
    ).to(tl.float32)
    gamma = tl.load(gamma_ptr + columns, mask=offsets < GROUP_SIZE, other=0).to(
        tl.float32
    )
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / GROUP_SIZE + EPS)
    result = values * inv_rms * (1.0 + gamma)
    tl.store(out_ptr + row * WIDTH + columns, result, mask=offsets < GROUP_SIZE)
    if COLLECT_AMAX:
        # The quantizer observes the materialized BF16 norm, not its FP32 value.
        absolute = tl.abs(result.to(tl.bfloat16).to(tl.float32))
        absolute = tl.where(
            (offsets < GROUP_SIZE) & (absolute == absolute), absolute, 0.0
        )
        tl.store(amax_ptr + row * (WIDTH // GROUP_SIZE) + group, tl.max(absolute, 0))


def is_supported(x: torch.Tensor, gamma: torch.Tensor, group_size: int) -> bool:
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and gamma.dtype == torch.bfloat16
        and gamma.device == x.device
        and x.is_contiguous()
        and gamma.is_contiguous()
        and x.dim() >= 2
        and gamma.dim() == 1
        and group_size > 0
        and group_size <= 4096
        and x.shape[-1] == gamma.numel()
        and x.shape[-1] % group_size == 0
    )


def grouped_rms_norm_triton(
    x: torch.Tensor, gamma: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    """Match the upstream FP32 gain and return BF16 in the input layout."""
    if not is_supported(x, gamma, group_size):
        raise ValueError("unsupported Qwen4 grouped RMSNorm layout")
    width = int(x.shape[-1])
    groups = width // group_size
    rows = x.numel() // width
    result = torch.empty_like(x)
    _grouped_rms_norm_kernel[(rows, groups)](
        x,
        gamma,
        result,
        result,
        WIDTH=width,
        GROUP_SIZE=group_size,
        EPS=eps,
        BLOCK=triton.next_power_of_2(group_size),
        num_warps=4,
    )
    return result


@triton.jit
def _norm_fp8_scale(AMAX, SCALE, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    maximum = tl.max(tl.load(AMAX + offsets, offsets < N, other=0.0), 0)
    # Match the CUDA per-tensor quantizer's constant reciprocal multiplication.
    tl.store(SCALE, maximum * (1.0 / 448.0))


def try_grouped_rms_norm_with_fp8_scale(x, gamma, group_size, eps):
    if not (
        is_supported(x, gamma, group_size)
        and torch.cuda.get_device_capability(x.device) == (12, 0)
        and x.shape[-1] == 10240
        and group_size == 2560
        and 256 <= x.numel() // 10240 <= 12288
        and not (x.requires_grad or gamma.requires_grad)
        and isinstance(eps, (float, int))
        and math.isfinite(eps)
        and eps >= 0
    ):
        return None
    rows = x.numel() // 10240
    result = torch.empty_like(x)
    partial = torch.empty(rows * 4, device=x.device, dtype=torch.float32)
    scale = torch.empty(1, device=x.device, dtype=torch.float32)
    _grouped_rms_norm_kernel[(rows, 4)](
        x,
        gamma,
        result,
        partial,
        WIDTH=10240,
        GROUP_SIZE=2560,
        EPS=eps,
        BLOCK=4096,
        COLLECT_AMAX=True,
        num_warps=4,
    )
    _norm_fp8_scale[(1,)](
        partial,
        scale,
        N=rows * 4,
        BLOCK=triton.next_power_of_2(rows * 4),
        num_warps=4 if rows < 4096 else 16,
    )
    return result, scale

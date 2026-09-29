"""Graph-safe grouped RMSNorm for the Qwen4 hyperconnection stream."""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_rms_norm_kernel(
    x_ptr,
    gamma_ptr,
    out_ptr,
    WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    columns = group * GROUP_SIZE + offsets
    values = tl.load(
        x_ptr + row * WIDTH + columns, mask=offsets < GROUP_SIZE, other=0
    ).to(tl.float32)
    gamma = tl.load(
        gamma_ptr + columns, mask=offsets < GROUP_SIZE, other=0
    ).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / GROUP_SIZE + EPS)
    result = values * inv_rms * (1.0 + gamma)
    tl.store(out_ptr + row * WIDTH + columns, result, mask=offsets < GROUP_SIZE)


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
        WIDTH=width,
        GROUP_SIZE=group_size,
        EPS=eps,
        BLOCK=triton.next_power_of_2(group_size),
        num_warps=4,
    )
    return result

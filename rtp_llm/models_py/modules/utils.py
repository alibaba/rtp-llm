from __future__ import annotations

import os
from typing import Iterable, Tuple

import torch

from rtp_llm.ops import DeviceType, get_device


# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# COPIED FROM DeepGEMM
def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# COPIED FROM DeepGEMM
def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


# COPIED FROM DeepGEMM
def per_token_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), sf


# COPIED FROM DeepGEMM
def per_channel_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(0) % 128 == 0
    m, n = x.shape
    x_view = x.view(-1, 128, n)
    x_amax = x_view.abs().float().amax(dim=1).view(-1, n).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(1))).to(torch.float8_e4m3fn).view(m, n), sf


# COPIED FROM DeepGEMM
def per_block_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_scales.dtype == torch.int:
        if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
            x_scales = x_scales << 23
        else:
            x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23

        x_scales = x_scales.view(dtype=torch.float)

    if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
        x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, x_fp8.size(1))
    else:
        x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)

    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


# COPIED FROM DeepGEMM
def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: Iterable[int], use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))  # type: ignore


def is_cuda():
    device_type = get_device().get_device_type()
    if device_type == DeviceType.Cuda:
        return True
    else:
        return False


def is_hip():
    device_type = get_device().get_device_type()
    if device_type == DeviceType.ROCm:
        return True
    else:
        return False

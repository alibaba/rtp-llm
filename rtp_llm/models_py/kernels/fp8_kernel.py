import functools
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import aiter.utility.dtypes as dtypes
import torch
import triton
from aiter.ops.quant import (
    dynamic_per_tensor_quant,
    dynamic_per_token_scaled_quant,
    static_per_tensor_quant,
)

logger = logging.getLogger(__name__)


def rocm_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
):
    """
    ROCm version of per-token group quantization to FP8.

    Args:
        x: Input tensor to quantize
        group_size: Size of quantization groups (typically 128)
        eps: Small epsilon value for numerical stability
        column_major_scales: Whether to use column-major scale layout
        scale_tma_aligned: Whether to align scales for TMA (Tensor Memory Accelerator)

    Returns:
        Tuple of (quantized_tensor, scales)
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    # Use aiter's FP8 dtype which supports different ROCm architectures
    # gfx942 (MI300X): torch.float8_e4m3fnuz
    # gfx950 (MI350X): torch.float8_e4m3fn
    fp8_dtype = dtypes.fp8

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)

    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    if x.shape[0] > 0:
        # Check group_size support - expanded from per_group_quant_hip logic
        assert group_size in [
            32,
            64,
            128,
        ], f"unsupported group size {group_size=}, only support [32, 64, 128]"

        # Simplified approach: directly use dynamic_per_token_scaled_quant for group_size=128
        if not group_size == 128:
            error_msg = f"FP8 quantization group_size {group_size} is not available."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            # Call the aiter quantization function directly - expanded from per_group_quant_hip
            dynamic_per_token_scaled_quant(x_q, x.view(-1, group_size), x_s)

    return x_q, x_s


def get_dtype_max(dtype):
    try:
        dtypeMax = torch.finfo(dtype).max
    except:
        dtypeMax = torch.iinfo(dtype).max
    return dtypeMax


def pertoken_quant(
    x,
    scale=None,
    x_scale=None,  # smooth_scale
    scale_dtype=dtypes.fp32,
    quant_dtype=dtypes.fp8,
    dtypeMax=None,
):
    x = x.to(dtypes.fp32)
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale

    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = scale
    if scale is None:
        # [m, 1]
        per_token_amax, _ = torch.max(
            input=torch.abs(hidden_states), dim=-1, keepdim=True
        )
        per_token_scale = per_token_amax / dtypeMax
        per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale


def rocm_per_token_quant_fp8(
    x: torch.Tensor,
    eps: float = 1e-10,
):
    """
    ROCm version of per-token quantization to FP8 (not per-token-block).
    This corresponds to InvokeROCmPTPCGemm logic in ROCmGemmOp.cc.

    Args:
        x: Input tensor to quantize
        eps: Small epsilon value for numerical stability

    Returns:
        Tuple of (quantized_tensor, scales)
    """
    assert x.is_contiguous(), "`x` is not contiguous"

    # Use aiter's FP8 dtype which supports different ROCm architectures
    fp8_dtype = dtypes.fp8

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)

    # Per-token scales: (M, 1)
    x_s = torch.empty(
        (x.shape[0], 1),
        device=x.device,
        dtype=torch.float32,
    )

    if x.shape[0] > 0:
        # Call the aiter quantization function for per-token quantization
        dynamic_per_token_scaled_quant(x_q, x, x_s)  # Use original tensor, not reshaped

    return x_q, x_s

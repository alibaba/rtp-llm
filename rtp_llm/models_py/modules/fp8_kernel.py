import functools
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

import rtp_llm.models_py.modules.utils as utils

if utils.is_cuda():
    from libth_transformer.rtp_llm_ops import (
        per_tensor_quant_fp8,
        per_token_group_quant_fp8,
        per_token_group_quant_int8,
        per_token_quant_fp8,
    )
else:
    logging.warning("can't import from rtp_llm_ops, only support cuda!")

logger = logging.getLogger(__name__)


def create_per_token_group_quant_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = align(x_s_mn, 4)
        aligned_k = align(x_s_k, 4)
        # TODO(FIXME): Fix cuda kernel and recover here to empty.
        return torch.zeros(
            (aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(0, 1)[:x_s_mn, :]
    elif column_major_scales:
        if scale_tma_aligned:
            # TODO extract "align" function
            # aligned to 4 * sizeof(float)
            aligned_size = (x_shape[-2] + 3) // 4 * 4
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x_shape[-2], :]
        else:
            return torch.empty(
                (x_shape[-1] // group_size,) + x_shape[:-1],
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        return torch.empty(
            x_shape[:-1] + (x_shape[-1] // group_size,),
            device=device,
            dtype=torch.float32,
        )


def sgl_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    # Define fp8 dtype and constants
    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_max = finfo.max
    fp8_min = finfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=x.shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    if x.shape[0] > 0:
        per_token_group_quant_fp8(
            x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
        )

    return x_q, x_s


def scaled_fp8_per_tensor_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim == 2

    shape: Union[Tuple[int, int], torch.Size] = input.shape
    out_dtype: torch.dtype = torch.float8_e4m3fn

    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        assert output.dtype == out_dtype

    if scale is None:
        # dynamic quant
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        per_tensor_quant_fp8(input, output, scale, False)
    else:
        # static quant
        assert scale.numel() == 1, f"{scale.shape}"
        per_tensor_quant_fp8(input, output, scale, True)

    return output, scale


def scaled_fp8_per_token_quant(
    input: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    if output is not None:
        assert output.dtype == torch.float8_e4m3fn
    else:
        output = torch.empty(
            input.shape, device=input.device, dtype=torch.float8_e4m3fn
        )

    per_token_quant_fp8(input, output, scale)
    scale = scale.reshape(-1, 1)
    return output, scale

import functools
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from rtp_llm.models_py.configs.get_best_config import get_cutlass_groupgemm_best_config
from rtp_llm.models_py.utils.arch import is_cuda
from rtp_llm.models_py.utils.math import align

if is_cuda():
    from rtp_llm.ops.compute_ops import (
        cutlass_moe_mm,
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


def cutlass_moe_mm_fp8_scaled(
    output,
    aq,
    w,
    aq_scale,
    w_scale,
    expert_offsets,
    problem_sizes,
    a_strides,
    b_strides,
    c_strides,
    per_act_token,
    per_out_ch,
    elements_m,
    swap_ab,
):

    assert per_act_token == True
    assert per_out_ch == False

    E, N, _ = w.shape
    M, K = aq.shape
    configs = get_cutlass_groupgemm_best_config(E, N, K)
    if configs:
        # Get the optimal config
        key = min(configs.keys(), key=lambda x: abs(x - elements_m))
        config = configs[min(configs.keys(), key=lambda x: abs(x - elements_m))]
        tile_m, tile_n, tile_k = config["tile_m"], config["tile_n"], config["tile_k"]
        cluster_m, cluster_n, cluster_k = (
            config["cluster_m"],
            config["cluster_n"],
            config["cluster_k"],
        )
        if swap_ab != config["swap_ab"]:
            logging.warning(
                "Using mismatched gemm config swap_ab, potentially causing cutlass groupgemm performance loss."
            )
        return cutlass_moe_mm(
            output,
            aq,
            w,
            aq_scale,
            w_scale,
            expert_offsets,
            problem_sizes,
            a_strides,
            b_strides,
            c_strides,
            per_act_token,
            per_out_ch,
            True,  # profile
            tile_m,
            tile_n,
            tile_k,
            cluster_m,
            cluster_n,
            cluster_k,
            swap_ab,
        )
    else:
        return cutlass_moe_mm(
            output,
            aq,
            w,
            aq_scale,
            w_scale,
            expert_offsets,
            problem_sizes,
            a_strides,
            b_strides,
            c_strides,
            per_act_token,
            per_out_ch,
            False,
            128,
            128,
            128,
            1,
            1,
            1,
            swap_ab,
        )


def get_best_config_swap_ab(
    E: int,
    M: int,
    N: int,
    K: int,
):
    configs = get_cutlass_groupgemm_best_config(E, N, K)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        return config["swap_ab"]
    else:
        return M <= 64 * E

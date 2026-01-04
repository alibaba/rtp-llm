import functools
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel.get_best_config import (
    get_cutlass_groupgemm_best_config,
)
from rtp_llm.models_py.utils.arch import is_cuda
from rtp_llm.models_py.utils.math import align

if is_cuda():
    from rtp_llm.ops.compute_ops import (
        cutlass_moe_mm,
        per_tensor_quant_fp8,
        per_token_group_quant_fp8,
        per_token_group_quant_fp8_v2,
        per_token_group_quant_int8,
        per_token_quant_fp8,
    )
else:
    logging.info("skip import fp8 quant from rtp_llm_ops for non cuda platform")

logger = logging.getLogger(__name__)

fp8_dtype = torch.float8_e4m3fn
finfo = torch.finfo(fp8_dtype)
fp8_max = finfo.max
fp8_min = -fp8_max


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# COPIED FROM DeepGEMM
def ceil_align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# COPIED FROM DeepGEMM
def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


# NOTE copy and modified from DeepGEMM
def _transform_scale_ue8m0(sf, mn):
    import deep_gemm.utils.layout

    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    sf = deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return sf


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
        *x_batch, x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = ceil_align(x_s_mn, 4)
        aligned_k = ceil_align(x_s_k, 4)
        # TODO(FIXME): Fix cuda kernel and recover here to empty.
        return torch.empty(
            (*x_batch, aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(-1, -2)[..., :x_s_mn, :]
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
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))
    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )
    if x.shape[0] > 0:
        if masked_m is not None:
            per_token_group_quant_fp8_v2(
                x,
                x_q,
                x_s,
                group_size,
                eps,
                fp8_min,
                fp8_max,
                scale_ue8m0,
                fuse_silu_and_mul,
                masked_m,
            )
        else:
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


def block_quant_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """This function converts block-wise quantization to unquantized.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The output is an unquantized tensor with dtype.
    """
    block_n, block_k = block_size[0], block_size[1]
    *_, n, k = x_q_block.shape

    # ... n_scale k_scale -> ... (n_scale block_n) (k_scale block_k)
    x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    x_scale_repeat = x_scale_repeat[..., :n, :k]

    return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)


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


def quant_weight_ue8m0(
    weight_dequant: torch.Tensor,
    weight_block_size: List[int],
):
    assert weight_block_size == [128, 128]
    assert (
        weight_dequant.dtype == torch.bfloat16
    ), f"{weight_dequant.dtype=} {weight_dequant.shape=}"

    *batch_dims, n, k = weight_dequant.shape

    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = per_block_cast_to_fp8(weight_dequant_flat, use_ue8m0=True)

    out_w = out_w_flat.view((*batch_dims, n, k))
    out_s = out_s_flat.view(
        (
            *batch_dims,
            ceil_div(n, weight_block_size[0]),
            ceil_div(k, weight_block_size[1]),
        )
    )

    return out_w, out_s


def requant_weight_ue8m0(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weight_block_size = [128, 128]

    weight_dequant = block_quant_dequant(
        weight,
        weight_scale_inv,
        weight_block_size,
        torch.bfloat16,
    )
    out_w, out_s = quant_weight_ue8m0(
        weight_dequant=weight_dequant,
        weight_block_size=weight_block_size,
    )
    out_s = _transform_scale_ue8m0(out_s, mn=out_w.shape[-2])
    return out_w, out_s


def per_token_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, 128)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[
        :, :n
    ].contiguous(), sf

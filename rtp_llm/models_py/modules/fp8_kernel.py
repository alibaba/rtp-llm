
import functools
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.utils import align

from libth_transformer.rtp_llm_ops import per_token_group_quant_int8, per_token_group_quant_fp8


logger = logging.getLogger(__name__)

def sgl_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
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
    if column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = (x.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                x.shape[:-2] + (x.shape[-1] // group_size, aligned_size),
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x.shape[-2], :]
        else:
            x_s = torch.empty(
                (x.shape[-1] // group_size,) + x.shape[:-1],
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )
    if x.shape[0] > 0:
        per_token_group_quant_fp8(x, x_q, x_s, group_size, eps, fp8_min, fp8_max)

    return x_q, x_s
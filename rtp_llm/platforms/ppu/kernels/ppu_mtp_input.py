"""Mask and normalize the two MTP inputs with their original FP8 boundaries."""

import math
from functools import lru_cache

import torch

from .cuda.sglang_jit import load_sglang_kernel


def supports_mtp_input(embedding, hidden, positions, e_weight, h_weight, e_eps, h_eps):
    if embedding.ndim != 2 or hidden.ndim != 3:
        return False
    rows, dim = embedding.shape
    return (
        0 < rows <= 128
        and dim == 4096
        and hidden.shape == (rows, 4, dim)
        and positions.shape == (rows,)
        and positions.dtype in (torch.int32, torch.int64)
        and e_weight.shape == h_weight.shape == (dim,)
        and embedding.is_cuda
        and all(
            t.dtype == torch.bfloat16 for t in (embedding, hidden, e_weight, h_weight)
        )
        and all(
            t.is_contiguous() and t.device == embedding.device
            for t in (embedding, hidden, positions, e_weight, h_weight)
        )
        and all(math.isfinite(eps) and eps > 0 for eps in (e_eps, h_eps))
    )


@lru_cache(maxsize=None)
def _module(arch, position_dtype):
    return load_sglang_kernel(
        "mtp_input_quant_" + position_dtype,
        "compat/mtp_input_quant.cuh",
        f"MtpInputQuant<{position_dtype}>::forward",
        arch,
    )


def mtp_input_quant(embedding, hidden, positions, e_weight, h_weight, e_eps, h_eps):
    if not supports_mtp_input(
        embedding, hidden, positions, e_weight, h_weight, e_eps, h_eps
    ):
        raise ValueError(
            "PPU MTP input fusion requires contiguous BF16 Flash decode inputs"
        )
    rows = embedding.shape[0]
    ep = torch.empty_like(embedding, dtype=torch.float8_e4m3fn)
    hp = torch.empty((rows * 4, 4096), dtype=torch.float8_e4m3fn, device=hidden.device)
    es = torch.empty((32, rows), dtype=torch.float32, device=hidden.device)
    hs = torch.empty((32, rows * 4), dtype=torch.float32, device=hidden.device)
    _module(
        torch.cuda.get_device_capability(hidden.device),
        "int64_t" if positions.dtype == torch.int64 else "int32_t",
    ).forward(
        embedding, hidden, positions, e_weight, h_weight, ep, es, hp, hs, e_eps, h_eps
    )
    return (ep, es.t()), (hp, hs.t())

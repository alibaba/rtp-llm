"""Optional SM90 Triton kernel for delayed-mHC post mixing.

The production path is opt-in through ``DSV41_MHC_SM90_POST=1``.  This module
deliberately contains only the post operation: the experimental pre fusion
changes residual-stream numerics and is not safe for speculative decoding.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

HC = 4
_C_HC = tl.constexpr(HC)
_ENV = "DSV41_MHC_SM90_POST"


def enabled() -> bool:
    """Whether the opt-in post kernel is requested for this process."""
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=None)
def _is_sm90(device: torch.device) -> bool:
    if torch.version.hip is not None or device.type != "cuda":
        return False
    try:
        return tuple(torch.cuda.get_device_capability(device)) == (9, 0)
    except Exception:  # pragma: no cover - invalid/unavailable CUDA device
        return False


def supported(residual: torch.Tensor, hc_mult: int, max_tokens: int = 256) -> bool:
    """Return whether ``residual`` satisfies the production kernel contract."""
    if hc_mult != HC or residual.dtype != torch.bfloat16:
        return False
    if residual.device.type != "cuda" or not residual.is_contiguous():
        return False
    if residual.ndim < 2 or int(residual.shape[-2]) != HC:
        return False
    dim = int(residual.shape[-1])
    if dim <= 0 or dim % 128 != 0:
        return False
    tokens = residual.numel() // (HC * dim)
    return 0 < tokens <= max_tokens and _is_sm90(residual.device)


def _pick_block_h(dim: int) -> int:
    for candidate in (1024, 512, 256, 128):
        if dim % candidate == 0:
            return candidate
    return triton.next_power_of_2(dim)


@triton.jit(do_not_specialize=["T"])
def _mhc_post_kernel(
    x_ptr,  # [T, DIM] bf16
    residual_ptr,  # [T, HC, DIM] bf16
    post_ptr,  # [T, HC] fp32
    comb_ptr,  # [T, HC*HC] fp32
    out_ptr,  # [T, HC, DIM] bf16
    T,
    DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    h_block = tl.program_id(1)

    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < DIM
    lanes = tl.arange(0, 4)

    x = tl.load(x_ptr + token * DIM + h, mask=hmask, other=0.0).to(tl.float32)
    post = tl.load(post_ptr + token * _C_HC + lanes)

    # Preserve TileLang's source-lane accumulation order: post*x, then lanes 0..3.
    mixed = post[:, None] * x[None, :]
    for source_lane in tl.static_range(4):
        residual_lane = tl.load(
            residual_ptr + (token * _C_HC + source_lane) * DIM + h,
            mask=hmask,
            other=0.0,
        ).to(tl.float32)
        coefficient = tl.load(
            comb_ptr + token * 16 + source_lane * 4 + lanes
        )
        mixed += coefficient[:, None] * residual_lane[None, :]

    tl.store(
        out_ptr + (token * _C_HC + lanes[:, None]) * DIM + h[None, :],
        mixed.to(tl.bfloat16),
        mask=hmask[None, :],
    )


def post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run post mixing; ``out`` may alias ``residual``."""
    tokens, hc, dim = residual.shape
    if hc != HC:
        raise ValueError(f"SM90 delayed-mHC post requires hc_mult={HC}")
    block_h = _pick_block_h(dim)
    _mhc_post_kernel[(tokens, triton.cdiv(dim, block_h))](
        x,
        residual,
        post,
        comb,
        out,
        tokens,
        DIM=dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


__all__ = ["enabled", "post_fwd", "supported"]

"""SM90 post-prenorm delayed-mHC epilogue used by ``v41_mega_mhc``.

This module is imported lazily for guarded SM90 decode shapes. It intentionally
accepts an already materialized FP32 ``mixes`` tensor and therefore cannot
change the production prenorm GEMM or split-reduction arithmetic boundary.
"""

from __future__ import annotations

import importlib

import torch

# Initialize libz3/TVM before importing the vendored TileLang kernel.
from rtp_llm.models_py.modules.dsv4 import (
    tilelang_kernels as _tilelang_kernels,  # noqa: F401
)

_mhc_pre_mixes_fuse = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.mhc.pre_mixes_fuse_kernel"
)._mhc_pre_mixes_fuse

HC = 4
DIM = 5120
MIX = 24


def pre_mixes_fwd(
    mixes: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flat ``(pre, post, comb, layer_input)`` tensors.

    The caller owns all shape/architecture guards.  Keeping this wrapper small
    makes graph replay allocation behavior explicit and leaves the exact
    standalone TileLang expressions visible in the kernel source.
    """
    tokens = residual.shape[0]
    if tuple(mixes.shape) != (tokens, MIX):
        raise ValueError(
            f"SM90 mHC mixes shape {tuple(mixes.shape)} != {(tokens, MIX)}"
        )
    if tuple(residual.shape) != (tokens, HC, DIM):
        raise ValueError(
            f"SM90 mHC residual shape {tuple(residual.shape)} != {(tokens, HC, DIM)}"
        )
    if tuple(previous_pre_mix.shape) != (tokens, HC):
        raise ValueError(
            "SM90 mHC predecessor mix shape "
            f"{tuple(previous_pre_mix.shape)} != {(tokens, HC)}"
        )

    pre = torch.empty((tokens, HC), dtype=torch.float32, device=residual.device)
    post = torch.empty_like(pre)
    comb = torch.empty((tokens, HC * HC), dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(
        (tokens, DIM), dtype=torch.bfloat16, device=residual.device
    )
    _mhc_pre_mixes_fuse(
        DIM,
        pre_eps,
        sinkhorn_eps,
        post_mult,
        sinkhorn_repeat,
        mhc_mult=HC,
    )(
        mixes,
        scale,
        base,
        residual,
        previous_pre_mix,
        pre,
        post,
        comb,
        layer_input,
    )
    return pre, post, comb, layer_input


__all__ = ["pre_mixes_fwd"]

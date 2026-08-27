"""Hadamard rotation with a CUDA-extension fast path and Torch fallback."""

from __future__ import annotations

import torch

try:
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard
except (ImportError, OSError):
    # CUDA 13 currently has no torch-2.11-compatible fast-hadamard wheel.
    _fast_hadamard = None


def normalized_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    width = int(x.size(-1))
    if width <= 0 or width & (width - 1):
        raise ValueError(f"Hadamard width must be a positive power of two, got {width}")
    scale = width**-0.5
    if _fast_hadamard is not None:
        return _fast_hadamard(x, scale=scale)

    # The fallback is intentionally expressed only with Torch view and
    # pointwise operations, so it runs on every supported CUDA architecture.
    # It is used by CUDA 13 until a compatible extension wheel is available.
    result = x
    stride = 1
    while stride < width:
        pairs = result.reshape(*result.shape[:-1], -1, 2, stride)
        left, right = pairs.unbind(dim=-2)
        result = torch.stack((left + right, left - right), dim=-2).reshape_as(result)
        stride *= 2
    return result * scale

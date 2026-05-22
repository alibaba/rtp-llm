"""Shared utilities for CUDA attention implementations."""

import functools

import torch


@functools.cache
def is_blackwell() -> bool:
    """Check if current GPU is Blackwell-class (SM 10.0 server / SM 12.0 consumer)."""
    return torch.cuda.get_device_capability()[0] in [10, 12]

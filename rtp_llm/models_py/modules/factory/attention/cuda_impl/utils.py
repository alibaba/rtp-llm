"""Shared utilities for CUDA attention implementations."""

import functools

import torch


@functools.cache
def is_sm_100() -> bool:
    """Check if current GPU is SM 10.0 (Blackwell architecture)."""
    return torch.cuda.get_device_capability()[0] in [10]

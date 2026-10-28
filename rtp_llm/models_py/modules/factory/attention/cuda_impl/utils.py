"""Shared utilities for CUDA attention implementations."""

import functools

import torch


def is_cuda_12_9_or_later() -> bool:
    if not torch.version.cuda:
        return False
    try:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (12, 9)


@functools.cache
def is_sm_100() -> bool:
    """Check if current GPU is SM 10.0 (Blackwell architecture)."""
    return torch.cuda.get_device_capability()[0] in [10]

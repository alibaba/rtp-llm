"""Shared utilities for CUDA attention implementations."""

import torch


def is_cuda_12_9_or_later() -> bool:
    if not torch.version.cuda:
        return False
    try:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (12, 9)

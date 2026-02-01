# -*- coding: utf-8 -*-
"""Decorators for triton kernels with CUDA environment detection."""

import torch
import triton


def cuda_autotune(*args, **kwargs):
    """Wrapper for triton.autotune that only applies when CUDA is available.

    This decorator conditionally applies triton.autotune based on CUDA availability.
    When CUDA is not available, it returns a no-op decorator to avoid errors.

    Usage:
        @cuda_autotune(
            configs=[...],
            key=[...],
        )
        @triton.jit
        def my_kernel(...):
            ...

    Args:
        *args: Positional arguments passed to triton.autotune
        **kwargs: Keyword arguments passed to triton.autotune

    Returns:
        If CUDA is available: triton.autotune decorator
        If CUDA is not available: no-op decorator (lambda f: f)
    """
    if torch.cuda.is_available():
        return triton.autotune(*args, **kwargs)
    else:
        # Return a no-op decorator when CUDA is not available
        return lambda f: f

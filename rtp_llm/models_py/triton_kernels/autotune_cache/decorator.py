# -*- coding: utf-8 -*-
"""CUDA-availability-guarded wrapper for the cache-aware autotune decorator."""

import torch


def cuda_cached_autotune(*args, **kwargs):
    """Wrapper for the cache-aware autotune decorator.

    Same surface as `triton.autotune` but routes through `CachedAutotuner`
    (in `autotune_cache/cache.py`), which consults checked-in JSON configs
    under `autotune_cache/configs/{GPU}/` before falling back to Triton's
    benchmark-and-pick autotune. This eliminates inter-process autotune
    flake on shapes/GPUs that have a generated config in the repo, while
    retaining a safe fallback on uncovered combinations.

    Lookup behavior is controlled by the `TRITON_AUTOTUNE_CACHE_MODE` env
    var (see `CacheMode`: "disabled" | "cached"); when unset (default), the
    cache is disabled and behavior is identical to plain `triton.autotune`.

    On hosts without a usable CUDA GPU (import-time, tests, dev machines),
    returns a no-op decorator so the kernel module still imports
    successfully. `get_gpu_info()` resolves to "unknown" in that case and
    JSON lookup never happens because we don't enter `cached_autotune`.

    Usage:
        @cuda_cached_autotune(
            configs=[...],
            key=[...],
            **autotune_cache_kwargs,
        )
        @triton.jit
        def my_kernel(...):
            ...
    """
    if torch.cuda.is_available():
        from rtp_llm.models_py.triton_kernels.autotune_cache.cache import (
            cached_autotune,
        )

        return cached_autotune(*args, **kwargs)
    else:
        return lambda f: f

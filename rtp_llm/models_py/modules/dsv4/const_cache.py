"""Read-only shape caches; capture-owned tensors are never shared across graphs."""

import torch

_ARANGE_CACHE: dict = {}
_ARANGE_CACHE_MAX = 256

_ZEROED_CACHE: dict = {}
_ZEROED_CACHE_MAX = 64


def _capturing() -> bool:
    """Bypass caches during capture to avoid sharing or freeing another graph's storage."""
    return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()


def cached_arange(n, *, dtype=torch.int64, device=None):
    """Read-only ``torch.arange(n, dtype=dtype, device=device)`` — cached."""
    if _capturing():
        return torch.arange(n, dtype=dtype, device=device)
    key = (n, dtype, device)
    t = _ARANGE_CACHE.get(key)
    if t is None:
        if len(_ARANGE_CACHE) >= _ARANGE_CACHE_MAX:
            _ARANGE_CACHE.clear()
        t = torch.arange(n, dtype=dtype, device=device)
        _ARANGE_CACHE[key] = t
    return t


def cached_zeroed(shape, *, dtype=torch.uint8, device=None):
    """Reuse a zeroed buffer; callers must rewrite consumed rows and preserve its zero tail."""
    key = (tuple(shape), dtype, device)
    if _capturing():
        return torch.zeros(shape, dtype=dtype, device=device)
    t = _ZEROED_CACHE.get(key)
    if t is None:
        if len(_ZEROED_CACHE) >= _ZEROED_CACHE_MAX:
            _ZEROED_CACHE.clear()
        t = torch.zeros(shape, dtype=dtype, device=device)
        _ZEROED_CACHE[key] = t
    return t

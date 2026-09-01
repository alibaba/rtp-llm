"""Shape-keyed caches for read-only torch constants (P1a).

Kills per-call ``torch.arange`` kernel launches in per-layer forward paths
(census: ``arange`` x856 per 8K forward, most hit every layer). All cached
tensors are READ-ONLY by contract — callers must never mutate them in place.
Caches are capped; on overflow they clear and re-warm (a few extra kernels,
no correctness impact).
"""
import torch

_ARANGE_CACHE: dict = {}
_ARANGE_CACHE_MAX = 256

_ZEROED_CACHE: dict = {}
_ZEROED_CACHE_MAX = 64


def cached_arange(n, *, dtype=torch.int64, device=None):
    """Read-only ``torch.arange(n, dtype=dtype, device=device)`` — cached."""
    key = (n, dtype, device)
    t = _ARANGE_CACHE.get(key)
    if t is None:
        if len(_ARANGE_CACHE) >= _ARANGE_CACHE_MAX:
            _ARANGE_CACHE.clear()
        t = torch.arange(n, dtype=dtype, device=device)
        _ARANGE_CACHE[key] = t
    return t


def cached_zeroed(shape, *, dtype=torch.uint8, device=None):
    """Zero-initialized buffer of ``shape`` — cached and reused.

    Contract: the caller's kernel must (a) fully rewrite every entry it
    consumes and (b) rely on zeros only in the never-written pad tail.
    Reuse is serialized on the caller's stream (FIFO), which also makes it
    CUDA-graph-capture safe. Never mutate outside the rewritten region."""
    key = (tuple(shape), dtype, device)
    t = _ZEROED_CACHE.get(key)
    if t is None:
        if len(_ZEROED_CACHE) >= _ZEROED_CACHE_MAX:
            _ZEROED_CACHE.clear()
        t = torch.zeros(shape, dtype=dtype, device=device)
        _ZEROED_CACHE[key] = t
    return t

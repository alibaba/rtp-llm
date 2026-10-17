"""Read-only zero-filled buffers reused across SM120 MoE calls."""

import torch

_ZEROED_CACHE: dict = {}
_ZEROED_CACHE_MAX = 64


def _capturing() -> bool:
    return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()


def cached_zeroed(shape, *, dtype=torch.uint8, device=None) -> torch.Tensor:
    """Reuse a zeroed buffer; callers must preserve its zero tail.

    Capture always allocates so a graph never shares or frees another graph's
    storage.
    """
    if _capturing():
        return torch.zeros(shape, dtype=dtype, device=device)
    key = (tuple(shape), dtype, device)
    cached = _ZEROED_CACHE.get(key)
    if cached is None:
        if len(_ZEROED_CACHE) >= _ZEROED_CACHE_MAX:
            _ZEROED_CACHE.clear()
        cached = torch.zeros(shape, dtype=dtype, device=device)
        _ZEROED_CACHE[key] = cached
    return cached

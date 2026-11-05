"""Cached host-to-device upload of the per-request chunk lengths (G2).

``cp.chunk_lengths_per_req`` is chunk-invariant, but the CP4 prefill
qualification rebuilds it with ``input_lengths.new_tensor(...)`` on every
layer — one pageable host-to-device copy per layer per chunk. With
``DSV41_EXPECTED_LENGTHS_CACHE=1`` the uploaded tensor is cached on the CP
context and reused while the values, dtype and device are unchanged.
Default off: the legacy per-call upload is kept.
"""

import atexit
import os
import sys

import torch

_CACHE_FLAG = "DSV41_EXPECTED_LENGTHS_CACHE"
_CACHE_ATTR = "_v41_expected_lengths_cache"
# R19/D1: the G2 context cache still misses once per chunk because the CP
# context is rebuilt every forward. The worker-level cache survives context
# rebuilds and is keyed only on (values, dtype, device) — the upload is a
# pure function of those, and consumers treat the tensor as read-only (the
# G2 hit path already shares one tensor per context).
_WORKER_FLAG = "DSV41_EXPECTED_LENGTHS_WORKER_CACHE"
_WORKER_CACHE: dict = {}
_WORKER_CACHE_LIMIT = 64
_WORKER_HITS = 0
_WORKER_ANNOUNCED = False


def expected_lengths_cache_enabled() -> bool:
    return os.environ.get(_CACHE_FLAG, "0") == "1"


def expected_lengths_worker_cache_enabled() -> bool:
    """R19/D1 (default off): cross-context cache; removes the one remaining
    pageable H2D upload per rank-chunk of the G2 mechanism."""
    return os.environ.get(_WORKER_FLAG, "0") == "1"


def expected_chunk_lengths(cp, input_lengths: torch.Tensor) -> torch.Tensor:
    """Return ``cp.chunk_lengths_per_req`` as a tensor like ``input_lengths``."""
    values = tuple(cp.chunk_lengths_per_req)
    if expected_lengths_worker_cache_enabled():
        global _WORKER_HITS, _WORKER_ANNOUNCED
        key = (values, input_lengths.dtype, input_lengths.device)
        hit = _WORKER_CACHE.get(key)
        if hit is not None:
            _WORKER_HITS += 1
            if not _WORKER_ANNOUNCED:
                _WORKER_ANNOUNCED = True
                print(
                    f"[ELCACHE] worker-level expected-lengths cache engaged "
                    f"pid={os.getpid()}",
                    file=sys.stderr,
                    flush=True,
                )
            return hit
        tensor = input_lengths.new_tensor(values)
        if len(_WORKER_CACHE) < _WORKER_CACHE_LIMIT:
            _WORKER_CACHE[key] = tensor
        return tensor
    if not expected_lengths_cache_enabled():
        return input_lengths.new_tensor(values)
    cache = getattr(cp, _CACHE_ATTR, None)
    if (
        cache is not None
        and cache[0] == values
        and cache[1].dtype == input_lengths.dtype
        and cache[1].device == input_lengths.device
    ):
        return cache[1]
    tensor = input_lengths.new_tensor(values)
    try:
        setattr(cp, _CACHE_ATTR, (values, tensor))
    except AttributeError:
        pass  # frozen or slotted context: keep the per-call upload
    return tensor

"""Cached host-to-device upload of the per-request chunk lengths (G2).

``cp.chunk_lengths_per_req`` is chunk-invariant, but the CP4 prefill
qualification rebuilds it with ``input_lengths.new_tensor(...)`` on every
layer — one pageable host-to-device copy per layer per chunk. With
``DSV41_EXPECTED_LENGTHS_CACHE=1`` the uploaded tensor is cached on the CP
context and reused while the values, dtype and device are unchanged.
Default off: the legacy per-call upload is kept.
"""

import os

import torch

_CACHE_FLAG = "DSV41_EXPECTED_LENGTHS_CACHE"
_CACHE_ATTR = "_v41_expected_lengths_cache"


def expected_lengths_cache_enabled() -> bool:
    return os.environ.get(_CACHE_FLAG, "0") == "1"


def expected_chunk_lengths(cp, input_lengths: torch.Tensor) -> torch.Tensor:
    """Return ``cp.chunk_lengths_per_req`` as a tensor like ``input_lengths``."""
    values = tuple(cp.chunk_lengths_per_req)
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

"""Process-cached CUDA device capability.

``torch.cuda.get_device_capability`` queries the driver on every call
(``cudaDeviceGetAttribute``); the per-layer guard chains in the hybrid
residual / GDN / MoE paths call it several times per model layer, adding a
real driver round trip to every layer of a serving step.  A device's
capability never changes within a process, so serve it from a cache keyed by
the device itself.  Non-CUDA devices return ``None`` so capability equality
guards decline without raising (guards always check ``is_cuda`` first, so the
distinction is unreachable in practice).
"""

import os
from functools import lru_cache

import torch


_ORIG_GET_DEVICE_CAPABILITY = torch.cuda.get_device_capability


@lru_cache(maxsize=None)
def _cached_device_capability_impl(device_index: int) -> tuple:
    return _ORIG_GET_DEVICE_CAPABILITY(torch.device("cuda", device_index))


def cached_device_capability(device) -> tuple | None:
    """(major, minor) for a CUDA device, cached once per process."""
    resolved = torch.device(device) if isinstance(device, str) else device
    if not isinstance(resolved, torch.device):
        resolved = torch.device(resolved if resolved is not None else "cuda")
    if resolved.type != "cuda":
        return None
    if torch.cuda.get_device_capability is not _ORIG_GET_DEVICE_CAPABILITY:
        return torch.cuda.get_device_capability(resolved)
    idx = resolved.index if resolved.index is not None else torch.cuda.current_device()
    return _cached_device_capability_impl(int(idx))


def sm90_pdl_enabled(device) -> bool:
    """Return True when Hopper Programmatic Dependent Launch is active."""
    return (
        torch.version.hip is None
        and os.environ.get("RTP_LLM_DISABLE_PDL") != "1"
        and cached_device_capability(device) == (9, 0)
    )

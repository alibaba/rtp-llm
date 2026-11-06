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

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def cached_device_capability(device) -> tuple | None:
    """(major, minor) for a CUDA device, cached once per process."""
    resolved = torch.device(device) if isinstance(device, str) else device
    if not isinstance(resolved, torch.device):
        resolved = torch.device(resolved if resolved is not None else "cuda")
    if resolved.type != "cuda":
        return None
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return torch.cuda.get_device_capability(resolved)

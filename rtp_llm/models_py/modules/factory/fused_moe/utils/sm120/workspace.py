"""Model-owned scratch for sequential SM120 MoE layers."""

import threading
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import MoeWorkspace
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    cuda_graph_warmup_forward_enabled,
)

_WORKSPACE_INIT_LOCK = threading.Lock()


def stream_is_capturing(device: Optional[object] = None) -> bool:
    """Return whether the active stream on ``device`` is CUDA-graph capturing."""
    if not torch.cuda.is_available():
        return False
    if not isinstance(device, torch.device):
        return bool(torch.cuda.is_current_stream_capturing())
    if device.type != "cuda":
        return False
    if device.index is None:
        return bool(torch.cuda.is_current_stream_capturing())
    with torch.cuda.device(device):
        return bool(torch.cuda.is_current_stream_capturing())


class Sm120MoeWorkspace:
    """Share scratch across layers, not across independent model instances.

    CudaGraphRunner serializes graph replays and warms every bucket before
    capture. Its graphs may share stable, preallocated scratch. Eager calls
    use separate per-stream storage so they cannot overwrite graph scratch.
    The model must outlive its graphs, as it also owns the captured weights.
    """

    def __init__(self):
        self._caches = {}

    def cache(self, kind: str, device: torch.device) -> tuple[dict, bool, bool]:
        capturing = stream_is_capturing(device)
        graph = capturing or cuda_graph_warmup_forward_enabled()
        if device.type == "cuda":
            index = torch.cuda.current_device() if device.index is None else device.index
            stream = None if graph else torch.cuda.current_stream(device).cuda_stream
        else:
            index = device.index
            stream = None
        key = (kind, index, stream)
        return self._caches.setdefault(key, {}), capturing, graph


def workspace_for(config) -> Sm120MoeWorkspace:
    holder = getattr(config, "moe_workspace", None)
    if isinstance(holder, Sm120MoeWorkspace):
        return holder
    if isinstance(holder, MoeWorkspace):
        return holder.get_or_create(Sm120MoeWorkspace)
    with _WORKSPACE_INIT_LOCK:
        holder = getattr(config, "moe_workspace", None)
        if isinstance(holder, Sm120MoeWorkspace):
            return holder
        if isinstance(holder, MoeWorkspace):
            return holder.get_or_create(Sm120MoeWorkspace)
        workspace = Sm120MoeWorkspace()
        config.moe_workspace = workspace
        return workspace

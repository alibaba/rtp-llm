"""Model-owned scratch for sequential SM120 MoE layers."""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    cuda_graph_warmup_forward_enabled,
)


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
        capturing = torch.cuda.is_current_stream_capturing()
        graph = capturing or cuda_graph_warmup_forward_enabled()
        stream = None if graph else torch.cuda.current_stream(device).cuda_stream
        key = (kind, device.index, stream)
        return self._caches.setdefault(key, {}), capturing, graph


def workspace_for(config) -> Sm120MoeWorkspace:
    workspace = getattr(config, "sm120_workspace", None)
    if workspace is None:
        workspace = Sm120MoeWorkspace()
        config.sm120_workspace = workspace
    return workspace

"""Model-owned scratch for sequential SM120 MoE layers."""

from math import prod
from types import SimpleNamespace

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
        self._decode_graph_pool = None
        self._decode_graph_views = {}

    def cache(self, kind: str, device: torch.device) -> tuple[dict, bool, bool]:
        capturing = torch.cuda.is_current_stream_capturing()
        graph = capturing or cuda_graph_warmup_forward_enabled()
        stream = None if graph else torch.cuda.current_stream(device).cuda_stream
        key = (kind, device.index, stream)
        return self._caches.setdefault(key, {}), capturing, graph

    def reserve_decode_graph_workspace(self, geometry, device):
        if self._decode_graph_pool is None:
            # Publish only a complete pool. Allocation failure releases all
            # scratch and is voted on by the engine before graph capture.
            storage = self._allocate_decode_storage(geometry, device)
            self._decode_graph_pool = (geometry, device, storage)
        elif self._decode_graph_pool[:2] != (geometry, device):
            raise RuntimeError("SM120 decode graph workspace is already reserved")
        return geometry.workspace_bytes

    @staticmethod
    def _allocate_decode_storage(geometry, device):
        return {
            name: torch.empty(prod(shape), dtype=dtype, device=device)
            for name, shape, dtype, _ in geometry.workspace_layout()
        }

    @staticmethod
    def _decode_view(geometry, storage):
        view = SimpleNamespace()
        for name, shape, _, transpose in geometry.workspace_layout():
            tensor = storage[name].narrow(0, 0, prod(shape)).view(shape)
            setattr(view, name, tensor.transpose(1, 2) if transpose else tensor)
        return view

    def decode_graph_workspace(self, plan, geometry, device):
        from rtp_llm.ops.compute_ops import DecodeExecutionMode

        if plan.mode == DecodeExecutionMode.GRAPH:
            view = self._decode_graph_views.get(geometry)
            if view is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "SM120 decode MoE workspace view must be warmed before capture"
                    )
                if self._decode_graph_pool is None:
                    raise RuntimeError("SM120 decode graph workspace was not reserved")
                capacity, pool_device, storage = self._decode_graph_pool
                if (
                    device != pool_device
                    or geometry.rows_per_rank > capacity.rows_per_rank
                    or geometry.world != capacity.world
                    or geometry.hidden != capacity.hidden
                    or geometry.inter != capacity.inter
                    or geometry.total_experts != capacity.total_experts
                    or geometry.topk != capacity.topk
                ):
                    raise ValueError("SM120 decode graph workspace geometry mismatch")
                view = self._decode_view(geometry, storage)
                self._decode_graph_views[geometry] = view
            elif device != self._decode_graph_pool[1]:
                raise ValueError("SM120 decode graph workspace device mismatch")
            return view

        # Uncaptured fixed calls need separate per-stream scratch.
        stream = torch.cuda.current_stream(device).cuda_stream
        key = ("decode", device, stream, geometry)
        workspace = self._caches.get(key)
        if workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "SM120 decode MoE workspace must be warmed before capture"
                )
            workspace = self._decode_view(
                geometry, self._allocate_decode_storage(geometry, device)
            )
            self._caches[key] = workspace
        return workspace


def workspace_for(config) -> Sm120MoeWorkspace:
    workspace = getattr(config, "sm120_workspace", None)
    if workspace is None:
        workspace = Sm120MoeWorkspace()
        config.sm120_workspace = workspace
    return workspace

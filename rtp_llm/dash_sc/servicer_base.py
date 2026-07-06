"""Shared protocol methods for DashSc inference and proxy servicers."""

from __future__ import annotations

from typing import Protocol

from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc


class DashScHealthState(Protocol):
    def is_unavailable(self) -> bool: ...


class DashScServicerBase(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """Implement the health RPCs declared by ``predict_v2.proto``.

    Liveness describes the running process, while readiness follows the
    shutdown manager's admission state so service discovery can remove a
    draining instance without treating it as crashed.
    """

    def __init__(self, health_state: DashScHealthState | None = None) -> None:
        self._health_state = health_state

    async def ServerLive(self, request, context):
        return predict_v2_pb2.ServerLiveResponse(live=True)

    async def ServerReady(self, request, context):
        ready = self._health_state is None or not self._health_state.is_unavailable()
        return predict_v2_pb2.ServerReadyResponse(ready=ready)

    async def ModelReady(self, request, context):
        ready = self._health_state is None or not self._health_state.is_unavailable()
        return predict_v2_pb2.ModelReadyResponse(ready=ready)

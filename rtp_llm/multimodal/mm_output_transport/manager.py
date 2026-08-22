import logging
from typing import List

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseEmbeddingPB,
)
from rtp_llm.multimodal.mm_output_transport.base import (
    MMOutputResult,
    MMTerminalBackend,
    MMTransportBackend,
)
from rtp_llm.multimodal.mm_output_transport.metrics import report_output_metrics
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes


class MMOutputTransport:
    """Selects a data plane and owns fallback and backend lifecycle."""

    def __init__(
        self,
        backends: List[MMTransportBackend],
        terminal: MMTerminalBackend,
    ):
        self._backends = backends
        self._terminal = terminal
        self._all_backends: List[MMTransportBackend] = [*backends, terminal]

    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MultimodalOutputPB:
        for backend in self._backends:
            if not backend.supports(request, res):
                continue
            result = backend.try_transfer(request, res)
            if result is not None:
                return self._finish(result)
        return self._finish(self._terminal.transfer(request, res))

    def _finish(self, result: MMOutputResult) -> MultimodalOutputPB:
        report_output_metrics(result)
        return result.receipt

    def release(self, request: ReleaseEmbeddingPB) -> None:
        handles = list(request.handle)
        if not handles:
            return
        for backend in self._all_backends:
            try:
                backend.release(handles)
            except Exception:  # noqa: BLE001 - remote slot GC is the backstop
                logging.exception(
                    "[VIT] release failed on transport backend %s", backend.name
                )

    def close(self) -> None:
        for backend in self._all_backends:
            try:
                backend.close()
            except Exception:  # noqa: BLE001 - shutdown must continue
                logging.exception(
                    "[VIT] close failed on transport backend %s", backend.name
                )

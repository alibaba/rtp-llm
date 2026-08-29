import logging
from typing import List

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseLeasePB,
)
from rtp_llm.multimodal.transport.base import (
    MMOutputResult,
    MMTerminalBackend,
    MMTransportBackend,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes

METRIC_SOURCE = "vit_server"


def report_output_metrics(result: MMOutputResult) -> None:
    tags = {"source": METRIC_SOURCE, "transport": result.transport}
    kmonitor.report(
        GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC, result.receipt.ByteSize(), tags
    )
    kmonitor.report(
        GaugeMetrics.VIT_RESPONSE_EMBEDDING_BYTES_METRIC,
        result.payload_embedding_bytes,
        tags,
    )
    kmonitor.report(
        GaugeMetrics.VIT_RESPONSE_POS_BYTES_METRIC, result.payload_pos_bytes, tags
    )
    kmonitor.report(
        GaugeMetrics.VIT_RESPONSE_DEEPSTACK_BYTES_METRIC,
        result.payload_extra_bytes,
        tags,
    )
    kmonitor.report(
        GaugeMetrics.VIT_OUTPUT_TOKEN_COUNT_METRIC,
        sum(result.receipt.split_size),
        tags,
    )


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

    def release(self, request: ReleaseLeasePB) -> None:
        handles = list(request.lease_id)
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

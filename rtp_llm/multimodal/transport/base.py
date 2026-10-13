import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseLeasePB,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes

METRIC_SOURCE = "vit_server"


@dataclass
class MMOutputResult:
    receipt: MultimodalOutputPB
    transport: str
    payload_embedding_bytes: int
    payload_pos_bytes: int
    payload_extra_bytes: int


class MMTransportBackend(ABC):
    """ViT-side data-plane interface."""

    name: str

    @abstractmethod
    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MMOutputResult:
        """Transfer the result or raise when the selected data plane fails."""

    def release(self, handles: List[str]) -> None:
        return None

    def close(self) -> None:
        return None


class MMTerminalBackend(MMTransportBackend):
    pass


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
    """Owns the explicitly selected data plane and its lifecycle."""

    def __init__(self, backend: MMTransportBackend):
        self._backend = backend

    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MultimodalOutputPB:
        result = self._backend.transfer(request, res)
        report_output_metrics(result)
        return result.receipt

    def release(self, request: ReleaseLeasePB) -> None:
        handles = list(request.lease_id)
        if not handles:
            return
        try:
            self._backend.release(handles)
        except Exception:  # noqa: BLE001 - remote slot GC is the backstop
            logging.exception(
                "[VIT] release failed on transport backend %s", self._backend.name
            )

    def close(self) -> None:
        try:
            self._backend.close()
        except Exception:  # noqa: BLE001 - shutdown must continue
            logging.exception(
                "[VIT] close failed on transport backend %s", self._backend.name
            )

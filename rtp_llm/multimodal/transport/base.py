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
RDMA_CLAIM_HEADER = "x-rtp-rdma-claim"


def receipt_handles(receipt: MultimodalOutputPB) -> List[str]:
    return [slot.rdma_descriptor.lease_id for slot in receipt.output_rdma_slots]


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

    def offer(self, handles: List[str]) -> bool:
        return False

    def encode_inline(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> bytes | None:
        return None

    def claim(self, handles: List[str]) -> bool:
        return False

    def release(self, handles: List[str]) -> None:
        return None

    def close(self) -> None:
        return None


class MMTerminalBackend(MMTransportBackend):
    pass


def report_output_metrics(result: MMOutputResult) -> None:
    report_output_payload_metrics(
        result.transport,
        result.payload_embedding_bytes,
        result.payload_pos_bytes,
        result.payload_extra_bytes,
        sum(result.receipt.split_size),
    )


def report_output_payload_metrics(
    transport: str,
    embedding_bytes: int,
    position_bytes: int,
    extra_bytes: int,
    token_count: int,
) -> None:
    tags = {"source": METRIC_SOURCE, "transport": transport}
    kmonitor.report(
        GaugeMetrics.VIT_RESPONSE_EMBEDDING_BYTES_METRIC,
        embedding_bytes,
        tags,
    )
    kmonitor.report(
        GaugeMetrics.VIT_RESPONSE_POS_BYTES_METRIC, position_bytes, tags
    )
    kmonitor.report(
        GaugeMetrics.VIT_RESPONSE_DEEPSTACK_BYTES_METRIC,
        extra_bytes,
        tags,
    )
    kmonitor.report(
        GaugeMetrics.VIT_OUTPUT_TOKEN_COUNT_METRIC,
        token_count,
        tags,
    )


def serialize_embedding_response(response: MultimodalOutputPB | bytes) -> bytes:
    # With the upb protobuf runtime, ByteSize() encodes and copies the
    # message too. Measure the bytes gRPC is already going to send.
    if isinstance(response, bytes):
        payload, transport = response, "bytes"
    else:
        payload = response.SerializeToString()
        transport = "rdma" if response.output_rdma_slots else "bytes"
    kmonitor.report(
        GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC,
        len(payload),
        {
            "source": METRIC_SOURCE,
            "transport": transport,
        },
    )
    return payload


class MMOutputTransport:
    """Owns the explicitly selected data plane and its lifecycle."""

    def __init__(self, backend: MMTransportBackend):
        self._backend = backend

    def encode_inline(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> bytes | None:
        return self._backend.encode_inline(request, res)

    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MultimodalOutputPB:
        result = self._backend.transfer(request, res)
        try:
            report_output_metrics(result)
        except Exception:
            self.rollback(result.receipt)
            raise
        return result.receipt

    def offer(self, receipt: MultimodalOutputPB) -> None:
        handles = receipt_handles(receipt)
        if handles:
            receipt.rdma_claim_required = self._backend.offer(handles)

    def claim(self, request: ReleaseLeasePB) -> bool:
        return self._backend.claim(list(request.lease_id))

    def rollback(self, receipt: MultimodalOutputPB) -> None:
        handles = receipt_handles(receipt)
        if handles:
            self._backend.release(handles)

    def release(self, request: ReleaseLeasePB) -> None:
        handles = list(request.lease_id)
        if not handles:
            return
        self._backend.release(handles)

    def close(self) -> None:
        try:
            self._backend.close()
        except Exception:  # noqa: BLE001 - shutdown must continue
            logging.exception(
                "[VIT] close failed on transport backend %s", self._backend.name
            )

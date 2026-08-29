import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaSlotPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
from rtp_llm.multimodal.transport.base import (
    MMOutputResult,
    MMTransportBackend,
)
from rtp_llm.multimodal.transport.manager import METRIC_SOURCE
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.utils.gpu_nic_affinity import configure_gpu_nic_affinity

if TYPE_CHECKING:
    from rtp_llm.ops import MMRdmaOutputExporter

TRANSPORT_RDMA = "rdma"


class RdmaOutputBackend(MMTransportBackend):
    """Export output to registered device memory and return descriptors."""

    name = TRANSPORT_RDMA

    def __init__(self, exporter: Optional["MMRdmaOutputExporter"], rdma_config=None):
        self._exporter = exporter
        self._rdma_config = rdma_config

    @classmethod
    def try_create(cls, rdma_config) -> Optional["RdmaOutputBackend"]:
        try:
            from rtp_llm.ops import MMRdmaOutputExporter

            available = getattr(MMRdmaOutputExporter, "available", None)
            if available is None or not available():
                logging.info(
                    "[VIT] RDMA implementation unavailable; skip GPU-NIC affinity"
                )
                return None

        except Exception as e:  # noqa: BLE001 - RDMA must not break inline fallback
            logging.warning("[VIT] probe mm rdma output exporter failed: %s", e)
            return None
        # Provider construction is deferred until the first request that actually
        # advertises RDMA and has CUDA output.
        return cls(None, rdma_config)

    def _ensure_exporter(self) -> bool:
        if self._exporter is not None:
            return True
        try:
            from rtp_llm.ops import MMRdmaOutputExporter

            device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
            if not configure_gpu_nic_affinity(device_id):
                logging.warning(
                    "[VIT] GPU-NIC affinity unavailable; RDMA may use a non-local NIC"
                )
            exporter = MMRdmaOutputExporter(self._rdma_config)
            if not exporter.enabled():
                return False
            self._exporter = exporter
            return True
        except Exception as e:  # noqa: BLE001 - RDMA must not break inline fallback
            logging.warning("[VIT] init mm rdma output exporter failed: %s", e)
            return False

    def supports(self, request: MultimodalInputsPB, res: MMEmbeddingRes) -> bool:
        return (
            request.support_rdma
            and bool(res.embeddings)
            and res.embeddings[0].is_cuda
        )

    def try_transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> Optional[MMOutputResult]:
        if not self._ensure_exporter():
            return None
        try:
            emb = torch.concat(res.embeddings).contiguous()
            pos = None
            if res.position_ids is not None and len(res.position_ids) > 0:
                pos = torch.concat(res.position_ids).to(device=emb.device).contiguous()
            extras = []
            if res.extra_input is not None and len(res.extra_input) > 0:
                extras = [e.to(device=emb.device).contiguous() for e in res.extra_input]
            desc_bytes_list = self._exporter.export_embedding(emb, pos, extras)
        except Exception as e:  # noqa: BLE001 - RDMA must not break inline fallback
            logging.warning(
                "[VIT] RDMA output preparation failed; falling back to bytes: %s", e
            )
            self._report_error("rdma_export_error")
            return None

        if not desc_bytes_list:
            logging.warning(
                "[VIT] mm rdma export failed; falling back to inline bytes "
                "(embedding_bytes=%d, pos=%s, extra_count=%d)",
                emb.numel() * emb.element_size(),
                pos is not None,
                len(extras),
            )
            return None

        slots: List[MMRdmaSlotPB] = []
        try:
            for desc_bytes in desc_bytes_list:
                if not desc_bytes:
                    raise ValueError("empty RDMA descriptor")
                slot = MMRdmaSlotPB()
                slot.ParseFromString(desc_bytes)
                if not slot.rdma_descriptor.lease_id:
                    raise ValueError("RDMA descriptor has no release handle")
                slots.append(slot)
        except Exception as e:  # noqa: BLE001 - malformed descriptors fall back
            self._roll_back(slots)
            logging.warning(
                "[VIT] invalid RDMA descriptor; falling back to inline bytes: %s", e
            )
            self._report_error("rdma_descriptor_invalid")
            return None

        receipt = MultimodalOutputPB(split_size=[e.shape[0] for e in res.embeddings])
        role_bytes: Dict[int, int] = {}
        for slot in slots:
            receipt.output_rdma_slots.add().CopyFrom(slot)
            for role, tensor in zip(slot.roles, slot.rdma_descriptor.tensors):
                role_bytes[role] = role_bytes.get(role, 0) + tensor.nbytes
        return MMOutputResult(
            receipt=receipt,
            transport=TRANSPORT_RDMA,
            payload_embedding_bytes=role_bytes.get(MMRdmaSlotPB.EMBEDDING, 0),
            payload_pos_bytes=role_bytes.get(MMRdmaSlotPB.POS_ID, 0),
            payload_extra_bytes=role_bytes.get(MMRdmaSlotPB.EXTRA_INPUT, 0),
        )

    def _roll_back(self, slots: List[MMRdmaSlotPB]) -> None:
        handles = [slot.rdma_descriptor.lease_id for slot in slots if slot.rdma_descriptor.lease_id]
        if not handles:
            return
        try:
            self._exporter.release(handles)
        except Exception:  # noqa: BLE001 - exporter GC is the final backstop
            logging.exception(
                "[VIT] failed to roll back RDMA slots after descriptor parse failure"
            )

    def release(self, handles: List[str]) -> None:
        try:
            self._exporter.release(handles)
        except Exception as e:  # noqa: BLE001 - slot GC is the backstop
            logging.warning("[VIT] RDMA release failed: %s", e)
            self._report_error("rdma_release_error")

    def close(self) -> None:
        # Drop the Python reference so the C++ exporter can release its transport resources.
        self._exporter = None
        self._rdma_config = None

    @staticmethod
    def _report_error(reason: str) -> None:
        kmonitor.report(
            AccMetrics.VIT_RPC_SERVER_ERROR_QPS_METRIC,
            1,
            {"source": METRIC_SOURCE, "reason": reason},
        )

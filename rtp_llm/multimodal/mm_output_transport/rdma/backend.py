import logging
from typing import Dict, List, Optional

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaDescPB,
    MMRdmaTensorPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
from rtp_llm.multimodal.mm_output_transport.base import (
    MMOutputResult,
    MMTransportBackend,
)
from rtp_llm.multimodal.mm_output_transport.metrics import METRIC_SOURCE
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.ops import MMRdmaEncoderOp

TRANSPORT_RDMA = "rdma"


class RdmaTransportBackend(MMTransportBackend):
    """Publish output in registered device memory and return descriptors."""

    name = TRANSPORT_RDMA

    def __init__(self, encoder: MMRdmaEncoderOp):
        self._encoder = encoder

    @classmethod
    def try_create(cls, rdma_config) -> Optional["RdmaTransportBackend"]:
        try:
            encoder = MMRdmaEncoderOp(rdma_config)
        except Exception as e:  # noqa: BLE001 - RDMA must not break inline fallback
            logging.warning("[VIT] init mm rdma encoder failed: %s", e)
            return None
        if not encoder.enabled():
            return None
        return cls(encoder)

    def supports(self, request: MultimodalInputsPB, res: MMEmbeddingRes) -> bool:
        return (
            request.support_rdma
            and bool(res.embeddings)
            and res.embeddings[0].is_cuda
        )

    def try_transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> Optional[MMOutputResult]:
        try:
            emb = torch.concat(res.embeddings).contiguous()
            pos = None
            if res.position_ids is not None and len(res.position_ids) > 0:
                pos = torch.concat(res.position_ids).to(device=emb.device).contiguous()
            extras = []
            if res.extra_input is not None and len(res.extra_input) > 0:
                extras = [e.to(device=emb.device).contiguous() for e in res.extra_input]
            desc_bytes_list = self._encoder.export_embedding(emb, pos, extras)
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

        descs: List[MMRdmaDescPB] = []
        try:
            for desc_bytes in desc_bytes_list:
                if not desc_bytes:
                    raise ValueError("empty RDMA descriptor")
                desc = MMRdmaDescPB()
                desc.ParseFromString(desc_bytes)
                if not desc.handle:
                    raise ValueError("RDMA descriptor has no release handle")
                descs.append(desc)
        except Exception as e:  # noqa: BLE001 - malformed descriptors fall back
            self._roll_back(descs)
            logging.warning(
                "[VIT] invalid RDMA descriptor; falling back to inline bytes: %s", e
            )
            self._report_error("rdma_descriptor_invalid")
            return None

        receipt = MultimodalOutputPB(split_size=[e.shape[0] for e in res.embeddings])
        receipt.output_rdma_slots.extend(descs)
        role_bytes: Dict[int, int] = {}
        for desc in descs:
            for tensor in desc.tensors:
                role_bytes[tensor.role] = role_bytes.get(tensor.role, 0) + tensor.nbytes
        return MMOutputResult(
            receipt=receipt,
            transport=TRANSPORT_RDMA,
            payload_embedding_bytes=role_bytes.get(MMRdmaTensorPB.EMBEDDING, 0),
            payload_pos_bytes=role_bytes.get(MMRdmaTensorPB.POS_ID, 0),
            payload_extra_bytes=role_bytes.get(MMRdmaTensorPB.EXTRA_INPUT, 0),
        )

    def _roll_back(self, descs: List[MMRdmaDescPB]) -> None:
        handles = [desc.handle for desc in descs if desc.handle]
        if not handles:
            return
        try:
            self._encoder.release(handles)
        except Exception:  # noqa: BLE001 - encoder GC is the final backstop
            logging.exception(
                "[VIT] failed to roll back RDMA slots after descriptor parse failure"
            )

    def release(self, handles: List[str]) -> None:
        try:
            self._encoder.release(handles)
        except Exception as e:  # noqa: BLE001 - slot GC is the backstop
            logging.warning("[VIT] RDMA release failed: %s", e)
            self._report_error("rdma_release_error")

    @staticmethod
    def _report_error(reason: str) -> None:
        kmonitor.report(
            AccMetrics.VIT_RPC_SERVER_ERROR_QPS_METRIC,
            1,
            {"source": METRIC_SOURCE, "reason": reason},
        )

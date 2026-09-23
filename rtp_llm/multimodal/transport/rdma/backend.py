import logging
import threading
import time
from typing import TYPE_CHECKING, Dict, List

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaSlotPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.multimodal.transport.base import MMOutputResult, MMTransportBackend

if TYPE_CHECKING:
    from rtp_llm.ops import MMRdmaExporter

TRANSPORT_RDMA = "rdma"


class RdmaOutputBackend(MMTransportBackend):
    """Export output to registered device memory and return descriptors."""

    name = TRANSPORT_RDMA

    def __init__(self, exporter: "MMRdmaExporter"):
        self._exporter = exporter

    @classmethod
    def create(cls, rdma_config, local_device_id: int = 0) -> "RdmaOutputBackend":
        from rtp_llm import ops

        ops.ensure_rdma_ops_loaded()
        exporter = ops.MMRdmaExporter(rdma_config, local_device_id)
        if not exporter.enabled():
            raise RuntimeError("RDMA output exporter is disabled")
        return cls(exporter)

    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MMOutputResult:
        if not request.support_rdma:
            raise RuntimeError(
                "RDMA transport was selected but the client did not advertise RDMA support"
            )
        if not res.embeddings:
            raise RuntimeError("RDMA transport received no multimodal embeddings")
        if not res.embeddings[0].is_cuda:
            raise RuntimeError("RDMA transport requires CUDA multimodal embeddings")

        emb = (
            res.embeddings[0]
            if len(res.embeddings) == 1
            else torch.concat(res.embeddings)
        ).contiguous()
        pos = None
        if res.position_ids is not None and len(res.position_ids) > 0:
            pos = (
                (
                    res.position_ids[0]
                    if len(res.position_ids) == 1
                    else torch.concat(res.position_ids)
                )
                .to(device=emb.device)
                .contiguous()
            )
        extras = []
        if res.extra_input is not None and len(res.extra_input) > 0:
            extras = [e.to(device=emb.device).contiguous() for e in res.extra_input]
        receipt = MultimodalOutputPB(split_size=[e.shape[0] for e in res.embeddings])
        embedding_bytes = emb.numel() * emb.element_size()
        pos_bytes = pos.numel() * pos.element_size() if pos is not None else 0
        extra_bytes = sum(e.numel() * e.element_size() for e in extras)
        export_start = time.monotonic()
        try:
            desc_bytes_list = self._exporter.export_embedding(emb, pos, extras)
            if not desc_bytes_list:
                raise RuntimeError(
                    "RDMA output export returned no descriptors "
                    f"(request_id={request.request_id}, embedding_bytes={embedding_bytes}, "
                    f"pos_bytes={pos_bytes}, extra_bytes={extra_bytes})"
                )
        except Exception:
            # native_thread_id matches the C++ log TID emitted by this call.
            logging.exception(
                "[MM-RDMA-EXPORT] failed request_id=%s native_thread_id=%s "
                "device=%s embedding_bytes=%s pos_bytes=%s extra_bytes=%s elapsed_ms=%.3f",
                request.request_id,
                threading.get_native_id(),
                emb.device,
                embedding_bytes,
                pos_bytes,
                extra_bytes,
                (time.monotonic() - export_start) * 1000,
            )
            raise

        slots: List[MMRdmaSlotPB] = []
        parse_error = None
        for desc_bytes in desc_bytes_list:
            try:
                if not desc_bytes:
                    raise ValueError("empty RDMA descriptor")
                slot = MMRdmaSlotPB()
                slot.ParseFromString(desc_bytes)
                if not slot.rdma_descriptor.lease_id:
                    raise ValueError("RDMA descriptor has no release handle")
                slots.append(slot)
            except (
                Exception
            ) as error:  # noqa: BLE001 - parse remaining release handles first
                if parse_error is None:
                    parse_error = error

        if parse_error is not None:
            logging.error(
                "[MM-RDMA-EXPORT] invalid descriptor request_id=%s native_thread_id=%s "
                "descriptors=%s parsed_slots=%s error=%s",
                request.request_id,
                threading.get_native_id(),
                len(desc_bytes_list),
                len(slots),
                parse_error,
            )
            self._roll_back(slots)
            raise RuntimeError(
                f"invalid RDMA descriptor: {parse_error}"
            ) from parse_error

        role_bytes: Dict[int, int] = {}
        for slot in slots:
            receipt.output_rdma_slots.add().CopyFrom(slot)
            for role, tensor in zip(slot.roles, slot.rdma_descriptor.tensors):
                role_bytes[role] = role_bytes.get(role, 0) + tensor.nbytes
        logging.info(
            "[MM-RDMA-EXPORT] ready request_id=%s native_thread_id=%s slots=%s "
            "lease_ids=%s embedding_bytes=%s pos_bytes=%s extra_bytes=%s elapsed_ms=%.3f",
            request.request_id,
            threading.get_native_id(),
            len(slots),
            [slot.rdma_descriptor.lease_id for slot in slots[:8]],
            embedding_bytes,
            pos_bytes,
            extra_bytes,
            (time.monotonic() - export_start) * 1000,
        )
        return MMOutputResult(
            receipt=receipt,
            transport=TRANSPORT_RDMA,
            payload_embedding_bytes=role_bytes.get(MMRdmaSlotPB.EMBEDDING, 0),
            payload_pos_bytes=role_bytes.get(MMRdmaSlotPB.POS_ID, 0),
            payload_extra_bytes=role_bytes.get(MMRdmaSlotPB.EXTRA_INPUT, 0),
        )

    def _roll_back(self, slots: List[MMRdmaSlotPB]) -> None:
        handles = [
            slot.rdma_descriptor.lease_id
            for slot in slots
            if slot.rdma_descriptor.lease_id
        ]
        if not handles:
            return
        try:
            self._exporter.release(handles)
        except Exception:  # noqa: BLE001 - exporter GC is the final backstop
            logging.exception(
                "[VIT] failed to roll back RDMA slots after descriptor parse failure: "
                "leases=%s lease_ids=%s",
                len(handles),
                handles[:8],
            )

    def release(self, handles: List[str]) -> None:
        try:
            self._exporter.release(handles)
        except Exception as e:  # noqa: BLE001 - slot GC is the backstop
            logging.exception(
                "[MM-RDMA-RELEASE] failed leases=%s lease_ids=%s native_thread_id=%s: %s",
                len(handles),
                handles[:8],
                threading.get_native_id(),
                e,
            )

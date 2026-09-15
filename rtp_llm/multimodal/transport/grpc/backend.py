import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.multimodal.transport.base import MMOutputResult, MMTerminalBackend
from rtp_llm.utils.grpc_util import trans_from_tensor

TRANSPORT_BYTES = "bytes"


def _tensor_bytes(tensors) -> int:
    # Reading protobuf bytes fields materializes the payload again.
    return sum(t.numel() * t.element_size() for t in (tensors or []) if t is not None)


def _concat(tensors):
    return tensors[0] if len(tensors) == 1 else torch.concat(tensors)


class GrpcInlineOutputBackend(MMTerminalBackend):
    """Encode the payload directly in the gRPC receipt."""

    name = TRANSPORT_BYTES

    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MMOutputResult:
        receipt = self._build_receipt(res)
        return MMOutputResult(
            receipt=receipt,
            transport=TRANSPORT_BYTES,
            payload_embedding_bytes=_tensor_bytes(res.embeddings),
            payload_pos_bytes=_tensor_bytes(res.position_ids) if res.embeddings else 0,
            payload_extra_bytes=_tensor_bytes(res.extra_input) if res.embeddings else 0,
        )

    def _build_receipt(self, res: MMEmbeddingRes) -> MultimodalOutputPB:
        if not res.embeddings:
            return MultimodalOutputPB()

        contain_pos = (res.position_ids is not None) and (len(res.position_ids) > 0)
        contain_extra_input = (res.extra_input is not None) and (
            len(res.extra_input) > 0
        )
        receipt = MultimodalOutputPB(split_size=[e.shape[0] for e in res.embeddings])
        for layout in res.token_layouts:
            trans_from_tensor(layout, receipt.multimodal_token_layout.add())
        trans_from_tensor(_concat(res.embeddings), receipt.multimodal_embedding)
        if contain_pos:
            trans_from_tensor(_concat(res.position_ids), receipt.multimodal_pos_id)
        if contain_extra_input:
            for extra in res.extra_input:
                trans_from_tensor(extra, receipt.multimodal_extra_input.add())
        return receipt

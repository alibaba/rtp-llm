import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.multimodal.transport.base import MMOutputResult, MMTerminalBackend
from rtp_llm.utils.grpc_util import trans_from_tensor

TRANSPORT_BYTES = "bytes"


class GrpcInlineOutputBackend(MMTerminalBackend):
    """Encode the payload directly in the gRPC receipt."""

    name = TRANSPORT_BYTES

    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MMOutputResult:
        if not res.embeddings:
            return MMOutputResult(MultimodalOutputPB(), TRANSPORT_BYTES, 0, 0, 0)

        receipt = MultimodalOutputPB(split_size=[e.shape[0] for e in res.embeddings])
        embeddings = torch.concat(res.embeddings)
        trans_from_tensor(embeddings, into=receipt.multimodal_embedding)
        pos_bytes = 0
        if res.position_ids:
            positions = torch.concat(res.position_ids)
            trans_from_tensor(positions, into=receipt.multimodal_pos_id)
            pos_bytes = positions.nbytes
        extra_bytes = 0
        for extra in res.extra_input or ():
            trans_from_tensor(extra, into=receipt.multimodal_extra_input.add())
            if extra is not None:
                extra_bytes += extra.nbytes
        return MMOutputResult(
            receipt=receipt,
            transport=TRANSPORT_BYTES,
            # Reading a protobuf bytes field would copy the whole tensor again.
            payload_embedding_bytes=embeddings.nbytes,
            payload_pos_bytes=pos_bytes,
            payload_extra_bytes=extra_bytes,
        )

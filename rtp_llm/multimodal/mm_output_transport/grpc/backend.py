import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.multimodal.mm_output_transport.base import (
    MMOutputResult,
    MMTerminalBackend,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.utils.grpc_util import trans_from_tensor

TRANSPORT_BYTES = "bytes"


def _tensor_pb_bytes(tensor_pb) -> int:
    return (
        len(tensor_pb.fp32_data)
        + len(tensor_pb.int32_data)
        + len(tensor_pb.fp16_data)
        + len(tensor_pb.bf16_data)
    )


class GrpcInlineTerminal(MMTerminalBackend):
    """Encode the payload directly in the gRPC receipt."""

    name = TRANSPORT_BYTES

    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MMOutputResult:
        receipt = self._build_receipt(res)
        return MMOutputResult(
            receipt=receipt,
            transport=TRANSPORT_BYTES,
            payload_embedding_bytes=_tensor_pb_bytes(receipt.multimodal_embedding),
            payload_pos_bytes=_tensor_pb_bytes(receipt.multimodal_pos_id),
            payload_extra_bytes=sum(
                _tensor_pb_bytes(extra) for extra in receipt.multimodal_extra_input
            ),
        )

    def _build_receipt(self, res: MMEmbeddingRes) -> MultimodalOutputPB:
        if not res.embeddings:
            return MultimodalOutputPB()

        contain_pos = (res.position_ids is not None) and (len(res.position_ids) > 0)
        contain_extra_input = (res.extra_input is not None) and (
            len(res.extra_input) > 0
        )
        receipt = MultimodalOutputPB(
            multimodal_embedding=trans_from_tensor(torch.concat(res.embeddings)),
            split_size=[e.shape[0] for e in res.embeddings],
        )
        if contain_pos:
            receipt.multimodal_pos_id.CopyFrom(
                trans_from_tensor(torch.concat(res.position_ids))
            )
        if contain_extra_input:
            for extra in res.extra_input:
                receipt.multimodal_extra_input.append(trans_from_tensor(extra))
        return receipt

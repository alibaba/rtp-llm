import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.multimodal.transport.base import (
    MMOutputResult,
    MMTerminalBackend,
    report_output_payload_metrics,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.ops import ensure_engine_ops_loaded
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate
from rtp_llm.utils.grpc_util import trans_from_tensor

TRANSPORT_BYTES = "bytes"


class GrpcInlineOutputBackend(MMTerminalBackend):
    """Encode the payload directly in the gRPC receipt."""

    name = TRANSPORT_BYTES

    def encode_inline(self, request: MultimodalInputsPB, res: MMEmbeddingRes) -> bytes:
        ensure_engine_ops_loaded()
        from libth_transformer import encode_vit_inline_response

        with cuda_graph_gate.operation():
            payload, embedding_bytes, position_bytes, extra_bytes, tokens = (
                encode_vit_inline_response(
                    res.embeddings,
                    res.position_ids or (),
                    res.extra_input or (),
                    res.feature_hashes,
                )
            )
        report_output_payload_metrics(
            TRANSPORT_BYTES, embedding_bytes, position_bytes, extra_bytes, tokens
        )
        return payload

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

from concurrent import futures

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMPreprocessConfigPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    MultimodalOutputsPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.worker_info import g_worker_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.multimodal.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


def trans_output(res: MMEmbeddingRes):
    output_pb = MultimodalOutputsPB()
    contain_pos = (res.position_ids is not None) and (len(res.position_ids) > 0)
    for i in range(len(res.embeddings)):
        output = MultimodalOutputPB(
            multimodal_embedding=trans_from_tensor(res.embeddings[i]),
            multimodal_pos_id=(
                trans_from_tensor(res.position_ids[i]) if contain_pos else None
            ),
        )
        output_pb.multimodal_outputs.append(output)
    return output_pb


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine):
        self.engine = mm_process_engine

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        res: MMEmbeddingRes = self.engine.mm_embedding_rpc(multimodal_inputs)
        return trans_output(res)


def vit_start_server():
    model = ModelFactory.create_from_env()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=200),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
        ],
    )
    add_MultimodalRpcServiceServicer_to_server(
        MultimodalRpcServer(MMProcessEngine(model)), server
    )
    server.add_insecure_port(f"0.0.0.0:{g_worker_info.rpc_server_port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    vit_start_server()

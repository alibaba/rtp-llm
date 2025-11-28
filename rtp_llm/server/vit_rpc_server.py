from concurrent import futures

import grpc

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    MMPreprocessConfigPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    MultimodalOutputsPB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.distributed_server import DistributedServer, get_world_info
from rtp_llm.distribute.worker_info import g_worker_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.base_model_datatypes import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


def trans_output(res: MMEmbeddingRes):
    output_pb = MultimodalOutputsPB()
    contain_pos = (res.position_ids is not None) and (len(res.position_ids) > 0)
    contain_deepstack = (res.deepstack_embeds is not None) and (
        len(res.deepstack_embeds) > 0
    )
    for i in range(len(res.embeddings)):
        output = MultimodalOutputPB(
            multimodal_embedding=trans_from_tensor(res.embeddings[i]),
            multimodal_pos_id=(
                trans_from_tensor(res.position_ids[i]) if contain_pos else None
            ),
            multimodal_deepstack_embeds=(
                trans_from_tensor(res.deepstack_embeds[i])
                if contain_deepstack
                else None
            ),
        )
        output_pb.multimodal_outputs.append(output)
    return output_pb


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine):
        self.engine = mm_process_engine

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        res: MMEmbeddingRes = self.engine.mm_embedding_rpc(multimodal_inputs))
        return trans_output(res)

    def GetWorkerStatus(self, request: StatusVersionPB, context):
        worker_status = WorkerStatusPB()
        worker_status.role = "VIT"
        worker_status.status_version = 1
        worker_status.alive = True
        return worker_status

    def GetCacheStatus(self, request: CacheVersionPB, context):
        return CacheStatusPB()

    def stop(self):
        self.engine.stop()


def create_rpc_server():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=200),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_concurrent_streams", -1),
            ("grpc.http2.min_ping_interval_without_data_ms", 1000),
            ("grpc.http2.max_ping_strikes", 1000),
        ],
    )
    return server

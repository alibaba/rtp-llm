import logging
from concurrent import futures

import grpc
import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    MMPreprocessConfigPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


def trans_output(res: MMEmbeddingRes):
    # Guard against empty embeddings (e.g. error path where mm_embedding_rpc
    # returns no tensors). torch.concat on an empty list raises RuntimeError.
    # The caller (RemoteMultimodalEmbedding) aborts the RPC with a clear status
    # instead of returning an empty OK response that triggers C++ CHECK failures.
    if not res.embeddings:
        raise ValueError("empty multimodal embedding returned by VIT engine")

    contain_pos = (res.position_ids is not None) and (len(res.position_ids) > 0)
    contain_extra_input = (res.extra_input is not None) and (len(res.extra_input) > 0)

    output_pb = MultimodalOutputPB(
        multimodal_embedding=trans_from_tensor(torch.concat(res.embeddings)),
        split_size=[e.shape[0] for e in res.embeddings],
    )
    if contain_pos:
        output_pb.multimodal_pos_id.CopyFrom(
            trans_from_tensor(torch.concat(res.position_ids))
        )
    if contain_extra_input:
        # Each extra-input is an opaque flat 1-D tensor (one per image).
        for extra in res.extra_input:
            output_pb.multimodal_extra_input.append(trans_from_tensor(extra))
    return output_pb


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine):
        self.engine = mm_process_engine

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        res: MMEmbeddingRes = self.engine.mm_embedding_rpc(multimodal_inputs)
        if not res.embeddings:
            context.abort(
                grpc.StatusCode.INTERNAL,
                "VIT engine returned empty multimodal embeddings",
            )
            return
        res = trans_output(res)
        return res

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

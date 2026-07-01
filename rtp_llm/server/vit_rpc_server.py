import logging
import time
from concurrent import futures
from typing import Dict

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
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.model_factory import ModelFactory
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


def _now_us() -> int:
    return time.monotonic_ns() // 1000


def _tensor_pb_bytes(tensor_pb) -> int:
    return (
        len(tensor_pb.fp32_data)
        + len(tensor_pb.int32_data)
        + len(tensor_pb.fp16_data)
        + len(tensor_pb.bf16_data)
    )


def _report_output_metrics(output_pb: MultimodalOutputPB, tags: Dict[str, str]) -> None:
    kmonitor.report(
        GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC, output_pb.ByteSize(), tags
    )
    kmonitor.report(
        GaugeMetrics.VIT_RESPONSE_EMBEDDING_BYTES_METRIC,
        _tensor_pb_bytes(output_pb.multimodal_embedding),
        tags,
    )
    kmonitor.report(
        GaugeMetrics.VIT_RESPONSE_POS_BYTES_METRIC,
        _tensor_pb_bytes(output_pb.multimodal_pos_id),
        tags,
    )
    kmonitor.report(
        GaugeMetrics.VIT_RESPONSE_DEEPSTACK_BYTES_METRIC,
        sum(_tensor_pb_bytes(extra) for extra in output_pb.multimodal_extra_input),
        tags,
    )
    kmonitor.report(
        GaugeMetrics.VIT_OUTPUT_TOKEN_COUNT_METRIC, sum(output_pb.split_size), tags
    )


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
        tags = {"source": "vit_server"}
        start_us = _now_us()
        lifecycle_reported = False

        def _report_lifecycle():
            nonlocal lifecycle_reported
            if lifecycle_reported:
                return
            lifecycle_reported = True
            kmonitor.report(
                GaugeMetrics.VIT_RPC_SERVER_LIFECYCLE_RT_US_METRIC,
                _now_us() - start_us,
                tags,
            )

        callback_added = False
        if hasattr(context, "add_callback"):
            callback_added = context.add_callback(_report_lifecycle)

        try:
            kmonitor.report(
                GaugeMetrics.VIT_RPC_REQUEST_BYTES_METRIC,
                multimodal_inputs.ByteSize(),
                tags,
            )
            kmonitor.report(
                GaugeMetrics.VIT_INPUT_IMAGE_COUNT_METRIC,
                len(multimodal_inputs.multimodal_inputs),
                tags,
            )
            res: MMEmbeddingRes = self.engine.mm_embedding_rpc(multimodal_inputs)
            if not res.embeddings:
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    "VIT engine returned empty multimodal embeddings",
                )
                return
            output_pb = trans_output(res)
            kmonitor.report(
                GaugeMetrics.VIT_RPC_SERVER_HANDLER_RT_US_METRIC,
                _now_us() - start_us,
                tags,
            )
            _report_output_metrics(output_pb, tags)
            return output_pb
        except Exception:
            kmonitor.report(
                AccMetrics.VIT_RPC_SERVER_ERROR_QPS_METRIC,
                1,
                {"source": "vit_server", "reason": "exception"},
            )
            raise
        finally:
            if not callback_added:
                _report_lifecycle()

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

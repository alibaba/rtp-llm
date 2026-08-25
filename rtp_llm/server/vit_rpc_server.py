import logging
import threading
from concurrent import futures

import grpc
import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    EmptyPB,
    ErrorDetailsPB,
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
from rtp_llm.multimodal.multimodal_util import (
    build_multimodal_output_pb,
    trans_mm_input,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.server.server_args.server_args import setup_args


def trans_output(res: MMEmbeddingRes):
    return build_multimodal_output_pb(res.embeddings, res.position_ids, res.extra_input)


def merge_embedding_results(results: list[MMEmbeddingRes]) -> MMEmbeddingRes:
    embeddings, position_ids, extra_input = [], [], []
    for res in results:
        embeddings.extend(res.embeddings)
        if res.position_ids:
            position_ids.extend(res.position_ids)
        if res.extra_input:
            extra_input.extend(res.extra_input)
    return MMEmbeddingRes(embeddings, position_ids or None, extra_input or None)


def _abort_ft_runtime(context, error: FtRuntimeException) -> None:
    details = ErrorDetailsPB(
        error_code=int(error.exception_type),
        error_message=error.message,
    )
    context.set_trailing_metadata(
        (("grpc-status-details-bin", details.SerializeToString()),)
    )
    if error.exception_type == ExceptionType.CONCURRENCY_LIMIT_ERROR:
        status = grpc.StatusCode.RESOURCE_EXHAUSTED
    elif error.exception_type == ExceptionType.GENERATE_TIMEOUT:
        status = grpc.StatusCode.DEADLINE_EXCEEDED
    elif error.exception_type == ExceptionType.CANCELLED_ERROR:
        status = grpc.StatusCode.CANCELLED
    else:
        status = grpc.StatusCode.INTERNAL
    context.abort(status, f"[{error.exception_type.name}] {error.message}")


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine):
        self.engine = mm_process_engine

    def _register_queue_cancellation(self, request_id: int, context):
        rpc_done = threading.Event()

        def cancel_queued_work() -> None:
            rpc_done.set()
            self.engine.cancel_queued_request(request_id)

        if not context.add_callback(cancel_queued_work):
            cancel_queued_work()
        return rpc_done

    def AsyncSubmitEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        try:
            converted_inputs = trans_mm_input(multimodal_inputs)
            self.engine.async_submit(converted_inputs, multimodal_inputs.request_id)
            return EmptyPB()
        except FtRuntimeException as error:
            _abort_ft_runtime(context, error)

    def WaitGreenNetVerdict(self, multimodal_inputs: MultimodalInputsPB, context):
        """Block until greennet decides for all inputs (kicked earlier by
        AsyncSubmitEmbedding). On a violation, fail the RPC with an
        ErrorDetailsPB(error_code=UNSAFE_INPUT_CONTENT) trailer so the LLM
        client reconstructs the exact FtRuntimeException."""
        try:
            converted_inputs = trans_mm_input(multimodal_inputs)
            cancellation_event = self._register_queue_cancellation(
                multimodal_inputs.request_id, context
            )
            verdict = self.engine.wait_greennet_verdict(
                converted_inputs,
                request_id=multimodal_inputs.request_id,
                cancellation_event=cancellation_event,
            )
        except FtRuntimeException as error:
            _abort_ft_runtime(context, error)
        if not verdict.passed:
            error_code = (
                ExceptionType.UNSAFE_INPUT_CONTENT
                if verdict.code == 2
                else ExceptionType.MM_PROCESS_ERROR
            )
            details = ErrorDetailsPB(
                error_code=int(error_code),
                error_message=verdict.message or "data inspection failed",
            )
            context.set_trailing_metadata(
                (("grpc-status-details-bin", details.SerializeToString()),)
            )
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(verdict.message or "data inspection failed")
        return EmptyPB()

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        try:
            converted_inputs = trans_mm_input(multimodal_inputs)
            cancellation_event = self._register_queue_cancellation(
                multimodal_inputs.request_id, context
            )
            results = self.engine.get_embedding_result(
                converted_inputs,
                request_id=multimodal_inputs.request_id,
                cancellation_event=cancellation_event,
            )
            merged = merge_embedding_results(results)
            return trans_output(merged)
        except FtRuntimeException as error:
            _abort_ft_runtime(context, error)
        except Exception as e:
            logging.exception("RemoteMultimodalEmbedding failed")
            context.abort(
                grpc.StatusCode.INTERNAL, f"[MM_PROCESS_ERROR] {type(e).__name__}: {e}"
            )

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

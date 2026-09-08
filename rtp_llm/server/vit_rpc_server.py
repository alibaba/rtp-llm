import time
from concurrent import futures

import grpc

from rtp_llm.config.exceptions import (
    ExceptionCategory,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    EmptyPB,
    MultimodalInputsPB,
    ReleaseLeasePB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.mm_error_messages import format_mm_rpc_error
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.multimodal.mm_scheduler import (
    MMSchedulerOverloadError,
    MMSchedulerRequestTooLargeError,
    MMSchedulerTimeoutError,
)
from rtp_llm.multimodal.transport import create_mm_output_transport


def _now_us() -> int:
    return time.monotonic_ns() // 1000


_EXCEPTION_CATEGORY_TO_GRPC_STATUS = {
    ExceptionCategory.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    ExceptionCategory.TOO_LONG: grpc.StatusCode.INVALID_ARGUMENT,
    ExceptionCategory.UNSUPPORTED: grpc.StatusCode.INVALID_ARGUMENT,
    ExceptionCategory.CAPACITY: grpc.StatusCode.RESOURCE_EXHAUSTED,
    ExceptionCategory.TIMEOUT: grpc.StatusCode.DEADLINE_EXCEEDED,
    ExceptionCategory.CANCELLED: grpc.StatusCode.CANCELLED,
}


def _grpc_status_for_runtime_exception(
    error: FtRuntimeException,
) -> grpc.StatusCode:
    return _EXCEPTION_CATEGORY_TO_GRPC_STATUS.get(
        error.exception_type.category, grpc.StatusCode.INTERNAL
    )


def _runtime_exception_reason(error: FtRuntimeException) -> str:
    return f"runtime_{error.exception_type.category.value}"


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(
        self,
        mm_process_engine: MMProcessEngine,
        transport_config=None,
        local_device_id: int = 0,
    ):
        self.engine = mm_process_engine
        self._transport = create_mm_output_transport(
            transport_config, local_device_id
        )

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
            output_pb = self._transport.transfer(multimodal_inputs, res)
            kmonitor.report(
                GaugeMetrics.VIT_RPC_SERVER_HANDLER_RT_US_METRIC,
                _now_us() - start_us,
                tags,
            )
            return output_pb
        except MMSchedulerOverloadError as e:
            # Backpressure, not a server fault: map to a defined, ret/backoff-able
            # status instead of a generic error. abort() raises to end the call.
            # NOTE: overload is returned directly to the client here; forwarding to
            # another (untried) worker in the proxy is intentionally NOT done for
            # now — the client/caller decides whether to retry or back off.
            kmonitor.report(
                AccMetrics.VIT_RPC_SERVER_ERROR_QPS_METRIC,
                1,
                {"source": "vit_server", "reason": "overload"},
            )
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                format_mm_rpc_error(
                    FtRuntimeException(ExceptionType.MM_PROCESS_ERROR, str(e))
                ),
            )
        except MMSchedulerTimeoutError as e:
            # Scheduler wait exceeded its embedding timeout.
            kmonitor.report(
                AccMetrics.VIT_RPC_SERVER_ERROR_QPS_METRIC,
                1,
                {"source": "vit_server", "reason": "timeout"},
            )
            context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                format_mm_rpc_error(
                    FtRuntimeException(ExceptionType.MM_PROCESS_ERROR, str(e))
                ),
            )
        except MMSchedulerRequestTooLargeError as e:
            # Client asked for more than a single request may carry -> a caller
            # error, so INVALID_ARGUMENT rather than UNKNOWN.
            kmonitor.report(
                AccMetrics.VIT_RPC_SERVER_ERROR_QPS_METRIC,
                1,
                {"source": "vit_server", "reason": "request_too_large"},
            )
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                format_mm_rpc_error(
                    FtRuntimeException(ExceptionType.MM_WRONG_FORMAT_ERROR, str(e))
                ),
            )
        except FtRuntimeException as e:
            grpc_status = _grpc_status_for_runtime_exception(e)
            kmonitor.report(
                AccMetrics.VIT_RPC_SERVER_ERROR_QPS_METRIC,
                1,
                {"source": "vit_server", "reason": _runtime_exception_reason(e)},
            )
            context.abort(grpc_status, format_mm_rpc_error(e))
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

    def ReleaseRdmaLease(self, request: ReleaseLeasePB, context):
        self._transport.release(request)
        return EmptyPB()

    def GetWorkerStatus(self, request: StatusVersionPB, context):
        worker_status = WorkerStatusPB()
        worker_status.role = "VIT"
        worker_status.status_version = 1
        worker_status.alive = True
        return worker_status

    def GetCacheStatus(self, request: CacheVersionPB, context):
        return CacheStatusPB()

    def stop(self):
        try:
            self.engine.stop()
        finally:
            self._transport.close()


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

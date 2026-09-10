import logging
import threading
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
    ErrorDetailsPB,
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
from rtp_llm.multimodal.multimodal_util import trans_mm_input
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.multimodal.mm_scheduler import (
    MMSchedulerOverloadError,
    MMSchedulerRequestTooLargeError,
    MMSchedulerTimeoutError,
)
from rtp_llm.multimodal.transport import create_mm_output_transport
from rtp_llm.server.vit_rpc_constants import VIT_ERROR_REPORTED_METADATA_KEY


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


def _merge_embedding_results(results: list[MMEmbeddingRes]) -> MMEmbeddingRes:
    embeddings, position_ids, extra_input = [], [], []
    hashes = [] if all(result.feature_hashes is not None for result in results) else None
    for result in results:
        embeddings.extend(result.embeddings)
        if result.position_ids:
            position_ids.extend(result.position_ids)
        if result.extra_input:
            extra_input.extend(result.extra_input)
        if hashes is not None:
            hashes.extend(result.feature_hashes)
    return MMEmbeddingRes(embeddings, position_ids or None, extra_input or None, hashes)


def _mark_vit_error_reported(context, status_details=None) -> None:
    """Tell an optional proxy that the worker already counted this error."""
    metadata = [(VIT_ERROR_REPORTED_METADATA_KEY, "1")]
    if status_details is not None:
        metadata.insert(0, ("grpc-status-details-bin", status_details))
    try:
        context.set_trailing_metadata(tuple(metadata))
    except Exception:
        # Metadata is only for metric de-duplication; never mask the request
        # failure if a custom gRPC context rejects it.
        logging.exception("Failed to attach ViT error metadata")


def _abort_ft_runtime(context, error: FtRuntimeException) -> None:
    details = ErrorDetailsPB(
        error_code=int(error.exception_type),
        error_message=error.message,
    )
    _mark_vit_error_reported(context, details.SerializeToString())
    context.abort(
        _grpc_status_for_runtime_exception(error), format_mm_rpc_error(error)
    )


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

    def _register_queue_cancellation(self, request_id: int, context):
        rpc_done = threading.Event()

        def cancel_queued_work() -> None:
            rpc_done.set()
            try:
                self.engine.cancel_queued_request(request_id)
            except Exception as error:
                # Cancellation runs in gRPC's callback thread, after the
                # handler may have returned; report failures here as well.
                self.engine.report_vit_error(error)
                logging.exception("Failed to cancel queued ViT work")

        if not context.add_callback(cancel_queued_work):
            cancel_queued_work()
        return rpc_done

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
            converted_inputs = trans_mm_input(multimodal_inputs)
            cancellation_event = self._register_queue_cancellation(
                multimodal_inputs.request_id, context
            )
            results = self.engine.get_embedding_result(
                converted_inputs,
                request_id=multimodal_inputs.request_id,
                cancellation_event=cancellation_event,
            )
            res = _merge_embedding_results(results)
            output_pb = self._transport.transfer(multimodal_inputs, res)
            kmonitor.report(
                GaugeMetrics.VIT_RPC_SERVER_HANDLER_RT_US_METRIC,
                _now_us() - start_us,
                tags,
            )
            return output_pb
        except MMSchedulerOverloadError as e:
            self.engine.report_vit_error(e)
            _mark_vit_error_reported(context)
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
            self.engine.report_vit_error(e)
            _mark_vit_error_reported(context)
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
            self.engine.report_vit_error(e)
            _mark_vit_error_reported(context)
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
            self.engine.report_vit_error(e)
            _mark_vit_error_reported(context)
            grpc_status = _grpc_status_for_runtime_exception(e)
            kmonitor.report(
                AccMetrics.VIT_RPC_SERVER_ERROR_QPS_METRIC,
                1,
                {"source": "vit_server", "reason": _runtime_exception_reason(e)},
            )
            context.abort(grpc_status, format_mm_rpc_error(e))
        except Exception as e:
            self.engine.report_vit_error(e)
            _mark_vit_error_reported(context)
            kmonitor.report(
                AccMetrics.VIT_RPC_SERVER_ERROR_QPS_METRIC,
                1,
                {"source": "vit_server", "reason": "exception"},
            )
            raise
        finally:
            if not callback_added:
                _report_lifecycle()

    def AsyncSubmitEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        try:
            converted_inputs = trans_mm_input(multimodal_inputs)
            self.engine.async_submit(converted_inputs, multimodal_inputs.request_id)
            return EmptyPB()
        except FtRuntimeException as error:
            self.engine.report_vit_error(error)
            _abort_ft_runtime(context, error)
        except Exception as error:
            self.engine.report_vit_error(error)
            _mark_vit_error_reported(context)
            logging.exception("AsyncSubmitEmbedding failed")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"[MM_PROCESS_ERROR] {type(error).__name__}: {error}",
            )

    def WaitGreenNetVerdict(self, multimodal_inputs: MultimodalInputsPB, context):
        """Start missing work and block until GreenNet decides for all inputs."""
        verdict = None
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
            if verdict is None:
                raise RuntimeError("ViT GreenNet returned no verdict")
        except FtRuntimeException as error:
            self.engine.report_vit_error(error)
            _abort_ft_runtime(context, error)
            return EmptyPB()
        except Exception as error:
            self.engine.report_vit_error(error)
            _mark_vit_error_reported(context)
            logging.exception("WaitGreenNetVerdict failed")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"[MM_PROCESS_ERROR] {type(error).__name__}: {error}",
            )
            return EmptyPB()

        try:
            if not verdict.passed:
                self.engine.report_vit_error(verdict)
                error_code = (
                    ExceptionType.UNSAFE_INPUT_CONTENT
                    if verdict.code == 2
                    else ExceptionType.MM_PROCESS_ERROR
                )
                details = ErrorDetailsPB(
                    error_code=int(error_code),
                    error_message=verdict.message or "data inspection failed",
                )
                _mark_vit_error_reported(context, details.SerializeToString())
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details(verdict.message or "data inspection failed")
        except Exception as error:
            # A malformed verdict or response-metadata failure is also an
            # exceptional result and must be visible in the error QPS.
            self.engine.report_vit_error(error)
            _mark_vit_error_reported(context)
            logging.exception("Failed to serialize ViT GreenNet verdict")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"[MM_PROCESS_ERROR] {type(error).__name__}: {error}",
            )
            return EmptyPB()
        return EmptyPB()

    def ReleaseRdmaLease(self, request: ReleaseLeasePB, context):
        try:
            self._transport.release(request)
            return EmptyPB()
        except Exception as error:
            self.engine.report_vit_error(error)
            _mark_vit_error_reported(context)
            logging.exception("ReleaseRdmaLease failed")
            raise

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

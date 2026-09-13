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
    MMRdmaDescPB,
    MultimodalHashRequestPB,
    MultimodalHashResponsePB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseEmbeddingPB,
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
    add_multimodal_feature_hashes,
    build_multimodal_output_pb,
    trans_mm_input,
)
from rtp_llm.ops import MMPreprocessConfig, MMRdmaEncoderOp, MultimodalInput
from rtp_llm.server.mm_cache_metadata import (
    MM_CACHE_SNAPSHOT_MAX_BYTES,
    get_mm_cache_keys,
    get_mm_cache_metadata,
    metadata_to_proto,
)
from rtp_llm.server.request_headers import extract_request_headers
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.server.vit_rpc_constants import VIT_ERROR_REPORTED_METADATA_KEY
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


def _rpc_timeout_ms(context, default_ms: int) -> int:
    remaining = context.time_remaining()
    if remaining is None:
        return default_ms
    return max(0, min(default_ms, int(remaining * 1000)))


def trans_output(res: MMEmbeddingRes):
    return build_multimodal_output_pb(
        res.embeddings, res.position_ids, res.extra_input, res.feature_hashes
    )


def merge_embedding_results(results: list[MMEmbeddingRes]) -> MMEmbeddingRes:
    embeddings, position_ids, extra_input = [], [], []
    hashes = [] if all(res.feature_hashes is not None for res in results) else None
    for res in results:
        embeddings.extend(res.embeddings)
        if res.position_ids:
            position_ids.extend(res.position_ids)
        if res.extra_input:
            extra_input.extend(res.extra_input)
        if hashes is not None:
            hashes.extend(res.feature_hashes)
    return MMEmbeddingRes(embeddings, position_ids or None, extra_input or None, hashes)


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
    elif error.exception_type == ExceptionType.UNSAFE_INPUT_CONTENT:
        status = grpc.StatusCode.PERMISSION_DENIED
    else:
        status = grpc.StatusCode.INTERNAL
    context.abort(status, f"[{error.exception_type.name}] {error.message}")


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine, vit_config=None):
        self.engine = mm_process_engine
        self._rdma = None
        if (
            vit_config is not None
            and getattr(vit_config, "mm_transport_mode", "auto") == "auto"
        ):
            try:
                rdma = MMRdmaEncoderOp(vit_config)
                if rdma.enabled():
                    self._rdma = rdma
                    logging.info("[VIT] mm rdma encoder enabled")
                else:
                    logging.warning(
                        "[VIT] mm rdma requested but unavailable, fall back to bytes"
                    )
            except (
                Exception
            ) as e:  # noqa: BLE001 - never let rdma init break the bytes path
                logging.warning(
                    "[VIT] init mm rdma encoder failed: %s, fall back to bytes", e
                )

    def _register_rpc_completion(self, context):
        rpc_done = threading.Event()
        # gRPC invokes this callback on successful completion too. Only notify
        # the waiter; the engine cancels queued work when its wait is aborted.
        if not context.add_callback(rpc_done.set):
            rpc_done.set()
        return rpc_done

    @cuda_graph_gate.operation()
    def _trans_output_rdma(self, res: MMEmbeddingRes):
        """Export the whole output of one request (embedding + pos_id + every extra_input)
        through one or more RDMA slots and return a descriptor-bearing MultimodalOutputPB. Only
        split_size stays inline. When the output fits one slot it is carried in output_rdma;
        when it is larger than one slot (mm_rdma_max_slot_bytes) the encoder splits it and the
        descriptors are carried in output_rdma_chunks. Returns None to signal fallback to the
        inline-bytes path.
        """
        if self._rdma is None or not res.embeddings:
            return None
        if not all(emb.is_cuda for emb in res.embeddings):
            return None

        device = res.embeddings[0].device
        pos = None
        if res.position_ids is not None and len(res.position_ids) > 0:
            pos = torch.concat(res.position_ids).to(device=device).contiguous()
        extras = []
        if res.extra_input is not None and len(res.extra_input) > 0:
            extras = [e.to(device=device).contiguous() for e in res.extra_input]

        # export_embedding returns a list of serialized MMRdmaDescPB (one per RDMA slot): a single
        # element for the common fits-in-one-slot case, N when the output was chunked, and an empty
        # list on failure (-> fall back to inline bytes).
        desc_bytes_list = self._rdma.export_embedding(res.embeddings, pos, extras)
        if not desc_bytes_list:
            logging.warning(
                "[VIT] mm rdma export failed; falling back to inline bytes "
                "(embedding_bytes=%d, pos=%s, extra_count=%d)",
                sum(emb.numel() * emb.element_size() for emb in res.embeddings),
                pos is not None,
                len(extras),
            )
            return None
        descs = []
        for desc_bytes in desc_bytes_list:
            if not desc_bytes:
                return None
            desc = MMRdmaDescPB()
            desc.ParseFromString(desc_bytes)
            descs.append(desc)

        output_pb = MultimodalOutputPB(split_size=[e.shape[0] for e in res.embeddings])
        add_multimodal_feature_hashes(output_pb, res.embeddings, res.feature_hashes)
        if len(descs) == 1:
            output_pb.output_rdma.CopyFrom(descs[0])
        else:
            for desc in descs:
                output_pb.output_rdma_chunks.add().CopyFrom(desc)
        return output_pb

    def GetMultimodalHashes(self, request: MultimodalHashRequestPB, context):
        try:
            inputs = request.inputs if request.HasField("inputs") else None
            cancellation = (
                self._register_rpc_completion(context) if inputs is not None else None
            )
            timeout_ms = request.timeout_ms or 120000
            remaining = context.time_remaining()
            if remaining is not None:
                if remaining <= 0:
                    raise TimeoutError("ViT hash acquisition timed out")
                timeout_ms = min(timeout_ms, max(1, int(remaining * 1000)))
            headers = extract_request_headers(dict(context.invocation_metadata() or ()))
            return metadata_to_proto(
                get_mm_cache_metadata(
                    self.engine,
                    request.keys,
                    inputs,
                    timeout_ms,
                    user_id=headers.get("x-dashscope-uid", ""),
                    service_name=headers.get("x-dashscope-service", ""),
                    cancellation_event=cancellation,
                    binary_hashes=True,
                )
            )
        except NotImplementedError as error:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(error))
        except OverflowError as error:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except TimeoutError as error:
            if inputs is not None:
                self.engine.cancel_queued_request(inputs.request_id)
            self.engine.report_vit_error(error)
            _abort_ft_runtime(
                context, FtRuntimeException(ExceptionType.GENERATE_TIMEOUT, str(error))
            )
        except FtRuntimeException as error:
            if inputs is not None and error.exception_type in (
                ExceptionType.CANCELLED_ERROR,
                ExceptionType.GENERATE_TIMEOUT,
            ):
                self.engine.cancel_queued_request(inputs.request_id)
            self.engine.report_vit_error(error)
            _abort_ft_runtime(context, error)
        except Exception as error:
            self.engine.report_vit_error(error)
            _abort_ft_runtime(
                context, FtRuntimeException(ExceptionType.MM_PROCESS_ERROR, str(error))
            )
        return MultimodalHashResponsePB()

    def AsyncSubmitEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        try:
            converted_inputs = trans_mm_input(multimodal_inputs)
            self.engine.async_submit(
                converted_inputs,
                multimodal_inputs.request_id,
                user_id=extract_request_headers(
                    dict(context.invocation_metadata() or ())
                ).get("x-dashscope-uid", ""),
                service_name=extract_request_headers(
                    dict(context.invocation_metadata() or ())
                ).get("x-dashscope-service", ""),
            )
            return EmptyPB()
        except FtRuntimeException as error:
            self.engine.report_vit_error(error)
            _abort_ft_runtime(context, error)
        except Exception as error:
            self.engine.report_vit_error(error)
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
            cancellation_event = self._register_rpc_completion(context)
            verdict = self.engine.wait_greennet_verdict(
                converted_inputs,
                timeout_ms=_rpc_timeout_ms(context, 60000),
                request_id=multimodal_inputs.request_id,
                cancellation_event=cancellation_event,
                user_id=extract_request_headers(
                    dict(context.invocation_metadata() or ())
                ).get("x-dashscope-uid", ""),
                service_name=extract_request_headers(
                    dict(context.invocation_metadata() or ())
                ).get("x-dashscope-service", ""),
            )
            if verdict is None:
                raise RuntimeError("ViT GreenNet returned no verdict")
        except FtRuntimeException as error:
            self.engine.report_vit_error(error)
            _abort_ft_runtime(context, error)
            return EmptyPB()
        except TimeoutError as error:
            timeout_error = FtRuntimeException(
                ExceptionType.GENERATE_TIMEOUT, str(error)
            )
            self.engine.report_vit_error(timeout_error)
            _abort_ft_runtime(context, timeout_error)
            return EmptyPB()
        except Exception as error:
            self.engine.report_vit_error(error)
            logging.exception("WaitGreenNetVerdict failed")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"[MM_PROCESS_ERROR] {type(error).__name__}: {error}",
            )
            return EmptyPB()

        try:
            if not verdict.passed:
                self.engine.cancel_queued_request(multimodal_inputs.request_id)
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
                context.set_trailing_metadata(
                    (("grpc-status-details-bin", details.SerializeToString()),)
                )
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details(verdict.message or "data inspection failed")
        except Exception as error:
            # A malformed verdict or response-metadata failure is also an
            # exceptional result and must be visible in the error QPS.
            self.engine.report_vit_error(error)
            logging.exception("Failed to serialize ViT GreenNet verdict")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"[MM_PROCESS_ERROR] {type(error).__name__}: {error}",
            )
            return EmptyPB()
        return EmptyPB()

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        try:
            converted_inputs = trans_mm_input(multimodal_inputs)
            cancellation_event = self._register_rpc_completion(context)
            results = self.engine.get_embedding_result(
                converted_inputs,
                timeout_ms=_rpc_timeout_ms(context, 120000),
                request_id=multimodal_inputs.request_id,
                cancellation_event=cancellation_event,
                user_id=extract_request_headers(
                    dict(context.invocation_metadata() or ())
                ).get("x-dashscope-uid", ""),
                service_name=extract_request_headers(
                    dict(context.invocation_metadata() or ())
                ).get("x-dashscope-service", ""),
            )
            merged = merge_embedding_results(results)
            logging.debug(
                "[VIT] transport negotiation: support_rdma=%s, rdma_ready=%s, embeddings=%d",
                getattr(multimodal_inputs, "support_rdma", False),
                self._rdma is not None,
                len(merged.embeddings),
            )
            if (
                getattr(multimodal_inputs, "support_rdma", False)
                and self._rdma is not None
            ):
                rdma_out = self._trans_output_rdma(merged)
                if rdma_out is not None:
                    return rdma_out
            return trans_output(merged)
        except FtRuntimeException as error:
            self.engine.report_vit_error(error)
            _abort_ft_runtime(context, error)
        except TimeoutError as error:
            timeout_error = FtRuntimeException(
                ExceptionType.GENERATE_TIMEOUT, str(error)
            )
            self.engine.report_vit_error(timeout_error)
            _abort_ft_runtime(context, timeout_error)
            return EmptyPB()
        except Exception as e:
            self.engine.report_vit_error(e)
            logging.exception("RemoteMultimodalEmbedding failed")
            context.abort(
                grpc.StatusCode.INTERNAL, f"[MM_PROCESS_ERROR] {type(e).__name__}: {e}"
            )

    def ReleaseMultimodalEmbedding(self, request: ReleaseEmbeddingPB, context):
        try:
            if self._rdma is not None and len(request.handle) > 0:
                self._rdma.release(list(request.handle))
            return EmptyPB()
        except Exception as error:
            self.engine.report_vit_error(error)
            logging.exception("ReleaseMultimodalEmbedding failed")
            raise

    def GetWorkerStatus(self, request: StatusVersionPB, context):
        worker_status = WorkerStatusPB()
        worker_status.role = "VIT"
        worker_status.status_version = 1
        worker_status.alive = True
        return worker_status

    def GetCacheStatus(self, request: CacheVersionPB, context):
        # Frequent status polls must not build or transfer the routing directory.
        if not request.need_cache_keys:
            return CacheStatusPB()
        try:
            response = CacheStatusPB(multimodal_cache=get_mm_cache_keys(self.engine))
            if response.ByteSize() > MM_CACHE_SNAPSHOT_MAX_BYTES:
                raise OverflowError("cache key snapshot exceeds gRPC response limit")
            return response
        except NotImplementedError as error:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(error))
        except OverflowError as error:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))

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

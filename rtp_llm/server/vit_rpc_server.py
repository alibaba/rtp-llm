import logging
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
    build_multimodal_output_pb,
    trans_mm_input,
)
from rtp_llm.ops import MMPreprocessConfig, MMRdmaEncoderOp, MultimodalInput
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


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine, vit_config=None):
        self.engine = mm_process_engine
        self._rdma = None
        self._rdma_min_bytes = 0
        if vit_config is not None and getattr(vit_config, "mm_rdma_enable", False):
            try:
                rdma = MMRdmaEncoderOp(vit_config)
                if rdma.enabled():
                    self._rdma = rdma
                    self._rdma_min_bytes = int(
                        getattr(vit_config, "mm_rdma_min_bytes", 256 * 1024)
                    )
                    logging.info(
                        "[VIT] mm rdma encoder enabled, min_bytes=%d",
                        self._rdma_min_bytes,
                    )
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

    def _trans_output_rdma(self, res: MMEmbeddingRes):
        """Export the whole output of one request (embedding + pos_id + every extra_input)
        through a single RDMA slot and return a descriptor-bearing MultimodalOutputPB. Only
        split_size stays inline. Returns None to signal fallback to the inline-bytes path.
        """
        if self._rdma is None or not res.embeddings:
            return None
        emb = torch.concat(res.embeddings).contiguous()
        if not emb.is_cuda:
            return None

        pos = None
        if res.position_ids is not None and len(res.position_ids) > 0:
            pos = torch.concat(res.position_ids).contiguous()
        extras = []
        if res.extra_input is not None and len(res.extra_input) > 0:
            extras = [e.contiguous() for e in res.extra_input]

        # Threshold on the TOTAL payload (embedding + pos_id + extra_input): extra_input
        # (deepstack) is often the larger share, so it must count toward the decision.
        nbytes = emb.numel() * emb.element_size()
        if pos is not None:
            nbytes += pos.numel() * pos.element_size()
        for extra in extras:
            nbytes += extra.numel() * extra.element_size()
        if nbytes < self._rdma_min_bytes:
            return None

        desc_bytes = self._rdma.export_embedding(emb, pos, extras)
        if not desc_bytes:
            return None
        desc = MMRdmaDescPB()
        desc.ParseFromString(desc_bytes)

        output_pb = MultimodalOutputPB(split_size=[e.shape[0] for e in res.embeddings])
        output_pb.output_rdma.CopyFrom(desc)
        return output_pb

    def AsyncSubmitEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        converted_inputs = trans_mm_input(multimodal_inputs)
        self.engine.async_submit(converted_inputs)
        return EmptyPB()

    def WaitGreenNetVerdict(self, multimodal_inputs: MultimodalInputsPB, context):
        """Block until greennet decides for all inputs (kicked earlier by
        AsyncSubmitEmbedding). On a violation, fail the RPC with an
        ErrorDetailsPB(error_code=UNSAFE_INPUT_CONTENT) trailer so the LLM
        client reconstructs the exact FtRuntimeException."""
        converted_inputs = trans_mm_input(multimodal_inputs)
        verdict = self.engine.wait_greennet_verdict(converted_inputs)
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
            results = self.engine.get_embedding_result(converted_inputs)
            merged = merge_embedding_results(results)
            if (
                getattr(multimodal_inputs, "support_rdma", False)
                and self._rdma is not None
            ):
                rdma_out = self._trans_output_rdma(merged)
                if rdma_out is not None:
                    return rdma_out
            return trans_output(merged)
        except FtRuntimeException as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"[{e.exception_type.name}] {e.message}"
            )
        except Exception as e:
            logging.exception("RemoteMultimodalEmbedding failed")
            context.abort(
                grpc.StatusCode.INTERNAL, f"[MM_PROCESS_ERROR] {type(e).__name__}: {e}"
            )

    def ReleaseMultimodalEmbedding(self, request: ReleaseEmbeddingPB, context):
        if self._rdma is not None and len(request.handle) > 0:
            self._rdma.release(list(request.handle))
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

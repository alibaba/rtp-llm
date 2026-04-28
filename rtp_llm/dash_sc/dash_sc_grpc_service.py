"""GRPCInferenceService servicer: ``ModelStreamInfer`` (predict_v2.proto wire)."""

from __future__ import annotations

import logging

from rtp_llm.dash_sc.dash_sc_grpc_real_infer import (
    _iter_enqueue_sync,
    iter_real_model_stream_infer,
)
from rtp_llm.dash_sc.dash_sc_grpc_request import parse_dash_sc_grpc_request
from rtp_llm.dash_sc.dash_sc_grpc_response_fake import iter_fake_model_stream_infer
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.utils.util import AtomicCounter


class DashScGrpcInferenceServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """ModelStreamInfer: fake mode (mock) or real mode (backend_visitor.enqueue).

    ``ip`` / ``port`` / ``server_id`` are used to derive the snowflake-style
    ``GenerateInput.request_id`` via ``generate_request_id`` — same scheme as the HTTP path
    in ``FrontendServer`` so the backend sees a single request_id generation policy.
    ``port`` should be the dash_sc gRPC listening port. The per-servicer sequence counter
    is intentionally independent of ``FrontendServer._global_controller``.
    """

    def __init__(
        self,
        backend_visitor=None,
        *,
        ip: str = "",
        port: int = 0,
        server_id: str = "",
    ):
        self._backend_visitor = backend_visitor
        self._ip = ip
        self._port = port
        self._server_id = server_id
        self._seq_counter = AtomicCounter()

    def close(self):
        pass

    def _next_rtp_llm_request_id(self) -> int:
        sequence = self._seq_counter.increment() % 4096  # 12 bits
        return generate_request_id(self._ip, self._port, self._server_id, sequence)

    def ModelStreamInfer(self, request_iterator, context):
        for request in request_iterator:
            logging.debug(
                "[DashScGrpc] ModelInferRequest: id=%s model_name=%s",
                request.id,
                request.model_name,
            )
            input_ids_list, sampling, other = parse_dash_sc_grpc_request(request)
            if input_ids_list is None:
                yield predict_v2_pb2.ModelStreamInferResponse(
                    error_message="input_ids not found or raw_input_contents mismatch"
                )
                return

            if self._backend_visitor is None:
                yield from iter_fake_model_stream_infer(
                    request, input_ids_list, sampling.top_k
                )
            else:
                yield from iter_real_model_stream_infer(
                    request,
                    input_ids_list,
                    sampling,
                    other,
                    self._backend_visitor,
                    rtp_llm_request_id=self._next_rtp_llm_request_id(),
                    run_enqueue_sync=_iter_enqueue_sync,
                )

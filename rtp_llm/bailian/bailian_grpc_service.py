"""GRPCInferenceService servicer: ``ModelStreamInfer`` (predict_v2.proto wire)."""

from __future__ import annotations

import logging

from rtp_llm.bailian.bailian_grpc_real_infer import (
    _iter_enqueue_sync,
    iter_real_model_stream_infer,
)
from rtp_llm.bailian.bailian_grpc_request import parse_bailian_grpc_request
from rtp_llm.bailian.bailian_grpc_response_fake import iter_fake_model_stream_infer
from rtp_llm.bailian.proto import predict_v2_pb2, predict_v2_pb2_grpc


class BailianGrpcInferenceServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """ModelStreamInfer: fake mode (mock) or real mode (backend_visitor.enqueue)."""

    def __init__(self, backend_visitor=None):
        self._backend_visitor = backend_visitor

    def ModelStreamInfer(self, request_iterator, context):
        for request in request_iterator:
            logging.debug(
                "[BailianGrpc] ModelInferRequest: id=%s model_name=%s",
                request.id,
                request.model_name,
            )
            input_ids_list, sampling, other = parse_bailian_grpc_request(request)
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
                    run_enqueue_sync=_iter_enqueue_sync,
                )

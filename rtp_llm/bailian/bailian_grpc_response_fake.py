"""Fake Bailian gRPC ModelStreamInfer responses (mock generated_ids, no backend)."""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterator

from rtp_llm.bailian.proto import predict_v2_pb2


def iter_fake_model_stream_infer(
    request,
    input_ids_list: list[int],
    top_k: int,
) -> Iterator[predict_v2_pb2.ModelStreamInferResponse]:
    """Mock: ``generated_ids = input_ids + 100``, single stream chunk, ``finish_reason=0`` (finished)."""
    del top_k  # unused in fake path
    out_ids = [x + 100 for x in input_ids_list]
    gen_raw = struct.pack("<%di" % len(out_ids), *out_ids)
    stream_resp = predict_v2_pb2.ModelStreamInferResponse()
    stream_resp.infer_response.id = request.id
    stream_resp.infer_response.model_name = request.model_name
    out_gen = stream_resp.infer_response.outputs.add()
    out_gen.name = "generated_ids"
    out_gen.datatype = "INT32"
    out_gen.shape[:] = [1, len(out_ids)]
    logging.debug("[BailianGrpc] fake out_gen.shape: %s", list(out_gen.shape))
    stream_resp.infer_response.raw_output_contents.append(gen_raw)
    out_finish = stream_resp.infer_response.outputs.add()
    out_finish.name = "finish_reason"
    out_finish.datatype = "INT64"
    out_finish.shape.append(1)
    stream_resp.infer_response.raw_output_contents.append(struct.pack("<q", 0))
    yield stream_resp

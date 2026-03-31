"""Build ``ModelStreamInferResponse`` from ``GenerateOutputs`` (predict_v2 wire / protobuf)."""

from __future__ import annotations

import logging
import struct
from typing import Any

from rtp_llm.bailian.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import GenerateOutputs


def _set_infer_response_identity(
    infer: predict_v2_pb2.ModelInferResponse,
    request_id: str,
    model_name: str,
) -> None:
    infer.id = request_id
    infer.model_name = model_name


def _token_ids_list_from_generate_output(out_py: Any) -> list[int]:
    ids: list[int] = []
    if out_py.output_ids is not None:
        t = out_py.output_ids
        if t.dim() > 1:
            t = t[0]
        ids = t.cpu().int().tolist()
    return ids


def _append_prompt_token_ids_output(
    infer: predict_v2_pb2.ModelInferResponse,
    prompt_token_ids: list[int],
) -> None:
    """``prompt_token_ids``：INT32 little-endian，shape ``[1, len]``。"""
    raw = (
        struct.pack("<%di" % len(prompt_token_ids), *prompt_token_ids)
        if prompt_token_ids
        else struct.pack("<i", 0)
    )
    out = infer.outputs.add()
    out.name = "prompt_token_ids"
    out.datatype = "INT32"
    out.shape[:] = [1, len(prompt_token_ids)]
    infer.raw_output_contents.append(raw)


def _append_generated_ids_output(
    infer: predict_v2_pb2.ModelInferResponse,
    generated_ids: list[int],
) -> None:
    """``generated_ids``：INT32 little-endian，shape ``[1, len]``。"""
    raw = (
        struct.pack("<%di" % len(generated_ids), *generated_ids)
        if generated_ids
        else struct.pack("<i", 0)
    )
    out = infer.outputs.add()
    out.name = "generated_ids"
    out.datatype = "INT32"
    out.shape[:] = [1, len(generated_ids)]
    infer.raw_output_contents.append(raw)


def _append_finish_reason_output(
    infer: predict_v2_pb2.ModelInferResponse,
    finished: bool,
) -> None:
    """``finish_reason``：INT64 little-endian 标量，shape ``[1]``；finished=0，not finished=2。"""
    out = infer.outputs.add()
    out.name = "finish_reason"
    out.datatype = "INT64"
    out.shape.append(1)
    infer.raw_output_contents.append(struct.pack("<q", 0 if finished else 2))


def _append_int32_scalar_output(
    infer: predict_v2_pb2.ModelInferResponse,
    tensor_name: str,
    value: int,
) -> None:
    """INT32 标量，shape ``[1]``（与客户端 ``_raw_matches_output_metadata`` 一致）。"""
    out = infer.outputs.add()
    out.name = tensor_name
    out.datatype = "INT32"
    out.shape.append(1)
    infer.raw_output_contents.append(struct.pack("<i", int(value)))


def _append_aux_info_metrics_outputs(
    infer: predict_v2_pb2.ModelInferResponse, out_py: Any
) -> None:
    """``prompt_token_num`` = ``AuxInfo.input_len``；``prompt_cached_token_num`` = ``AuxInfo.reuse_len``。"""
    ax = getattr(out_py, "aux_info", None)
    input_len = int(ax.input_len) if ax is not None else 0
    reuse_len = int(ax.reuse_len) if ax is not None else 0
    _append_int32_scalar_output(infer, "prompt_token_num", input_len)
    _append_int32_scalar_output(infer, "prompt_cached_token_num", reuse_len)


def build_stream_response_from_generate_outputs(
    bailian_request_id: str,
    model_name: str,
    go: GenerateOutputs,
    request_log_tag: str,
    request_input_ids: list[int] | None = None,
    return_input_ids: bool = False,
    _request_shape: list[int] | None = None,
) -> predict_v2_pb2.ModelStreamInferResponse:
    """Build ``ModelStreamInferResponse`` from one ``GenerateOutputs`` chunk.

    When ``return_input_ids`` is True, prepends ``prompt_token_ids`` (request ``input_ids``)
    before ``generated_ids`` and ``finish_reason``. After ``finish_reason`` appends
    ``prompt_token_num`` (``AuxInfo.input_len``) and ``prompt_cached_token_num`` (``AuxInfo.reuse_len``).
    Output order is stable across chunks.

    ``bailian_request_id``: wire id for ``ModelInferResponse.id`` / trace string.
    ``request_log_tag``: same format as ``stream_log_tag`` in ``bailian_grpc_real_infer`` (debug logs).

    ``_request_shape`` reserved for future shape alignment.
    """
    del _request_shape  # reserved
    if not go.generate_outputs:
        raise ValueError(
            "build_stream_response_from_generate_outputs expects non-empty go.generate_outputs"
        )
    stream_resp = predict_v2_pb2.ModelStreamInferResponse()
    infer = stream_resp.infer_response
    _set_infer_response_identity(infer, bailian_request_id, model_name)

    out_py = go.generate_outputs[0]
    finished = out_py.finished
    generated_ids = _token_ids_list_from_generate_output(out_py)

    if return_input_ids and request_input_ids is not None:
        _append_prompt_token_ids_output(infer, request_input_ids)

    _append_generated_ids_output(infer, generated_ids)
    _append_finish_reason_output(infer, finished)
    _append_aux_info_metrics_outputs(infer, out_py)

    logging.debug(
        "[BailianGrpc] [%s] generated_ids: %s", request_log_tag, generated_ids
    )
    logging.debug(
        "[BailianGrpc] [%s] return_input_ids=%s prompt_len=%s",
        request_log_tag,
        return_input_ids,
        len(request_input_ids or []),
    )

    return stream_resp

"""DashSc gRPC wire codec: parse ``ModelInferRequest`` tensors; build ``ModelStreamInferResponse``.

Merged from the original ``dash_sc_grpc_request`` / ``_response_real`` / ``_response_fake``
modules so the single public entry is now this file.

Defaults for ``SamplingParams`` align with ``rtp_llm.config.generate_config.GenerateConfig``
(same field names).
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import GenerateOutputs

# ----------------------------------------------------------------------------
# Low-level tensor decoding helpers (shared by request parsing and access log)
# ----------------------------------------------------------------------------


def unpack_int_tensor_flat(datatype: str, raw: bytes | None) -> list[int] | None:
    """Bulk unpack INT32/INT64 little-endian tensor bytes.

    Returns ``None`` if ``raw`` is missing / misaligned / has an unsupported ``datatype``.
    One ``struct.unpack`` call instead of a per-element list comprehension.
    """
    if raw is None or not raw:
        return None
    if datatype == "INT32":
        if len(raw) & 3:
            return None
        n = len(raw) >> 2
        return list(struct.unpack(f"<{n}i", raw)) if n else []
    if datatype == "INT64":
        if len(raw) & 7:
            return None
        n = len(raw) >> 3
        return [int(x) for x in struct.unpack(f"<{n}q", raw)] if n else []
    return None


def _find_input_raw(request, tensor_name: str):
    """Return ``(InferInputTensor | None, raw bytes | None)`` for ``tensor_name``."""
    for i, inp in enumerate(request.inputs):
        if inp.name != tensor_name:
            continue
        if i >= len(request.raw_input_contents):
            return inp, None
        return inp, request.raw_input_contents[i]
    return None, None


def _parse_int_tensor_flat(inp, raw: bytes | None) -> list[int] | None:
    if raw is None:
        return None
    return unpack_int_tensor_flat(inp.datatype, raw)


def _parse_optional_scalar_int(request, tensor_name: str) -> int | None:
    inp, raw = _find_input_raw(request, tensor_name)
    if inp is None or raw is None:
        return None
    ids = _parse_int_tensor_flat(inp, raw)
    if not ids:
        return None
    return int(ids[0])


def _parse_optional_scalar_float(request, tensor_name: str) -> float | None:
    inp, raw = _find_input_raw(request, tensor_name)
    if inp is None or raw is None or not raw:
        return None
    dt = inp.datatype
    if dt == "FP32" and len(raw) >= 4:
        return float(struct.unpack_from("<f", raw, 0)[0])
    if dt == "FP64" and len(raw) >= 8:
        return float(struct.unpack_from("<d", raw, 0)[0])
    # Tolerate integer-typed scalars as floats (e.g. top_p arriving as INT32 1).
    if dt == "INT32" and len(raw) >= 4:
        return float(struct.unpack_from("<i", raw, 0)[0])
    if dt == "INT64" and len(raw) >= 8:
        return float(struct.unpack_from("<q", raw, 0)[0])
    return None


def _parse_stop_words_list_input(request) -> tuple[tuple[int, ...], ...] | None:
    """Input name ``stop_words_list`` -> ``GenerateConfig.stop_words_list`` (groups of token ids)."""
    inp, raw = _find_input_raw(request, "stop_words_list")
    if inp is None or raw is None:
        return None
    flat = _parse_int_tensor_flat(inp, raw)
    if flat is None:
        return None
    shape = [int(x) for x in inp.shape]
    if not shape or (len(shape) == 1 and shape[0] <= 0):
        return tuple()
    if len(shape) == 1:
        return (tuple(flat),) if flat else tuple()
    if len(shape) == 2:
        rows, cols = shape[0], shape[1]
        if rows * cols != len(flat):
            return None
        return tuple(tuple(flat[r * cols : (r + 1) * cols]) for r in range(rows))
    return (tuple(flat),) if flat else tuple()


# ----------------------------------------------------------------------------
# Sampling / Other params (dataclasses consumed by the inference path)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class OtherParams:
    """Non-sampling knobs carried alongside ``input_ids`` (filled by ``parse_other_params``)."""

    return_input_ids: bool = False


@dataclass(frozen=True)
class SamplingParams:
    """Sampling / generation options from ``request.inputs`` (+ legacy ``top_k`` in ``request.parameters``)."""

    max_new_tokens: int = 32000
    num_return_sequences: int = 0
    top_p: float = 1.0
    top_k: int = 0
    temperature: float = 1.0
    min_new_tokens: int = 0
    random_seed: int | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_words_list: tuple[tuple[int, ...], ...] = field(default_factory=tuple)

    @property
    def n(self) -> int:
        """Alias for ``num_return_sequences`` (same as HuggingFace ``n``)."""
        return self.num_return_sequences

    def stop_words_list_py(self) -> list[list[int]]:
        """``GenerateConfig.stop_words_list`` shape: ``List[List[int]]``."""
        return [list(group) for group in self.stop_words_list]

    def to_generate_config(self, *, other: OtherParams | None = None):
        """Build ``GenerateConfig``; ``other`` supplies ``return_input_ids`` etc."""
        from rtp_llm.config.generate_config import GenerateConfig

        return_input_ids = other.return_input_ids if other is not None else False
        return GenerateConfig(
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=self.num_return_sequences,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            min_new_tokens=self.min_new_tokens,
            random_seed=self.random_seed,
            repetition_penalty=self.repetition_penalty,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop_words_list=self.stop_words_list_py(),
            return_input_ids=return_input_ids,
            is_streaming=True,
        )


# ----------------------------------------------------------------------------
# Request parsing: input_ids + sampling + other params
# ----------------------------------------------------------------------------


def parse_input_ids_from_request(request) -> list[int] | None:
    """Read ``input_ids`` (INT32 / INT64, little-endian).

    Returns ``None`` if tensor missing, index mismatch, or unsupported datatype.
    """
    inp, raw = _find_input_raw(request, "input_ids")
    if inp is None or raw is None:
        return None
    return _parse_int_tensor_flat(inp, raw)


def parse_sampling_params(request) -> SamplingParams:
    """Read sampling fields from ``request.inputs``.

    Tensor names: ``max_new_tokens``, ``num_return_sequences``, ``top_p``, ``top_k``,
    ``stop_words_list``, ``temperature``, ``min_new_tokens``, ``seed``,
    ``repetition_penalty``, ``frequency_penalty``, ``presence_penalty``.

    Legacy: if there is no ``top_k`` input, ``request.parameters["top_k"].int64_param``
    is used instead.
    """
    max_new_tokens = 32000
    num_return_sequences = 0
    top_p = 1.0
    top_k = 0
    temperature = 1.0
    min_new_tokens = 0
    random_seed: int | None = None
    repetition_penalty = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
    stop_words_list: tuple[tuple[int, ...], ...] = tuple()

    v = _parse_optional_scalar_int(request, "max_new_tokens")
    if v is not None:
        max_new_tokens = max(0, v)

    v = _parse_optional_scalar_int(request, "num_return_sequences")
    if v is not None:
        num_return_sequences = max(0, v)

    vf = _parse_optional_scalar_float(request, "top_p")
    if vf is not None:
        top_p = vf

    v = _parse_optional_scalar_int(request, "top_k")
    if v is not None:
        top_k = v
    elif "top_k" in request.parameters:
        p = request.parameters["top_k"]
        if p.HasField("int64_param"):
            top_k = int(p.int64_param)

    vf = _parse_optional_scalar_float(request, "temperature")
    if vf is not None:
        temperature = vf

    v = _parse_optional_scalar_int(request, "min_new_tokens")
    if v is not None:
        min_new_tokens = max(0, v)

    v = _parse_optional_scalar_int(request, "seed")
    if v is not None:
        random_seed = v

    vf = _parse_optional_scalar_float(request, "repetition_penalty")
    if vf is not None:
        repetition_penalty = vf

    vf = _parse_optional_scalar_float(request, "frequency_penalty")
    if vf is not None:
        frequency_penalty = vf

    vf = _parse_optional_scalar_float(request, "presence_penalty")
    if vf is not None:
        presence_penalty = vf

    sw = _parse_stop_words_list_input(request)
    if sw is not None:
        stop_words_list = sw

    return SamplingParams(
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        min_new_tokens=min_new_tokens,
        random_seed=random_seed,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop_words_list=stop_words_list,
    )


def parse_other_params(request) -> OtherParams:
    """Non-sampling tensors. Currently: ``return_input_ids`` (BOOL byte or numeric scalar)."""
    return_input_ids = False
    inp, raw = _find_input_raw(request, "return_input_ids")
    if inp is not None and raw:
        if inp.datatype == "BOOL" and len(raw) >= 1:
            return_input_ids = raw[0] != 0
        else:
            v = _parse_optional_scalar_int(request, "return_input_ids")
            if v is not None:
                return_input_ids = v != 0
            else:
                vf = _parse_optional_scalar_float(request, "return_input_ids")
                if vf is not None:
                    return_input_ids = vf != 0.0
    return OtherParams(return_input_ids=return_input_ids)


def parse_dash_sc_grpc_request(
    request,
) -> tuple[list[int] | None, SamplingParams | None, OtherParams | None]:
    """Parse one ``ModelInferRequest``: ``input_ids``, sampling tensors, ``other`` params."""
    ids = parse_input_ids_from_request(request)
    if ids is None:
        return None, None, None
    return ids, parse_sampling_params(request), parse_other_params(request)


# ----------------------------------------------------------------------------
# Response builders (real backend + fake / mock)
# ----------------------------------------------------------------------------


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
    """``prompt_token_ids``: INT32 little-endian, shape ``[1, len]``."""
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
    """``generated_ids``: INT32 little-endian, shape ``[1, len]``.

    When empty, a 4-byte filler (single INT32 ``0``) is appended because
    ``raw_input_contents`` indices must stay aligned with ``outputs``. The consumer
    side (access_log ``_scan_response_outputs``) checks declared ``shape`` so the
    filler byte does not leak into token accumulators.
    """
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


def prepend_to_generated_ids_tensor(
    infer: predict_v2_pb2.ModelInferResponse,
    token_ids: list[int],
) -> bool:
    """Prepend ``token_ids`` to the already-appended ``generated_ids`` tensor on ``infer``.

    Returns ``False`` and leaves ``infer`` untouched when ``token_ids`` is empty, when
    ``generated_ids`` is absent, or when its declared shape is a zero-length / filler
    payload (``shape[-1] <= 0``). On success, re-packs the raw bytes as
    ``token_ids + existing_ids`` (INT32 little-endian) and updates ``shape`` to
    ``[1, len(token_ids) + cur_len]``.
    """
    if not token_ids:
        return False
    for i, out in enumerate(infer.outputs):
        if out.name != "generated_ids":
            continue
        if i >= len(infer.raw_output_contents):
            return False
        shape = list(out.shape)
        cur_len = shape[-1] if shape else 0
        if cur_len <= 0:
            return False
        prefix_raw = struct.pack("<%di" % len(token_ids), *token_ids)
        infer.raw_output_contents[i] = prefix_raw + bytes(infer.raw_output_contents[i])
        out.shape[:] = [1, cur_len + len(token_ids)]
        return True
    return False


def _append_finish_reason_output(
    infer: predict_v2_pb2.ModelInferResponse,
    finished: bool,
) -> None:
    """``finish_reason``: INT64 scalar (``[1]``). finished=0, not finished=2."""
    out = infer.outputs.add()
    out.name = "finish_reason"
    out.datatype = "INT64"
    out.shape.append(1)
    infer.raw_output_contents.append(struct.pack("<q", 0 if finished else 2))


def _append_finished_output(
    infer: predict_v2_pb2.ModelInferResponse,
    finished: bool,
) -> None:
    """``finished``: BOOL scalar (``[1]``), 1 byte."""
    out = infer.outputs.add()
    out.name = "finished"
    out.datatype = "BOOL"
    out.shape.append(1)
    infer.raw_output_contents.append(b"\x01" if finished else b"\x00")


def _append_int32_scalar_output(
    infer: predict_v2_pb2.ModelInferResponse,
    tensor_name: str,
    value: int,
) -> None:
    """INT32 scalar tensor (``[1]``) matching client ``_raw_matches_output_metadata``."""
    out = infer.outputs.add()
    out.name = tensor_name
    out.datatype = "INT32"
    out.shape.append(1)
    infer.raw_output_contents.append(struct.pack("<i", int(value)))


def _append_aux_info_metrics_outputs(
    infer: predict_v2_pb2.ModelInferResponse, out_py: Any
) -> None:
    """``prompt_token_num`` = AuxInfo.input_len; ``prompt_cached_token_num`` = AuxInfo.reuse_len."""
    ax = getattr(out_py, "aux_info", None)
    input_len = int(ax.input_len) if ax is not None else 0
    reuse_len = int(ax.reuse_len) if ax is not None else 0
    _append_int32_scalar_output(infer, "prompt_token_num", input_len)
    _append_int32_scalar_output(infer, "prompt_cached_token_num", reuse_len)


def build_stream_response_from_generate_outputs(
    dash_sc_request_id: str,
    model_name: str,
    go: GenerateOutputs,
    request_log_tag: str,
    request_input_ids: list[int] | None = None,
    return_input_ids: bool = False,
    is_streaming: bool = True,
    _request_shape: list[int] | None = None,
) -> predict_v2_pb2.ModelStreamInferResponse:
    """Build ``ModelStreamInferResponse`` from one ``GenerateOutputs`` chunk.

    When ``return_input_ids`` is True, prepends ``prompt_token_ids`` (request ``input_ids``)
    before ``generated_ids`` and ``finish_reason``. After ``finish_reason`` appends
    ``prompt_token_num`` (``AuxInfo.input_len``) and ``prompt_cached_token_num``
    (``AuxInfo.reuse_len``). Output order is stable across chunks.
    """
    del _request_shape  # reserved for future shape alignment
    if not go.generate_outputs:
        raise ValueError(
            "build_stream_response_from_generate_outputs expects non-empty go.generate_outputs"
        )
    stream_resp = predict_v2_pb2.ModelStreamInferResponse()
    infer = stream_resp.infer_response
    infer.id = dash_sc_request_id
    infer.model_name = model_name

    out_py = go.generate_outputs[0]
    finished = out_py.finished
    generated_ids = _token_ids_list_from_generate_output(out_py)

    if return_input_ids and request_input_ids is not None:
        _append_prompt_token_ids_output(infer, request_input_ids)

    _append_generated_ids_output(infer, generated_ids)
    _append_finish_reason_output(infer, finished)
    _append_finished_output(infer, finished)
    _append_aux_info_metrics_outputs(infer, out_py)
    infer.parameters["incremental_output"].int64_param = 1 if is_streaming else 0

    logging.debug("[DashScGrpc] [%s] generated_ids: %s", request_log_tag, generated_ids)
    logging.debug(
        "[DashScGrpc] [%s] return_input_ids=%s prompt_len=%s is_streaming=%s",
        request_log_tag,
        return_input_ids,
        len(request_input_ids or []),
        is_streaming,
    )
    return stream_resp


def iter_fake_model_stream_infer(
    request,
    input_ids_list: list[int],
    top_k: int,
) -> Iterator[predict_v2_pb2.ModelStreamInferResponse]:
    """Mock: ``generated_ids = input_ids + 100``; single chunk; ``finish_reason=0`` (finished)."""
    del top_k  # unused in fake path
    out_ids = [x + 100 for x in input_ids_list]
    stream_resp = predict_v2_pb2.ModelStreamInferResponse()
    infer = stream_resp.infer_response
    infer.id = request.id
    infer.model_name = request.model_name
    _append_generated_ids_output(infer, out_ids)
    logging.debug("[DashScGrpc] fake out_gen.shape: %s", list(infer.outputs[0].shape))
    _append_finish_reason_output(infer, finished=True)
    _append_finished_output(infer, finished=True)
    infer.parameters["incremental_output"].int64_param = 1
    yield stream_resp

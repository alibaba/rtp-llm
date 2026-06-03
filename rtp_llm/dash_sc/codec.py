"""DashSc gRPC wire codec: parse ``ModelInferRequest`` tensors; build ``ModelStreamInferResponse``.

Merged from the original ``dash_sc_grpc_request`` / ``_response_real`` / ``_response_fake``
modules so the single public entry is now this file.

Defaults for ``SamplingParams`` align with ``rtp_llm.config.generate_config.GenerateConfig``
(same field names).
"""

from __future__ import annotations

import json
import logging
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import GenerateOutputs

_INT32_MAX = 2_147_483_647
_DEFAULT_MAX_NEW_TOKENS = 32000

FINISH_REASON_LENGTH = 1
FINISH_REASON_STOP_ENGINE_PARAM = 8
FINISH_REASON_ABORT = 10
FINISH_REASON_STOP_TIMEOUT = 13

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


def _parse_optional_parameter_int(request, param_name: str) -> int | None:
    """Read a scalar int from ``request.parameters``.

    DashScope-serving usually sends request controls as tensors, but some proxy
    paths put scalar knobs into the Triton ``parameters`` map. Accept both native
    int64 and numeric strings so the hot path does not silently fall back to
    defaults when the wire shape changes.
    """
    if param_name not in request.parameters:
        return None
    p = request.parameters[param_name]
    if p.HasField("int64_param"):
        return int(p.int64_param)
    if p.HasField("string_param"):
        s = str(p.string_param).strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None
    if p.HasField("bool_param"):
        return 1 if p.bool_param else 0
    return None


def _parse_optional_parameter_bool(request, param_name: str) -> bool | None:
    if param_name not in request.parameters:
        return None
    p = request.parameters[param_name]
    if p.HasField("bool_param"):
        return bool(p.bool_param)
    if p.HasField("int64_param"):
        return _parse_optional_bool(p.int64_param)
    if p.HasField("string_param"):
        return _parse_optional_bool(p.string_param)
    return None


def _parse_optional_parameter_string(request, param_name: str) -> str | None:
    if param_name not in request.parameters:
        return None
    p = request.parameters[param_name]
    if not p.HasField("string_param"):
        return None
    value = str(p.string_param).strip()
    return value or None


def _parse_optional_int_value(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def _parse_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return None


def _parse_ds_header_attributes(request) -> dict[str, Any]:
    """Parse ``ds_header_attributes`` into a lower-case-key dict.

    The value is a JSON string produced by dashscope-serving. Returning an empty
    dict on malformed input preserves inference while keeping the parser
    defensive against partial or legacy requests.
    """
    if "ds_header_attributes" not in request.parameters:
        return {}
    p = request.parameters["ds_header_attributes"]
    if not p.HasField("string_param") or not p.string_param:
        return {}
    try:
        attrs = json.loads(p.string_param)
    except Exception as e:
        logging.warning("failed to parse ds_header_attributes: %s", e)
        return {}
    if not isinstance(attrs, dict):
        return {}
    return {str(k).lower(): v for k, v in attrs.items()}


def _dict_get_case_insensitive(value: Any, key: str) -> Any:
    if not isinstance(value, dict):
        return None
    lower_key = key.lower()
    for k, v in value.items():
        if str(k).lower() == lower_key:
            return v
    return None


def _nested_get_case_insensitive(value: Any, path: tuple[str, ...]) -> Any:
    cur = value
    for key in path:
        cur = _dict_get_case_insensitive(cur, key)
        if cur is None:
            return None
    return cur


def _lookup_ds_request_control(attrs: dict[str, Any], name: str) -> Any:
    """Find a Dash request control in common ds_header_attributes layouts."""
    direct = _dict_get_case_insensitive(attrs, name)
    if direct is not None:
        return direct
    for prefix in (
        ("parameters",),
        ("body",),
        ("body", "parameters"),
        ("payload",),
        ("payload", "parameters"),
        ("request",),
        ("request", "parameters"),
    ):
        value = _nested_get_case_insensitive(attrs, prefix + (name,))
        if value is not None:
            return value
    return None


def _is_openai_compatible_request(request) -> bool:
    attrs = _parse_ds_header_attributes(request)
    path = str(attrs.get("x-envoy-original-path", "")).lower()
    raw_path = str(attrs.get("x-dashscope-inner-rawhttppath", "")).lower()
    baggage = str(attrs.get("baggage", "")).lower()
    return any(
        marker in text
        for text in (path, raw_path, baggage)
        for marker in ("/compatible-mode/", "/api-openai/")
    )


def _normalize_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _jsonable_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return _normalize_non_empty_str(value)


_REASONING_EFFORT_ALIASES: dict[str, str | None] = {
    "none": None,
    "minimum": "minimum",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "xhigh",
}


def _extract_reasoning_effort_value(value: Any) -> str | None:
    """Read Dash/OpenAI reasoning_effort shapes and normalize max -> xhigh."""
    if value is None:
        return None
    parsed = value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = s
    if isinstance(parsed, dict):
        nested = _dict_get_case_insensitive(parsed, "effort")
        if nested is None:
            nested = _dict_get_case_insensitive(parsed, "reasoning_effort")
        return _extract_reasoning_effort_value(nested)
    if isinstance(parsed, list):
        for item in parsed:
            normalized = _extract_reasoning_effort_value(item)
            if normalized is not None:
                return normalized
        return None
    effort = str(parsed).strip().lower()
    return _REASONING_EFFORT_ALIASES.get(effort)


def _parse_optional_parameter_reasoning_effort(request, param_name: str) -> str | None:
    if param_name not in request.parameters:
        return None
    p = request.parameters[param_name]
    if p.HasField("string_param"):
        return _extract_reasoning_effort_value(p.string_param)
    return None


def _normalize_response_format_value(value: Any) -> str | None:
    """Normalize Dash wire shapes to the OpenAI response_format object contract."""
    if value is None:
        return None
    parsed = value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except Exception:
            return s
    if isinstance(parsed, list):
        if len(parsed) == 1:
            parsed = parsed[0]
        else:
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return _jsonable_to_string(parsed)


def _normalize_guided_json_response_format(value: Any) -> str | None:
    """Normalize Dash/dashllm guided_json to response_format json_schema."""
    if value is None:
        return None
    schema = value
    if isinstance(schema, str):
        s = schema.strip()
        if not s:
            return None
        try:
            schema = json.loads(s)
        except Exception:
            schema = s
    if isinstance(schema, list):
        if not schema:
            return None
        schema = schema[0]
        if isinstance(schema, str):
            s = schema.strip()
            if not s:
                return None
            try:
                schema = json.loads(s)
            except Exception:
                schema = s
    response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
    return json.dumps(response_format, ensure_ascii=False, separators=(",", ":"))


def _parse_json_format_controls(request) -> tuple[str | None, bool]:
    response_format = _parse_optional_parameter_string(request, "response_format")
    guided_json = _parse_optional_parameter_string(request, "guided_json")
    json_format_value = _parse_optional_parameter_bool(request, "json_format")
    ds_attrs = _parse_ds_header_attributes(request)

    response_format = _normalize_response_format_value(response_format)
    if response_format is None:
        response_format = _normalize_response_format_value(
            _lookup_ds_request_control(ds_attrs, "response_format")
        )

    guided_response_format = _normalize_guided_json_response_format(guided_json)
    if guided_response_format is None:
        guided_response_format = _normalize_guided_json_response_format(
            _lookup_ds_request_control(ds_attrs, "guided_json")
        )
    if guided_response_format is not None:
        response_format = guided_response_format

    if json_format_value is None:
        json_format_value = _parse_optional_bool(
            _lookup_ds_request_control(ds_attrs, "json_format")
        )
    return response_format, bool(json_format_value)


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
    enable_thinking: bool | None = None
    max_new_think_tokens: int | None = None
    timeout_ms: int | None = None
    traffic_reject_priority: int | None = None
    reasoning_effort: str | None = None
    request_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingParams:
    """Sampling / generation options from ``request.inputs`` (+ legacy ``top_k`` in ``request.parameters``)."""

    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
    max_new_tokens_from_completion_alias: bool = False
    max_total_tokens: int | None = None
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
    max_new_think_tokens: int | None = None
    response_format: str | None = None
    json_format: bool = False

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
        request_max_think = self.max_new_think_tokens
        if request_max_think is None and other is not None:
            request_max_think = other.max_new_think_tokens
        if request_max_think is None:
            max_thinking_tokens = 32000
        elif request_max_think < 0:
            max_thinking_tokens = _INT32_MAX
        else:
            max_thinking_tokens = request_max_think
        backend_max_new_tokens = self.max_new_tokens
        if (
            other is not None
            and self.max_new_tokens_from_completion_alias
            and backend_max_new_tokens > 0
        ):
            if self.max_total_tokens is not None and self.max_total_tokens > 0:
                backend_max_new_tokens = min(
                    backend_max_new_tokens, int(self.max_total_tokens)
                )
        return GenerateConfig(
            max_new_tokens=backend_max_new_tokens,
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
            max_thinking_tokens=max_thinking_tokens,
            return_input_ids=return_input_ids,
            is_streaming=True,
            response_format=self.response_format,
            json_format=self.json_format,
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

    Tensor names: ``max_completion_tokens`` (or legacy ``max_new_tokens`` /
    ``max_tokens``), ``num_return_sequences`` (or DashScope alias ``n``),
    ``top_p``, ``top_k``, ``stop_words_list``, ``temperature``,
    ``min_new_tokens`` (or DashScope alias ``min_length``), ``seed``,
    ``repetition_penalty``, ``frequency_penalty``, ``presence_penalty``,
    ``max_new_think_tokens`` / ``max_think_length``.

    Legacy: if there is no ``top_k`` input, ``request.parameters["top_k"].int64_param``
    is used instead.
    """
    max_new_tokens = _DEFAULT_MAX_NEW_TOKENS
    max_new_tokens_from_completion_alias = False
    max_total_tokens: int | None = None
    num_return_sequences = 0
    top_p = 1.0
    top_k = 0
    temperature = 1.0
    min_new_tokens = 0
    random_seed: int | None = None
    repetition_penalty = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
    max_new_think_tokens: int | None = None
    stop_words_list: tuple[tuple[int, ...], ...] = tuple()
    openai_compatible_request = _is_openai_compatible_request(request)

    v = _parse_optional_scalar_int(request, "max_tokens")
    if v is None:
        v = _parse_optional_parameter_int(request, "max_tokens")
    if v is not None and v > 0:
        max_total_tokens = v

    v = _parse_optional_scalar_int(request, "max_completion_tokens")
    if v is None:
        v = _parse_optional_parameter_int(request, "max_completion_tokens")
    if v is not None:
        if v > 0:
            max_new_tokens = v
            max_new_tokens_from_completion_alias = True
        # Compatible-mode max_completion_tokens values <= 0 mean "unset".
        # Keep the server default and do not fall through to legacy aliases.
    else:
        legacy_max_new_tokens = _parse_optional_scalar_int(request, "max_new_tokens")
        if legacy_max_new_tokens is None:
            v = _parse_optional_scalar_int(request, "max_tokens")
        else:
            v = legacy_max_new_tokens
        if v is None:
            legacy_max_new_tokens = _parse_optional_parameter_int(
                request, "max_new_tokens"
            )
            v = legacy_max_new_tokens
        if v is None:
            v = _parse_optional_parameter_int(request, "max_tokens")
        if (
            legacy_max_new_tokens is not None
            and legacy_max_new_tokens <= 0
            and openai_compatible_request
        ):
            pass
        elif v is not None:
            max_new_tokens = v

    v = _parse_optional_scalar_int(request, "num_return_sequences")
    if v is None:
        v = _parse_optional_scalar_int(request, "n")
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
    if v is None:
        v = _parse_optional_scalar_int(request, "min_length")
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

    for tensor_name in ("max_think_length", "max_new_think_tokens"):
        v = _parse_optional_scalar_int(request, tensor_name)
        if v is not None:
            max_new_think_tokens = v
            break

    sw = _parse_stop_words_list_input(request)
    if sw is not None:
        stop_words_list = sw

    response_format, json_format = _parse_json_format_controls(request)

    return SamplingParams(
        max_new_tokens=max_new_tokens,
        max_new_tokens_from_completion_alias=max_new_tokens_from_completion_alias,
        max_total_tokens=max_total_tokens,
        num_return_sequences=num_return_sequences,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        min_new_tokens=min_new_tokens,
        random_seed=random_seed,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_new_think_tokens=max_new_think_tokens,
        stop_words_list=stop_words_list,
        response_format=response_format,
        json_format=json_format,
    )


def parse_other_params(request) -> OtherParams:
    """Parse non-sampling request controls.

    ``ds_header_attributes`` carries DashScope request-scoped controls that need
    backend effects (thinking switch, timeout, priority, scheduler headers) even
    though they are not ordinary sampler behavior.
    """
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

    ds_attrs = _parse_ds_header_attributes(request)
    enable_thinking = _parse_optional_bool(
        _lookup_ds_request_control(ds_attrs, "x-ds-llm-thinking")
    )
    if enable_thinking is None:
        enable_thinking = _parse_optional_bool(
            _lookup_ds_request_control(ds_attrs, "enable_thinking")
        )
    if enable_thinking is None:
        enable_thinking = _parse_optional_parameter_bool(request, "enable_thinking")

    max_new_think_tokens = _parse_optional_scalar_int(request, "max_new_think_tokens")
    if max_new_think_tokens is None:
        max_new_think_tokens = _parse_optional_parameter_int(
            request, "max_new_think_tokens"
        )
    if max_new_think_tokens is None:
        max_new_think_tokens = _parse_optional_int_value(
            _lookup_ds_request_control(ds_attrs, "thinking_budget")
        )
    if max_new_think_tokens is None:
        max_new_think_tokens = _parse_optional_parameter_int(request, "thinking_budget")
    if max_new_think_tokens is not None:
        max_new_think_tokens = int(max_new_think_tokens)

    reasoning_effort = _parse_optional_parameter_reasoning_effort(
        request, "reasoning_effort"
    )
    if reasoning_effort is None:
        reasoning_effort = _extract_reasoning_effort_value(
            _lookup_ds_request_control(ds_attrs, "reasoning_effort")
        )

    timeout_s = _parse_optional_int_value(ds_attrs.get("x-dashscope-inner-timeout"))
    timeout_ms = timeout_s * 1000 if timeout_s is not None and timeout_s > 0 else None

    traffic_reject_priority = _parse_optional_int_value(
        ds_attrs.get("x-ds-request-priority")
    )
    if traffic_reject_priority is None:
        traffic_reject_priority = _parse_optional_int_value(
            ds_attrs.get("x-dashscope-inner-request-priority")
        )

    request_headers: dict[str, str] = {}
    for header_name in ("user_id", "x-dashscope-apikeyid"):
        value = _normalize_non_empty_str(ds_attrs.get(header_name))
        if value is not None:
            request_headers[header_name] = value

    return OtherParams(
        return_input_ids=return_input_ids,
        enable_thinking=enable_thinking,
        max_new_think_tokens=max_new_think_tokens,
        timeout_ms=timeout_ms,
        traffic_reject_priority=traffic_reject_priority,
        reasoning_effort=reasoning_effort,
        request_headers=request_headers,
    )


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


def _first_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, list):
        return int(value[0]) if value else None
    if isinstance(value, tuple):
        return int(value[0]) if value else None
    return int(value)


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
    finish_reason_override: int | None = None,
) -> None:
    """``finish_reason``: INT64 scalar (``[1]``). finished=0, not finished=2."""
    out = infer.outputs.add()
    out.name = "finish_reason"
    out.datatype = "INT64"
    out.shape.append(1)
    if finish_reason_override is not None:
        value = finish_reason_override
    else:
        value = 0 if finished else 2
    infer.raw_output_contents.append(struct.pack("<q", value))


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


def _append_dashllm_limit_parameters(
    infer: predict_v2_pb2.ModelInferResponse,
    *,
    generate_config: Any = None,
    eos_token_id: int | None = None,
    max_token_id: int | None = None,
    generate_think_token_num: int | None = None,
) -> None:
    """Mirror dashllm response parameters consumed by dashscope-serving."""
    if generate_config is not None:
        infer.parameters["max_new_tokens"].int64_param = int(
            getattr(generate_config, "max_new_tokens", 0) or 0
        )
        max_think = int(getattr(generate_config, "max_thinking_tokens", 0) or 0)
        if max_think > 0:
            infer.parameters["max_new_think_tokens"].int64_param = max_think

    stop_id = _first_int(eos_token_id)
    if stop_id is not None:
        infer.parameters["stop_token_id"].int64_param = stop_id
    if max_token_id is not None:
        infer.parameters["max_token_id"].int64_param = int(max_token_id)
    if generate_think_token_num is not None:
        infer.parameters["generate_think_token_num"].int64_param = int(
            generate_think_token_num
        )


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


def _append_prompt_cache_usage_parameters(
    infer: predict_v2_pb2.ModelInferResponse,
    prompt_tokens: int,
    cached_tokens: int,
) -> None:
    prompt_tokens = int(prompt_tokens)
    cached_tokens = int(cached_tokens)
    # dashllm converts response parameters to extra_params, then emits
    # output_body["prompt_cached_token_num"]. DashScope uses that field to build
    # usage.prompt_tokens_details.cached_tokens.
    infer.parameters["prompt_token_num"].int64_param = prompt_tokens
    infer.parameters["prompt_cached_token_num"].int64_param = cached_tokens


def _append_aux_info_metrics_outputs(
    infer: predict_v2_pb2.ModelInferResponse,
    out_py: Any,
    prompt_token_fallback: int = 0,
) -> None:
    """``prompt_token_num`` = AuxInfo.input_len; ``prompt_cached_token_num`` = AuxInfo.reuse_len."""
    ax = getattr(out_py, "aux_info", None)
    input_len = int(ax.input_len) if ax is not None else int(prompt_token_fallback)
    reuse_len = int(ax.reuse_len) if ax is not None else 0
    _append_int32_scalar_output(infer, "prompt_token_num", input_len)
    _append_int32_scalar_output(infer, "prompt_cached_token_num", reuse_len)
    _append_prompt_cache_usage_parameters(infer, input_len, reuse_len)


def build_stream_response_from_generate_outputs(
    dash_sc_request_id: str,
    model_name: str,
    go: GenerateOutputs,
    request_log_tag: str,
    request_input_ids: list[int] | None = None,
    return_input_ids: bool = False,
    is_streaming: bool = True,
    generate_config: Any = None,
    eos_token_id: int | None = None,
    max_token_id: int | None = None,
    generate_think_token_num: int | None = None,
    finish_reason_override: int | None = None,
    _request_shape: list[int] | None = None,
    *,
    stream_finished: bool | None = None,
    token_ids: list[int] | None = None,
) -> predict_v2_pb2.ModelStreamInferResponse:
    """Build ``ModelStreamInferResponse`` from one ``GenerateOutputs`` chunk.

    ``stream_finished``: if provided, overrides ``out_py.finished`` for the wire
    ``finished`` / ``finish_reason`` fields. Use when the servicer knows the gRPC
    stream is not done yet (e.g. phase-2 will follow).

    ``token_ids``: if provided, overrides the generated_ids from ``out_py``.
    Use when the servicer rewrites the token payload (e.g. injecting </think>).
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
    finished = stream_finished if stream_finished is not None else out_py.finished
    generated_ids = (
        token_ids
        if token_ids is not None
        else _token_ids_list_from_generate_output(out_py)
    )

    if return_input_ids and request_input_ids is not None:
        _append_prompt_token_ids_output(infer, request_input_ids)

    _append_generated_ids_output(infer, generated_ids)
    _append_finish_reason_output(infer, finished, finish_reason_override)
    _append_finished_output(infer, finished)
    _append_aux_info_metrics_outputs(
        infer,
        out_py,
        prompt_token_fallback=len(request_input_ids or []),
    )
    infer.parameters["incremental_output"].int64_param = 1 if is_streaming else 0
    _append_dashllm_limit_parameters(
        infer,
        generate_config=generate_config,
        eos_token_id=eos_token_id,
        max_token_id=max_token_id,
        generate_think_token_num=generate_think_token_num,
    )

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


def build_finish_reason_done_response(
    dash_sc_request_id: str,
    model_name: str,
    finish_reason: int,
) -> predict_v2_pb2.ModelStreamInferResponse:
    """Build a terminal response with empty content and explicit finish_reason.

    Used when the engine stops for a non-error reason (timeout, abort) that
    should NOT be reported as a 5xx to the upstream. The finish_reason integer
    aligns with DashLLM's LLMFinishReason enum so dashscope-serving can map it
    correctly (e.g. 13=STOP_TIMEOUT → 200 + X-DashScope-PartialResponse).
    """
    stream_resp = predict_v2_pb2.ModelStreamInferResponse()
    infer = stream_resp.infer_response
    infer.id = dash_sc_request_id
    infer.model_name = model_name
    _append_generated_ids_output(infer, [])
    _append_finish_reason_output(
        infer, finished=True, finish_reason_override=finish_reason
    )
    _append_finished_output(infer, finished=True)
    infer.parameters["incremental_output"].int64_param = 1
    return stream_resp


def build_parameter_error_response(
    request_id: str, message: str, *, status_code: int = 400
) -> predict_v2_pb2.ModelStreamInferResponse:
    """Build a DashLLM-protocol error via ``__messages__`` (gateway reads status_code from here)."""
    resp = predict_v2_pb2.ModelStreamInferResponse()
    resp.infer_response.parameters["__messages__"].string_param = json.dumps(
        {
            "header": {
                "status_code": status_code,
                "status_name": "InvalidParameter",
                "status_message": message,
                "finished": True,
                "request_id": request_id,
            },
            "payload": {},
        }
    )
    return resp

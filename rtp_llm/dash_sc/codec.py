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

import torch

from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.structural_tag import (
    DashScStructuralTagError,
    adapt_dashscope_tool_call_wrapper_to_tag,
    structural_tag_from_response_format,
    validate_structural_tag_shape,
)
from rtp_llm.utils.base_model_datatypes import GenerateOutputs

_DEFAULT_MAX_THINKING_TOKENS = 131072
_DEFAULT_MAX_NEW_TOKENS = 131072

FINISH_REASON_LENGTH = 1
FINISH_REASON_STOP_ENGINE_PARAM = 8
FINISH_REASON_ABORT = 10
FINISH_REASON_STOP_TIMEOUT = 13
FINISH_REASON_USE_PARAMETER_STATUS = 1000


class DashScParameterError(ValueError):
    """Explicit user-parameter parse/validation error for dash-sc gRPC."""


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


def _parse_num_return_sequences_parameter_fallback(
    request, ds_attrs: dict[str, Any]
) -> int | None:
    """Read the non-tensor ``num_return_sequences`` / ``n`` fallback.

    Input tensors are the established Dash wire contract and retain their
    existing precedence.  Direct Triton parameters were added as a compatibility
    fallback for OpenAI-style logprob requests, so reject conflicting aliases
    instead of silently choosing one.  Nested ``ds_header_attributes`` keeps the
    same alias priority as before.
    """
    parameter_values = {
        name: _parse_optional_parameter_int(request, name)
        for name in ("num_return_sequences", "n")
        if name in request.parameters
    }
    parsed_parameter_values = {
        name: value for name, value in parameter_values.items() if value is not None
    }
    if len(set(parsed_parameter_values.values())) > 1:
        raise DashScParameterError("conflicting n and num_return_sequences parameters")
    for name in ("num_return_sequences", "n"):
        if name in parsed_parameter_values:
            return parsed_parameter_values[name]

    value = _parse_optional_int_value(
        _lookup_ds_request_control(ds_attrs, "num_return_sequences")
    )
    if value is None:
        value = _parse_optional_int_value(_lookup_ds_request_control(ds_attrs, "n"))
    return value


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


def _is_single_value_tensor_shape(shape: Any) -> bool:
    """Return whether a tensor shape contains exactly one scalar value.

    Triton clients use both rank-0 ``[]`` and batch-wrapped ``[1, 1]`` shapes
    for scalar controls in addition to ``[1]``.  All of them carry one value;
    dimensions other than one would make the control ambiguous.
    """
    return all(int(dim) == 1 for dim in shape)


def _parse_strict_bool_control(
    request,
    ds_attrs: dict[str, Any],
    *names: str,
) -> tuple[bool | None, bool]:
    """Read an optional boolean request control without silently coercing errors.

    The DashScope bridge has shipped the same OpenAI-compatible controls as input
    tensors, Triton parameters, and nested ``ds_header_attributes`` values.  The
    second return value records explicit presence so callers can distinguish an
    omitted value from an invalid/false value.
    """
    for name in names:
        inp, raw = _find_input_raw(request, name)
        if inp is not None:
            if raw is None:
                raise DashScParameterError(f"invalid {name}: missing tensor payload")
            if not _is_single_value_tensor_shape(inp.shape):
                raise DashScParameterError(
                    f"invalid {name}: tensor must contain exactly one value, "
                    f"got shape {list(inp.shape)}"
                )
            if inp.datatype == "BOOL" and len(raw) == 1 and raw[0] in (0, 1):
                return raw[0] == 1, True
            if inp.datatype == "INT32" and len(raw) == 4:
                value = struct.unpack("<i", raw)[0]
                if value in (0, 1):
                    return bool(value), True
            if inp.datatype == "INT64" and len(raw) == 8:
                value = struct.unpack("<q", raw)[0]
                if value in (0, 1):
                    return bool(value), True
            raise DashScParameterError(f"invalid {name}: must be a boolean scalar")

        if name in request.parameters:
            param = request.parameters[name]
            if param.HasField("bool_param"):
                return bool(param.bool_param), True
            if param.HasField("int64_param"):
                value = int(param.int64_param)
                if value in (0, 1):
                    return bool(value), True
            elif param.HasField("string_param"):
                value = _parse_optional_bool(param.string_param)
                if value is not None:
                    return value, True
            raise DashScParameterError(f"invalid {name}: must be a boolean")

        raw_value = _lookup_ds_request_control(ds_attrs, name)
        if raw_value is not None:
            if isinstance(raw_value, bool):
                value = raw_value
            elif isinstance(raw_value, int) and raw_value in (0, 1):
                value = bool(raw_value)
            elif isinstance(raw_value, str):
                value = _parse_optional_bool(raw_value)
            else:
                value = None
            if value is None:
                raise DashScParameterError(f"invalid {name}: must be a boolean")
            return value, True
    return None, False


def _parse_strict_int_control(
    request,
    ds_attrs: dict[str, Any],
    *names: str,
) -> tuple[int | None, bool]:
    """Read an optional integer request control, rejecting bool/float values."""
    for name in names:
        inp, raw = _find_input_raw(request, name)
        if inp is not None:
            if raw is None:
                raise DashScParameterError(f"invalid {name}: missing tensor payload")
            if not _is_single_value_tensor_shape(inp.shape):
                raise DashScParameterError(
                    f"invalid {name}: tensor must contain exactly one value, "
                    f"got shape {list(inp.shape)}"
                )
            if inp.datatype == "INT32" and len(raw) == 4:
                return int(struct.unpack("<i", raw)[0]), True
            if inp.datatype == "INT64" and len(raw) == 8:
                return int(struct.unpack("<q", raw)[0]), True
            raise DashScParameterError(f"invalid {name}: must be an integer scalar")

        if name in request.parameters:
            param = request.parameters[name]
            if param.HasField("int64_param"):
                return int(param.int64_param), True
            if param.HasField("string_param"):
                value = str(param.string_param).strip()
                try:
                    return int(value), True
                except (TypeError, ValueError):
                    pass
            raise DashScParameterError(f"invalid {name}: must be an integer")

        raw_value = _lookup_ds_request_control(ds_attrs, name)
        if raw_value is not None:
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                if isinstance(raw_value, str):
                    try:
                        return int(raw_value.strip()), True
                    except (TypeError, ValueError):
                        pass
                raise DashScParameterError(f"invalid {name}: must be an integer")
            return int(raw_value), True
    return None, False


def parse_ds_header_attributes(request) -> dict[str, Any]:
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


def _is_openai_compatible_request(
    request, ds_attrs: dict[str, Any] | None = None
) -> bool:
    attrs = ds_attrs if ds_attrs is not None else parse_ds_header_attributes(request)
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


def _parse_response_format_value(value: Any) -> Any:
    """Parse Dash wire shapes to the OpenAI response_format object contract."""
    if value is None:
        return None
    parsed = value
    while isinstance(parsed, (str, list)):
        if isinstance(parsed, list):
            if not parsed:
                return None
            parsed = parsed[0]
            continue
        s = parsed.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except Exception:
            return s
    return parsed


def _parse_guided_json_response_format(value: Any) -> dict[str, Any] | None:
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
    return {"type": "json_schema", "json_schema": {"schema": schema}}


def _decode_structural_tag_payload(
    value: Any, field_name: str
) -> dict[str, Any] | None:
    if value is None:
        return None
    while isinstance(value, (str, list)):
        if isinstance(value, list):
            if not value:
                return None
            value = value[0]
            continue
        s = value.strip()
        if not s:
            return None
        try:
            value = json.loads(s)
        except Exception:
            raise DashScParameterError(f"invalid {field_name}") from None
    if not isinstance(value, dict):
        raise DashScParameterError(f"invalid {field_name}: $ must be an object")
    try:
        validate_structural_tag_shape(value, field_name)
    except DashScStructuralTagError as e:
        raise DashScParameterError(str(e)) from None
    return adapt_dashscope_tool_call_wrapper_to_tag(value)


def _parse_grammar_controls(
    request, ds_attrs: dict[str, Any] | None = None
) -> tuple[str | None, bool, str | None]:
    response_format = _parse_optional_parameter_string(request, "response_format")
    guided_json = _parse_optional_parameter_string(request, "guided_json")
    json_format_value = _parse_optional_parameter_bool(request, "json_format")
    ds_attrs = ds_attrs if ds_attrs is not None else parse_ds_header_attributes(request)

    response_format = _parse_response_format_value(response_format)
    if response_format is None:
        response_format = _parse_response_format_value(
            _lookup_ds_request_control(ds_attrs, "response_format")
        )

    guided_response_format = _parse_guided_json_response_format(guided_json)
    if guided_response_format is None:
        guided_response_format = _parse_guided_json_response_format(
            _lookup_ds_request_control(ds_attrs, "guided_json")
        )
    if guided_response_format is not None:
        response_format = guided_response_format

    if json_format_value is None:
        json_format_value = _parse_optional_bool(
            _lookup_ds_request_control(ds_attrs, "json_format")
        )

    raw_structural_tag = _parse_optional_parameter_string(
        request, "tool_call_structural_tag"
    )
    if raw_structural_tag is None:
        raw_structural_tag = _parse_optional_parameter_string(request, "structural_tag")
    if raw_structural_tag is None:
        raw_structural_tag = _lookup_ds_request_control(
            ds_attrs, "tool_call_structural_tag"
        )
    if raw_structural_tag is None:
        raw_structural_tag = _lookup_ds_request_control(ds_attrs, "structural_tag")

    structural_tag = _decode_structural_tag_payload(
        raw_structural_tag, "tool_call_structural_tag"
    )

    if isinstance(response_format, dict):
        if response_format.get("type") == "structural_tag":
            try:
                response_structural_tag = structural_tag_from_response_format(
                    response_format, "response_format"
                )
            except DashScStructuralTagError as e:
                raise DashScParameterError(str(e)) from None
            response_structural_tag = adapt_dashscope_tool_call_wrapper_to_tag(
                response_structural_tag
            )
            if structural_tag is None:
                structural_tag = response_structural_tag
            response_format = None

    return (
        _jsonable_to_string(response_format),
        bool(json_format_value),
        _jsonable_to_string(structural_tag),
    )


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


def parse_max_new_tokens_for_proxy(request) -> tuple[int, bool]:
    """Read only max-token controls; proxy hot path must not parse grammar."""
    max_new_tokens, from_completion_alias, _ = _parse_max_token_limits(request)
    return max_new_tokens, from_completion_alias


def _parse_max_token_limits(
    request, ds_attrs: dict[str, Any] | None = None
) -> tuple[int, bool, int | None]:
    max_new_tokens = _DEFAULT_MAX_NEW_TOKENS
    max_new_tokens_from_completion_alias = False
    max_total_tokens: int | None = None

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
        else:
            # Preserve the invalid alias value so dash-sc can reject it before
            # it reaches the engine as max_new_tokens=0/-1 and becomes a 500.
            max_new_tokens = v
        max_new_tokens_from_completion_alias = True
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
        if legacy_max_new_tokens is not None and legacy_max_new_tokens <= 0:
            if ds_attrs is None:
                ds_attrs = parse_ds_header_attributes(request)
            if _is_openai_compatible_request(request, ds_attrs):
                pass
            else:
                max_new_tokens = v if v is not None else max_new_tokens
        elif v is not None:
            max_new_tokens = v

    return max_new_tokens, max_new_tokens_from_completion_alias, max_total_tokens


@dataclass(frozen=True)
class OtherParams:
    """Non-sampling knobs carried alongside ``input_ids`` (filled by ``parse_other_params``)."""

    return_input_ids: bool = False
    enable_thinking: bool | None = None
    max_new_think_tokens: int | None = None
    timeout_ms: int | None = None
    traffic_reject_priority: int | None = None
    reasoning_effort: str | None = None
    debug: bool = False
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
    return_logprobs: bool = False
    top_logprobs: int = 0
    stop_words_list: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    max_new_think_tokens: int | None = None
    response_format: str | None = None
    json_format: bool = False
    structural_tag: str | None = None

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
        max_thinking_tokens = (
            _DEFAULT_MAX_THINKING_TOKENS
            if request_max_think is None
            else request_max_think
        )
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
            return_logprobs=self.return_logprobs,
            top_logprobs=self.top_logprobs,
            stop_words_list=self.stop_words_list_py(),
            max_thinking_tokens=max_thinking_tokens,
            return_input_ids=return_input_ids,
            is_streaming=True,
            response_format=self.response_format,
            json_format=self.json_format,
            structural_tag=self.structural_tag,
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


def parse_sampling_params(
    request, ds_attrs: dict[str, Any] | None = None
) -> SamplingParams:
    """Read sampling fields from ``request.inputs``.

    Tensor names: ``max_completion_tokens`` (or legacy ``max_new_tokens`` /
    ``max_tokens``), ``num_return_sequences`` (or DashScope alias ``n``),
    ``top_p``, ``top_k``, ``stop_words_list``, ``temperature``,
    ``min_new_tokens`` (or DashScope alias ``min_length``), ``seed``,
    ``repetition_penalty``, ``frequency_penalty``, ``presence_penalty``,
    ``logprobs`` / ``return_logprobs``, ``top_logprobs``,
    ``max_new_think_tokens`` / ``max_think_length``.

    Legacy: if there is no ``top_k`` input, ``request.parameters["top_k"].int64_param``
    is used instead.
    """
    num_return_sequences = 0
    top_p = 1.0
    top_k = 0
    temperature = 1.0
    min_new_tokens = 0
    random_seed: int | None = None
    repetition_penalty = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
    return_logprobs = False
    top_logprobs = 0
    max_new_think_tokens: int | None = None
    stop_words_list: tuple[tuple[int, ...], ...] = tuple()
    ds_attrs = ds_attrs if ds_attrs is not None else parse_ds_header_attributes(request)
    (
        max_new_tokens,
        max_new_tokens_from_completion_alias,
        max_total_tokens,
    ) = _parse_max_token_limits(request, ds_attrs)

    tensor_num_return_sequences = _parse_optional_scalar_int(
        request, "num_return_sequences"
    )
    if tensor_num_return_sequences is None:
        tensor_num_return_sequences = _parse_optional_scalar_int(request, "n")
    parameter_num_return_sequences: int | None = None
    if tensor_num_return_sequences is not None:
        # Tensor controls are the pre-existing contract and continue to map to
        # GenerateConfig regardless of the logprobs switch.
        num_return_sequences = max(0, tensor_num_return_sequences)
    else:
        parameter_num_return_sequences = _parse_num_return_sequences_parameter_fallback(
            request, ds_attrs
        )
        if parameter_num_return_sequences is not None:
            parameter_num_return_sequences = max(0, parameter_num_return_sequences)
            # Dash encodes only generate_outputs[0].  Never let a parameter
            # fallback silently create backend results that cannot be returned.
            if parameter_num_return_sequences > 1:
                raise DashScParameterError("DashScope response does not support n > 1")

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

    parsed_logprobs, logprobs_was_set = _parse_strict_bool_control(
        request, ds_attrs, "logprobs", "return_logprobs"
    )
    if logprobs_was_set:
        return_logprobs = bool(parsed_logprobs)
    parsed_top_logprobs, top_logprobs_was_set = _parse_strict_int_control(
        request, ds_attrs, "top_logprobs"
    )
    if top_logprobs_was_set:
        assert parsed_top_logprobs is not None
        top_logprobs = parsed_top_logprobs
        if not 0 <= top_logprobs <= 20:
            raise DashScParameterError("top_logprobs must be between 0 and 20")
        # DashScope serving always sends the disabled defaults together as
        # ``logprobs=false, top_logprobs=0``.  Zero requests no candidates and
        # must preserve the legacy non-logprobs path; only a positive K needs
        # the feature to be enabled explicitly.
        if top_logprobs > 0 and not return_logprobs:
            raise DashScParameterError("top_logprobs requires logprobs=true")
    if parameter_num_return_sequences is not None and return_logprobs:
        # Before logprob support, request.parameters did not control backend
        # fan-out.  Preserve that behavior for ordinary requests while allowing
        # OpenAI-style logprob requests to carry their explicit single-result n.
        num_return_sequences = parameter_num_return_sequences
    if return_logprobs and num_return_sequences > 1:
        raise DashScParameterError("logprobs does not support n > 1")

    for tensor_name in ("max_think_length", "max_new_think_tokens"):
        v = _parse_optional_scalar_int(request, tensor_name)
        if v is not None:
            max_new_think_tokens = v
            break

    sw = _parse_stop_words_list_input(request)
    if sw is not None:
        stop_words_list = sw

    response_format, json_format, structural_tag = _parse_grammar_controls(
        request, ds_attrs
    )

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
        return_logprobs=return_logprobs,
        top_logprobs=top_logprobs,
        max_new_think_tokens=max_new_think_tokens,
        stop_words_list=stop_words_list,
        response_format=response_format,
        json_format=json_format,
        structural_tag=structural_tag,
    )


def parse_other_params(request, ds_attrs: dict[str, Any] | None = None) -> OtherParams:
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

    ds_attrs = ds_attrs if ds_attrs is not None else parse_ds_header_attributes(request)
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

    debug = _parse_optional_parameter_bool(request, "debug")
    if debug is None:
        debug = _parse_optional_bool(_lookup_ds_request_control(ds_attrs, "debug"))

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
        debug=bool(debug),
        request_headers=request_headers,
    )


def parse_dash_sc_grpc_request(
    request,
) -> tuple[list[int] | None, SamplingParams | None, OtherParams | None]:
    """Parse one ``ModelInferRequest``: ``input_ids``, sampling tensors, ``other`` params."""
    ids = parse_input_ids_from_request(request)
    if ids is None:
        return None, None, None
    ds_attrs = parse_ds_header_attributes(request)
    return (
        ids,
        parse_sampling_params(request, ds_attrs),
        parse_other_params(request, ds_attrs),
    )


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
    ``generated_ids`` is absent, when its declared shape is a zero-length / filler
    payload, or when the frame already contains logprob tensors. Prompt echo tokens
    have no sampled probability; callers must emit them in a separate no-logprob
    frame instead of making the existing tensors misaligned.
    """
    if not token_ids:
        return False
    if any(
        out.name in {"token_logprobs", "top_logprob_token_ids", "top_logprobs"}
        for out in infer.outputs
    ):
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
    debug: bool = False,
) -> None:
    """Mirror dashllm response parameters consumed by dashscope-serving."""
    if generate_config is not None:
        infer.parameters["max_new_tokens"].int64_param = int(
            getattr(generate_config, "max_new_tokens", 0) or 0
        )
        max_think = int(getattr(generate_config, "max_thinking_tokens", 0) or 0)
        if max_think != 0:
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
    if debug:
        llm_params: dict[str, int] = {}
        if generate_config is not None:
            llm_params["max_new_tokens"] = int(
                getattr(generate_config, "max_new_tokens", 0) or 0
            )
            llm_params["max_new_think_tokens"] = int(
                getattr(generate_config, "max_thinking_tokens", 0) or 0
            )
        infer.parameters["debug_info"].string_param = json.dumps(
            {"llm_params": llm_params}, separators=(",", ":")
        )


def _placeholder_top_token_ids(token_id: int, width: int) -> list[int]:
    """Return one sampled-token-first row for a positional placeholder."""
    candidates = [int(token_id)]
    candidate = 0
    while len(candidates) < width:
        if candidate != token_id:
            candidates.append(candidate)
        candidate += 1
    return candidates


def _placeholder_top_logprob_tensors(
    generated_ids: list[int],
    width: int,
    *,
    id_dtype: torch.dtype = torch.int32,
    prob_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(
        [_placeholder_top_token_ids(token_id, width) for token_id in generated_ids],
        dtype=id_dtype,
    ).reshape(len(generated_ids), width)
    probs = torch.full(
        (len(generated_ids), width),
        -float("inf"),
        dtype=prob_dtype,
    )
    if generated_ids and width:
        probs[:, 0] = 0.0
    return ids, probs


def _append_forced_token_logprob_outputs(
    infer: predict_v2_pb2.ModelInferResponse,
    *,
    generated_ids: list[int],
    generate_config: Any,
) -> None:
    """Encode positional placeholders for tokens outside normal content.

    DashScope accumulates token ids and logprob rows in parallel, then slices
    the accumulated rows at the reasoning/content boundary.  Reasoning and
    controller-inserted tokens therefore still need one alignment row each,
    even though their model probability is intentionally not exposed.  A
    sampled-token-first ``0.0`` row is an alignment marker, not a reported
    model probability.
    """
    if not bool(getattr(generate_config, "return_logprobs", False)):
        return
    num_tokens = len(generated_ids)
    if not num_tokens:
        return
    token_values = torch.zeros(num_tokens, dtype=torch.float32)
    _append_tensor_output(
        infer,
        tensor_name="token_logprobs",
        datatype="FP32",
        shape=[1, num_tokens],
        tensor=token_values,
    )
    requested_k = int(getattr(generate_config, "top_logprobs", 0) or 0)
    wire_ids, wire_values = _placeholder_top_logprob_tensors(
        generated_ids, max(1, requested_k)
    )
    wire_rows = [
        {
            str(candidate_id): float(candidate_logprob)
            for candidate_id, candidate_logprob in zip(id_row, prob_row)
        }
        for id_row, prob_row in zip(wire_ids.tolist(), wire_values.tolist())
    ]
    if requested_k > 0:
        top_ids = wire_ids[:, :requested_k]
        top_values = wire_values[:, :requested_k]
        _append_tensor_output(
            infer,
            tensor_name="top_logprob_token_ids",
            datatype="INT32",
            shape=[1, num_tokens, requested_k],
            tensor=top_ids,
        )
        _append_tensor_output(
            infer,
            tensor_name="top_logprobs",
            datatype="FP32",
            shape=[1, num_tokens, requested_k],
            tensor=top_values,
        )
    infer.parameters["logprobs"].string_param = json.dumps(
        wire_rows,
        separators=(",", ":"),
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


def _normalize_token_logprobs_tensor(
    tensor: Any,
    *,
    tensor_name: str,
    num_tokens: int,
) -> torch.Tensor:
    value = tensor.detach().cpu()
    if not value.is_floating_point():
        raise ValueError(f"{tensor_name} must use a floating-point dtype")
    if value.dim() == 1 and value.shape[0] == num_tokens:
        pass
    elif value.dim() == 2 and tuple(value.shape) == (1, num_tokens):
        value = value[0]
    else:
        raise ValueError(
            f"{tensor_name} must have shape [{num_tokens}] or [1,{num_tokens}]: "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{tensor_name} contains NaN or infinity")
    return value


def _normalize_top_logprobs_tensor(
    tensor: Any,
    *,
    tensor_name: str,
    num_tokens: int,
    floating_point: bool,
) -> torch.Tensor:
    value = tensor.detach().cpu()
    if floating_point:
        if not value.is_floating_point():
            raise ValueError(f"{tensor_name} must use a floating-point dtype")
    elif value.is_floating_point() or value.dtype == torch.bool:
        raise ValueError(f"{tensor_name} must use an integer dtype")
    if value.dim() == 3 and value.shape[0] == 1:
        value = value[0]
    if value.dim() != 2 or value.shape[0] != num_tokens:
        raise ValueError(
            f"{tensor_name} must have shape [num_output_tokens, top_logprobs]: "
            f"got {tuple(value.shape)} for {num_tokens} tokens"
        )
    if floating_point and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{tensor_name} contains NaN or infinity")
    if not floating_point and value.numel():
        min_id = int(value.min().item())
        max_id = int(value.max().item())
        if min_id < -(1 << 31) or max_id >= (1 << 31):
            raise ValueError(f"{tensor_name} contains values outside INT32 range")
    return value


def _logprob_token_rows(tensor: Any, *, tensor_name: str) -> int:
    """Return the token-row count without moving the tensor off device."""
    if tensor_name == "token_logprobs":
        if tensor.dim() == 1:
            return int(tensor.shape[0])
        if tensor.dim() == 2 and tensor.shape[0] == 1:
            return int(tensor.shape[1])
    else:
        if tensor.dim() == 2:
            return int(tensor.shape[0])
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            return int(tensor.shape[1])
    raise ValueError(f"invalid {tensor_name} shape: {tuple(tensor.shape)}")


def _slice_logprob_token_rows(tensor: Any, *, tensor_name: str, end: int) -> Any:
    if tensor_name == "token_logprobs":
        return tensor[:end] if tensor.dim() == 1 else tensor[:, :end]
    return tensor[:end, :] if tensor.dim() == 2 else tensor[:, :end, :]


def _drop_unused_mtp_logprob_tail(
    token_logprobs: Any,
    top_token_ids: Any,
    top_logprobs: Any,
    *,
    num_tokens: int,
) -> tuple[Any, Any, Any]:
    """Discard the uncommitted MTP verification row from a stream packet.

    Some MTP result packets expose probabilities for ``accept_len + 1`` target
    rows while ``output_ids`` contains only the ``accept_len`` committed rows.
    The final row is the unused verification tail and must never be forwarded
    to DashLLM: its V3 parser requires one probability dictionary per emitted
    token.  Keep strict validation for every other mismatch.
    """
    tensors = [
        ("token_logprobs", token_logprobs),
        ("top_logprob_token_ids", top_token_ids),
        ("top_logprobs", top_logprobs),
    ]
    present = [(name, value) for name, value in tensors if value is not None]
    if not present:
        return token_logprobs, top_token_ids, top_logprobs
    row_counts = {
        _logprob_token_rows(value, tensor_name=name) for name, value in present
    }
    if row_counts == {num_tokens + 1}:
        logging.debug(
            "[DashScGrpc] dropping one uncommitted MTP logprob row: ids=%s probabilities=%s",
            num_tokens,
            num_tokens + 1,
        )
        sliced = {
            name: _slice_logprob_token_rows(value, tensor_name=name, end=num_tokens)
            for name, value in present
        }
        return (
            sliced["token_logprobs"],
            sliced.get("top_logprob_token_ids"),
            sliced.get("top_logprobs"),
        )
    return token_logprobs, top_token_ids, top_logprobs


def _append_tensor_output(
    infer: predict_v2_pb2.ModelInferResponse,
    *,
    tensor_name: str,
    datatype: str,
    shape: list[int],
    tensor: torch.Tensor,
) -> None:
    out = infer.outputs.add()
    out.name = tensor_name
    out.datatype = datatype
    out.shape[:] = shape
    if datatype == "FP32":
        values = tensor.to(dtype=torch.float32).contiguous().reshape(-1).tolist()
        raw = struct.pack(f"<{len(values)}f", *values) if values else b""
    elif datatype == "INT32":
        values = tensor.to(dtype=torch.int32).contiguous().reshape(-1).tolist()
        raw = struct.pack(f"<{len(values)}i", *values) if values else b""
    else:
        raise ValueError(f"unsupported DashScope response datatype: {datatype}")
    infer.raw_output_contents.append(raw)


def _trim_one_elided_trailing_logprob_row(out_py: Any, num_tokens: int) -> None:
    """Align compact rows after the engine elides one trailing output token.

    A final async/MTP frame can retain placement metadata and probability
    tensors for the sampled stop/capped token after that token has already
    been removed from ``output_ids``.  Only this exact one-token trailing skew
    is recoverable: keep the rows for surviving ids and discard the final row.
    All other placement mismatches remain hard errors in the caller.
    """
    raw_offset = getattr(out_py, "logprobs_offset", None)
    raw_count = getattr(out_py, "logprobs_count", None)
    if raw_offset is None or raw_count is None:
        return

    logprobs_offset = int(raw_offset)
    logprobs_count = int(raw_count)
    if logprobs_offset < 0 or logprobs_count < 0:
        return
    if logprobs_offset + logprobs_count != num_tokens + 1:
        return

    trimmed_offset = min(logprobs_offset, num_tokens)
    trimmed_count = num_tokens - trimmed_offset
    for name in ("token_logprobs", "top_logprob_token_ids", "top_logprobs"):
        value = getattr(out_py, name, None)
        if value is None:
            continue
        if value.dim() >= 1 and value.shape[0] == logprobs_count:
            trimmed = value.narrow(0, 0, trimmed_count)
        elif (
            value.dim() >= 2
            and value.shape[0] == 1
            and value.shape[1] == logprobs_count
        ):
            trimmed = value.narrow(1, 0, trimmed_count)
        else:
            raise ValueError(
                f"{name} does not align with {logprobs_count} compact rows: "
                f"shape={tuple(value.shape)}"
            )
        setattr(out_py, name, trimmed.clone())

    out_py.logprobs_offset = trimmed_offset
    out_py.logprobs_count = trimmed_count
    logging.getLogger().debug(
        "trimmed one elided trailing compact logprob row: "
        "offset=%s count=%s tokens=%s",
        logprobs_offset,
        logprobs_count,
        num_tokens,
    )


def _append_logprob_outputs(
    infer: predict_v2_pb2.ModelInferResponse,
    out_py: Any,
    *,
    generated_ids: list[int],
    num_tokens: int,
    generate_config: Any,
    placeholder_prefix_token_count: int = 0,
) -> None:
    """Expand compact content logprobs into the aligned DashScope wire form.

    RTP-LLM carries only real rows for
    ``output_ids[logprobs_offset:logprobs_offset + logprobs_count]``.  This is
    the sole boundary that materializes positional placeholders for the
    preceding thinking/controller tokens. Legacy aligned backend tensors are
    still accepted during rolling upgrades.
    """
    token_logprobs = getattr(out_py, "token_logprobs", None)
    top_token_ids = getattr(out_py, "top_logprob_token_ids", None)
    top_logprobs = getattr(out_py, "top_logprobs", None)
    requested = bool(getattr(generate_config, "return_logprobs", False))
    requested_k = int(getattr(generate_config, "top_logprobs", 0) or 0)
    placeholder_prefix_token_count = int(placeholder_prefix_token_count)
    if not 0 <= placeholder_prefix_token_count <= num_tokens:
        raise ValueError(
            "placeholder_prefix_token_count must be between 0 and the number "
            f"of output tokens: got {placeholder_prefix_token_count} for {num_tokens}"
        )

    if not requested:
        return

    _trim_one_elided_trailing_logprob_row(out_py, num_tokens)
    token_logprobs = getattr(out_py, "token_logprobs", None)
    top_token_ids = getattr(out_py, "top_logprob_token_ids", None)
    top_logprobs = getattr(out_py, "top_logprobs", None)

    raw_offset = getattr(out_py, "logprobs_offset", None)
    raw_count = getattr(out_py, "logprobs_count", None)
    has_compact_placement = raw_offset is not None or raw_count is not None
    if has_compact_placement and (raw_offset is None or raw_count is None):
        raise ValueError("logprobs_offset and logprobs_count must be provided together")

    if has_compact_placement:
        logprobs_offset = int(raw_offset)
        logprobs_count = int(raw_count)
        if (
            logprobs_offset < 0
            or logprobs_count < 0
            or logprobs_offset + logprobs_count != num_tokens
        ):
            raise ValueError(
                "compact logprobs must cover one output_ids suffix: "
                f"offset={logprobs_offset}, count={logprobs_count}, "
                f"tokens={num_tokens}"
            )
        if (
            placeholder_prefix_token_count
            and placeholder_prefix_token_count != logprobs_offset
        ):
            raise ValueError(
                "DashSC thinking boundary disagrees with backend logprob "
                f"placement: dashsc={placeholder_prefix_token_count}, "
                f"backend={logprobs_offset}"
            )
        real_row_count = logprobs_count
    else:
        # Legacy responses have one backend row per id. The servicer-provided
        # boundary marks which aligned prefix rows must be replaced.
        logprobs_offset = placeholder_prefix_token_count
        real_row_count = num_tokens

    if token_logprobs is None:
        if top_token_ids is not None or top_logprobs is not None:
            raise ValueError("top-logprob tensors require token_logprobs")
        if (
            has_compact_placement
            and logprobs_offset == num_tokens
            and not real_row_count
        ):
            _append_forced_token_logprob_outputs(
                infer,
                generated_ids=generated_ids,
                generate_config=generate_config,
            )
            return
        if num_tokens:
            raise ValueError(
                "token_logprobs is missing for a DashScope logprobs request"
            )
        return

    token_logprobs, top_token_ids, top_logprobs = _drop_unused_mtp_logprob_tail(
        token_logprobs,
        top_token_ids,
        top_logprobs,
        num_tokens=real_row_count,
    )

    real_token_values = _normalize_token_logprobs_tensor(
        token_logprobs,
        tensor_name="token_logprobs",
        num_tokens=real_row_count,
    )
    if has_compact_placement:
        token_values = torch.cat(
            [
                torch.zeros(logprobs_offset, dtype=torch.float32),
                real_token_values,
            ]
        )
    else:
        token_values = real_token_values
        if logprobs_offset:
            token_values = token_values.clone()
            token_values[:logprobs_offset] = 0.0
    _append_tensor_output(
        infer,
        tensor_name="token_logprobs",
        datatype="FP32",
        shape=[1, num_tokens],
        tensor=token_values,
    )

    # dashllm/dashserving does not consume the compact tensors above when it
    # builds the public DashScope response.  Its established response contract
    # carries one token-id -> logprob dictionary per generated token in the
    # string parameter named ``logprobs``.  Keep both encodings: the tensors
    # are useful to direct ModelStreamInfer consumers, while this parameter is
    # what reaches output.choices[].logprobs at the DashScope frontend.
    wire_logprobs: list[dict[str, float]] = [
        {str(token_id): float(token_values[i].item())}
        for i, token_id in enumerate(generated_ids)
    ]

    if (top_token_ids is None) != (top_logprobs is None):
        raise ValueError(
            "top_logprob_token_ids and top_logprobs must be returned together"
        )
    if top_token_ids is None:
        if requested_k > 0 and num_tokens:
            raise ValueError(
                "top-logprob tensors are missing for a positive top_logprobs request"
            )
        infer.parameters["logprobs"].string_param = json.dumps(
            wire_logprobs, separators=(",", ":")
        )
        return

    top_id_values = _normalize_top_logprobs_tensor(
        top_token_ids,
        tensor_name="top_logprob_token_ids",
        num_tokens=real_row_count,
        floating_point=False,
    )
    top_prob_values = _normalize_top_logprobs_tensor(
        top_logprobs,
        tensor_name="top_logprobs",
        num_tokens=real_row_count,
        floating_point=True,
    )
    if top_id_values.shape != top_prob_values.shape:
        raise ValueError(
            "top_logprob_token_ids and top_logprobs must have identical shapes"
        )
    actual_k = int(top_id_values.shape[1])
    if requested and actual_k > requested_k:
        raise ValueError(
            f"backend returned top_logprobs={actual_k}, exceeding requested "
            f"top_logprobs={requested_k}"
        )
    if has_compact_placement:
        if actual_k > 0:
            prefix_top_ids, prefix_top_probs = _placeholder_top_logprob_tensors(
                generated_ids[:logprobs_offset],
                actual_k,
                id_dtype=top_id_values.dtype,
                prob_dtype=top_prob_values.dtype,
            )
            top_id_values = torch.cat([prefix_top_ids, top_id_values], dim=0)
            top_prob_values = torch.cat([prefix_top_probs, top_prob_values], dim=0)
        else:
            top_id_values = torch.empty((num_tokens, 0), dtype=top_id_values.dtype)
            top_prob_values = torch.empty((num_tokens, 0), dtype=top_prob_values.dtype)
    elif logprobs_offset and actual_k > 0:
        top_id_values = top_id_values.clone()
        top_prob_values = top_prob_values.clone()
        prefix_top_ids, prefix_top_probs = _placeholder_top_logprob_tensors(
            generated_ids[:logprobs_offset],
            actual_k,
            id_dtype=top_id_values.dtype,
            prob_dtype=top_prob_values.dtype,
        )
        top_id_values[:logprobs_offset] = prefix_top_ids
        top_prob_values[:logprobs_offset] = prefix_top_probs
    for token_index in range(num_tokens):
        for top_index in range(actual_k):
            wire_logprobs[token_index][
                str(int(top_id_values[token_index, top_index].item()))
            ] = float(top_prob_values[token_index, top_index].item())
    infer.parameters["logprobs"].string_param = json.dumps(
        wire_logprobs, separators=(",", ":")
    )
    _append_tensor_output(
        infer,
        tensor_name="top_logprob_token_ids",
        datatype="INT32",
        shape=[1, num_tokens, actual_k],
        tensor=top_id_values,
    )
    _append_tensor_output(
        infer,
        tensor_name="top_logprobs",
        datatype="FP32",
        shape=[1, num_tokens, actual_k],
        tensor=top_prob_values,
    )


def _log_dashsc_logprob_frame(
    *,
    infer: predict_v2_pb2.ModelInferResponse,
    out_py: Any,
    request_log_tag: str,
    generated_ids: list[int],
    generate_config: Any,
    generate_think_token_num: int | None,
    phase: str,
    pre_content_token_count: int | None,
    include_logprobs: bool,
    include_forced_token_logprobs: bool,
    placeholder_prefix_token_count: int,
    finished: bool,
) -> None:
    """Log backend-to-wire logprob counts for one DashScope response frame."""
    if not bool(getattr(generate_config, "return_logprobs", False)):
        return
    logger = logging.getLogger()
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        backend_placement_offset = getattr(out_py, "logprobs_offset", None)
        source = "none"
        if include_logprobs:
            source = (
                "backend_compact_with_placeholder_prefix"
                if backend_placement_offset or placeholder_prefix_token_count
                else "backend"
            )
        elif include_forced_token_logprobs:
            source = "placeholder"
        backend_generated_ids = _token_ids_list_from_generate_output(out_py)
        backend_token_logprob_rows: int | None = None
        backend_token_logprobs = getattr(out_py, "token_logprobs", None)
        if include_logprobs and backend_token_logprobs is not None:
            try:
                backend_token_logprob_rows = _logprob_token_rows(
                    backend_token_logprobs,
                    tensor_name="token_logprobs",
                )
            except (AttributeError, ValueError):
                backend_token_logprob_rows = -1

        wire_tensor_logprob_rows: int | None = None
        for output in infer.outputs:
            if output.name != "token_logprobs":
                continue
            shape = list(output.shape)
            wire_tensor_logprob_rows = (
                int(shape[1]) if len(shape) == 2 and shape[0] == 1 else -1
            )
            break

        wire_rows: list[Any] = []
        wire_parse_error: str | None = None
        wire_param = infer.parameters.get("logprobs")
        if (
            wire_param is not None
            and wire_param.WhichOneof("parameter_choice") == "string_param"
        ):
            try:
                parsed = json.loads(wire_param.string_param)
                if isinstance(parsed, list):
                    wire_rows = parsed
                else:
                    wire_parse_error = f"decoded_{type(parsed).__name__}"
            except (TypeError, json.JSONDecodeError) as error:
                wire_parse_error = type(error).__name__

        token_count = len(generated_ids)
        wire_count = len(wire_rows)
        effective_prefix_count = (
            int(backend_placement_offset)
            if include_logprobs and backend_placement_offset is not None
            else int(placeholder_prefix_token_count)
        )
        placeholder_rows = (
            token_count
            if include_forced_token_logprobs
            else min(max(effective_prefix_count, 0), token_count)
        )
        real_rows = max(0, wire_count - placeholder_rows)
        aligned = token_count == wire_count and (
            wire_tensor_logprob_rows == token_count
            or (token_count == 0 and wire_tensor_logprob_rows is None)
        )
        sampled_tokens_match = token_count == wire_count and all(
            isinstance(row, dict) and str(token_id) in row
            for token_id, row in zip(generated_ids, wire_rows)
        )
        if pre_content_token_count is None:
            pre_content_count = None
            content_count = None
        else:
            pre_content_count = max(0, min(int(pre_content_token_count), token_count))
            content_count = token_count - pre_content_count
        logger.debug(
            "[DashScGrpcLogprobs] [%s] stage=wire response_id=%s "
            "phase=%s source=%s "
            "finished=%s backend_token_count=%s backend_token_logprob_rows=%s "
            "wire_token_count=%s wire_tensor_logprob_rows=%s "
            "wire_json_logprob_rows=%s aligned=%s sampled_tokens_match=%s "
            "placeholder_logprob_rows=%s real_logprob_rows=%s "
            "pre_content_token_count=%s content_token_count=%s "
            "generate_think_token_num=%s top_logprobs=%s "
            "token_ids_preview=%s wire_parse_error=%s",
            request_log_tag,
            infer.id,
            phase,
            source,
            finished,
            len(backend_generated_ids),
            backend_token_logprob_rows,
            token_count,
            wire_tensor_logprob_rows,
            wire_count,
            aligned,
            sampled_tokens_match,
            placeholder_rows,
            real_rows,
            pre_content_count,
            content_count,
            generate_think_token_num,
            int(getattr(generate_config, "top_logprobs", 0) or 0),
            generated_ids[:4],
            wire_parse_error,
        )
    except Exception as error:
        # Observability must never change inference behavior.
        logger.warning(
            "[DashScGrpcLogprobs] [%s] stage=wire diagnostic_error=%s",
            request_log_tag,
            type(error).__name__,
        )


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
    include_logprobs: bool = True,
    include_forced_token_logprobs: bool = False,
    logprob_placeholder_prefix_token_count: int = 0,
    logprob_phase: str = "unspecified",
    logprob_pre_content_token_count: int | None = None,
    debug: bool = False,
) -> predict_v2_pb2.ModelStreamInferResponse:
    """Build ``ModelStreamInferResponse`` from one ``GenerateOutputs`` chunk.

    ``stream_finished``: if provided, overrides ``out_py.finished`` for the wire
    ``finished`` / ``finish_reason`` fields. Use when the servicer knows the gRPC
    stream is not done yet (e.g. phase-2 will follow).

    ``token_ids``: if provided, overrides the generated_ids from ``out_py``.
    Use when the servicer rewrites the token payload (e.g. injecting </think>).

    ``include_logprobs``: encode probability tensors from the backend output.
    Prompt-echo frames disable this because those ids were not sampled.

    ``include_forced_token_logprobs``: encode logprob-zero alignment rows for
    a frame containing only non-content tokens.

    ``logprob_placeholder_prefix_token_count``: replace this many leading
    backend rows with alignment placeholders for a reasoning/content boundary
    frame.  The remaining suffix keeps its real backend probabilities.

    ``logprob_phase`` / ``logprob_pre_content_token_count`` annotate only the
    diagnostic log. The latter counts this frame's reasoning and delimiter
    tokens before normal content begins.
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
    backend_generated_ids = _token_ids_list_from_generate_output(out_py)
    generated_ids = token_ids if token_ids is not None else backend_generated_ids
    if (
        include_logprobs
        and token_ids is not None
        and generated_ids != backend_generated_ids
        and any(
            getattr(out_py, name, None) is not None
            for name in ("token_logprobs", "top_logprob_token_ids", "top_logprobs")
        )
    ):
        raise ValueError(
            "rewritten generated_ids require matching logprob tensor slicing or "
            "include_logprobs=False"
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
    # Append new optional tensors after the legacy response outputs so existing
    # positional consumers retain their original indices. raw_output_contents
    # stays one-to-one with outputs because each helper appends both together.
    if include_logprobs:
        _append_logprob_outputs(
            infer,
            out_py,
            generated_ids=generated_ids,
            num_tokens=len(generated_ids),
            generate_config=generate_config,
            placeholder_prefix_token_count=logprob_placeholder_prefix_token_count,
        )
    elif include_forced_token_logprobs:
        _append_forced_token_logprob_outputs(
            infer,
            generated_ids=generated_ids,
            generate_config=generate_config,
        )
    infer.parameters["incremental_output"].int64_param = 1 if is_streaming else 0
    _append_dashllm_limit_parameters(
        infer,
        generate_config=generate_config,
        eos_token_id=eos_token_id,
        max_token_id=max_token_id,
        generate_think_token_num=generate_think_token_num,
        debug=debug and finished,
    )

    _log_dashsc_logprob_frame(
        infer=infer,
        out_py=out_py,
        request_log_tag=request_log_tag,
        generated_ids=generated_ids,
        generate_config=generate_config,
        generate_think_token_num=generate_think_token_num,
        phase=logprob_phase,
        pre_content_token_count=logprob_pre_content_token_count,
        include_logprobs=include_logprobs,
        include_forced_token_logprobs=include_forced_token_logprobs,
        placeholder_prefix_token_count=logprob_placeholder_prefix_token_count,
        finished=bool(finished),
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


def build_error_response(
    request_id: str,
    message: str,
    *,
    status_code: int,
    status_name: str,
) -> predict_v2_pb2.ModelStreamInferResponse:
    """Build a business error frame without using the gRPC hard-error channel."""
    dashscope_frame = {
        "header": {
            "status_code": status_code,
            "status_name": status_name,
            "status_message": message,
            "finished": True,
            "request_id": request_id,
        },
        "payload": {},
    }

    resp = predict_v2_pb2.ModelStreamInferResponse()
    infer = resp.infer_response
    infer.id = request_id
    _append_generated_ids_output(infer, [])
    _append_finish_reason_output(
        infer,
        finished=True,
        finish_reason_override=FINISH_REASON_USE_PARAMETER_STATUS,
    )
    _append_finished_output(infer, finished=True)
    infer.parameters["incremental_output"].int64_param = 1
    infer.parameters["status_code"].int64_param = int(status_code)
    infer.parameters["status_name"].string_param = status_name
    infer.parameters["status_message"].string_param = message
    infer.parameters["__messages__"].string_param = json.dumps(
        dashscope_frame, ensure_ascii=False, separators=(",", ":")
    )
    return resp


def build_parameter_error_response(
    request_id: str, message: str, *, status_code: int = 400
) -> predict_v2_pb2.ModelStreamInferResponse:
    return build_error_response(
        request_id,
        message,
        status_code=status_code,
        status_name="InvalidParameter",
    )

"""DashScope-serving (dashllm) protocol compatibility layer for the dash_sc gRPC path.

Why a separate module: ``codec.py`` is a *wire codec* (tensor pack/unpack +
``InferParameter`` reads/writes). The dashllm protocol however carries a layer
of policy semantics on top of that wire — what ``parameters["stop"]`` JSON
shape means, which ``invocation_metadata`` headers gate which response field,
which input names map to which ``GenerateConfig`` knob. Mixing those rules
into ``codec.py`` would couple the wire layer to dashllm-specific quirks and
break the rtp_llm/openai parallel.

Coverage scope (P1 + P2 only — see ``rtp_llm/dash_sc/AIO_MIGRATION.md``):

P1 (blocking dashscope-serving):
  - ``parameters["stop"]`` (string JSON list)             -> GenerateConfig.stop_words_str
  - ``parameters["stop_token_ids"]`` (string JSON list)   -> GenerateConfig.stop_words_list
  - metadata ``x-ds-max-matched-token-num``               -> response ``prompt_cached_token_num``
                                                            forced to 0 (mirrors dashllm
                                                            ``model.py:827-829``)

P2 (OpenAI-compat + observability):
  - ``parameters["incremental_output"]`` (int64)          -> GenerateConfig.return_incremental
  - ``parameters["scheduler_request_id"]`` (string)       -> GenerateConfig.chat_id
  - input ``max_new_think_tokens`` / ``max_think_length`` -> GenerateConfig.max_thinking_tokens
  - input ``logprobs`` (INT32 > 0)                        -> GenerateConfig.return_all_probs
  - input ``top_logprobs`` (INT32)                        -> recorded for log only
  - response echo ``model_name`` / ``model_instance_ip`` / ``model_hostname`` /
    ``scheduler_request_id``                              -> InferResponse.parameters

P3 (grammar / structured output — protocol parse only; engine wiring lands
with the xgrammar backend integration):
  - ``parameters["response_format"]`` (string JSON, OpenAI ResponseFormat shape)
        type=json_object   -> GenerateConfig.json_schema = '{"type": "object"}'
        type=json_schema   -> GenerateConfig.json_schema = json.dumps(payload)
        type=regex         -> GenerateConfig.regex
        type=ebnf          -> GenerateConfig.ebnf
        type=structural_tag-> GenerateConfig.structural_tag
        type=text          -> noop
  - ``parameters["guided_json"]`` (string JSON, vLLM bare-schema convenience)
        -> GenerateConfig.json_schema (when response_format absent)

  Field semantics mirror ``rtp_llm/openai/openai_endpoint.py:_apply_response_format``
  so HTTP and gRPC paths converge on the same GenerateConfig knobs. Until the
  xgrammar branch merges those four fields onto ``GenerateConfig``, the applier
  detects their absence via ``model_fields`` and emits a single WARN per request
  per field — same observability shape as the legacy unsupported bucket.

Engine-unsupported P2 fields (parsed but dropped, with a single WARN log so
upstream misuse becomes visible):
  - ``parameters["logit_bias"]``
  - ``parameters["context_cache_ttl"]`` (also ``inputs["context_cache_ttl"]``)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional

from rtp_llm.dash_sc.codec import _find_input_raw, _parse_optional_scalar_int
from rtp_llm.dash_sc.proto import predict_v2_pb2

if TYPE_CHECKING:
    from rtp_llm.config.generate_config import GenerateConfig


# Metadata header that gates whether the response advertises a non-zero
# ``prompt_cached_token_num``. dashllm reads it via
# ``ds_header_attributes["x-ds-max-matched-token-num"]`` and, when the value
# is non-positive AND no ``context_cache_ttl`` is set, force-rewrites
# ``prompt_cached_token_num`` to 0 in the response — see
# ``dashllm/worker/processors/model.py:827-829``. Keeping the same gate here
# ensures cache hit-rate dashboards on dashscope-serving stay calibrated.
_MAX_MATCHED_TOKEN_NUM_HEADER = "x-ds-max-matched-token-num"

# Parameter / input names that are parsed but cannot map onto GenerateConfig.
# A single WARN log per request per field surfaces upstream usage so engine-
# side support can be prioritised; we deliberately do NOT silently swallow.
_UNSUPPORTED_PARAMETER_FIELDS: tuple[str, ...] = (
    "logit_bias",
    "context_cache_ttl",
)
_UNSUPPORTED_INPUT_FIELDS: tuple[str, ...] = ("context_cache_ttl",)


# ResponseFormat ``type`` values that map onto a grammar-constrained
# GenerateConfig field. Mirrors the literal in
# ``rtp_llm/openai/api_datatype.py:ResponseFormat.type`` on the xgrammar
# branch. ``text`` is accepted but produces no constraint.
_GRAMMAR_RESPONSE_FORMAT_TYPES: frozenset[str] = frozenset(
    {"json_object", "json_schema", "regex", "ebnf", "structural_tag"}
)

# GenerateConfig fields written by the grammar applier. Used at apply-time to
# detect whether the running engine has the xgrammar fields declared — when
# they're absent we emit a WARN instead of silently dropping (matches the
# observability contract of ``_UNSUPPORTED_PARAMETER_FIELDS``).
_GRAMMAR_GENERATE_CONFIG_FIELDS: tuple[str, ...] = (
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
)


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DashScopeRequestExtras:
    """Parsed dashscope-serving fields that aren't part of the core sampling/wire codec.

    All fields are immutable so the same instance can be safely shared
    between the request-applier (``apply_dashscope_extras_to_generate_config``)
    and the response-builder (``DashScopeResponseEchoArgs`` derivation).
    """

    # P1
    stop_strs: tuple[str, ...] = ()
    extra_stop_token_groups: tuple[tuple[int, ...], ...] = ()
    suppress_cached_token_num: bool = False
    # P2
    incremental_output: Optional[bool] = None
    scheduler_request_id: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    enable_logprobs: bool = False
    top_logprobs: int = 0
    # P3 grammar (engine wiring lands with the xgrammar backend). Values are
    # already normalised to the GenerateConfig shape — ``json_schema`` is a
    # JSON-encoded string (so the dict round-trip happens once at parse time,
    # not on every applier call), ``regex`` / ``ebnf`` are raw bodies,
    # ``structural_tag`` is a JSON-encoded string of the wrapper dict.
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None
    # Parsed-but-dropped (engine doesn't support these knobs).
    unsupported: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DashScopeResponseEchoArgs:
    """Inputs for ``append_dashscope_response_extras``. Built once per RPC by the
    servicer and threaded into every chunk's ``build_stream_response_from_generate_outputs``
    call so each ``ModelStreamInferResponse`` carries the same dashllm parameters.
    """

    suppress_cached_token_num: bool = False
    model_name: Optional[str] = None
    instance_ip: Optional[str] = None
    hostname: Optional[str] = None
    scheduler_request_id: Optional[str] = None


# ----------------------------------------------------------------------------
# Parameter / metadata helpers
# ----------------------------------------------------------------------------


def _string_param(
    request: predict_v2_pb2.ModelInferRequest, name: str
) -> Optional[str]:
    p = request.parameters.get(name)
    if p is None:
        return None
    if not p.HasField("string_param"):
        return None
    return p.string_param


def _int64_param(request: predict_v2_pb2.ModelInferRequest, name: str) -> Optional[int]:
    p = request.parameters.get(name)
    if p is None:
        return None
    if not p.HasField("int64_param"):
        return None
    return int(p.int64_param)


def _parse_json_param(
    request: predict_v2_pb2.ModelInferRequest, name: str, request_log_tag: str
) -> Any:
    """Parse a ``string_param`` carrying a JSON document. Returns ``None`` when the
    parameter is absent; raises ``ValueError`` when present but malformed (caller
    converts to an ``unsupported`` entry instead of bubbling a 500 to the client).
    """
    raw = _string_param(request, name)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Surface the failure into the unsupported bucket — explicit log,
        # never let a malformed payload crash the RPC.
        raise ValueError(f"json_decode_failed: {exc}") from exc


def _normalize_stop_strs(payload: Any) -> Optional[tuple[str, ...]]:
    """``parameters["stop"]`` may arrive as ``str`` (single stop) or ``list[str]``.

    Mirrors ``rtp_llm/openai/openai_endpoint.py:184-185`` which coerces a bare
    string to a one-element list. Returns ``None`` for any other shape so the
    caller can record it as unsupported rather than silently dropping.
    """
    if payload is None:
        return None
    if isinstance(payload, str):
        return (payload,) if payload else ()
    if isinstance(payload, list) and all(isinstance(s, str) for s in payload):
        return tuple(payload)
    return None


def _normalize_stop_token_groups(
    payload: Any,
) -> Optional[tuple[tuple[int, ...], ...]]:
    """``parameters["stop_token_ids"]`` may arrive as ``list[int]`` (one group)
    or ``list[list[int]]`` (multiple groups). Returns ``None`` for other shapes.
    """
    if payload is None:
        return None
    if not isinstance(payload, list):
        return None
    if not payload:
        return ()
    if all(isinstance(t, int) for t in payload):
        return (tuple(int(t) for t in payload),)
    if all(isinstance(g, list) and all(isinstance(t, int) for t in g) for g in payload):
        return tuple(tuple(int(t) for t in g) for g in payload)
    return None


def _extract_metadata_value(
    invocation_metadata: Optional[Iterable[tuple[str, str]]], key: str
) -> Optional[str]:
    """Case-insensitive lookup over gRPC ``invocation_metadata`` entries. Returns
    the first non-empty value for ``key`` (lower-cased). Mirrors
    ``access_log._extract_correlation_id`` so both modules share the same
    behaviour for header normalisation.
    """
    if not invocation_metadata:
        return None
    target = key.lower()
    for entry in invocation_metadata:
        try:
            k, v = entry
        except Exception:
            continue
        if k is None or v is None:
            continue
        if str(k).lower() == target and v:
            return str(v)
    return None


def _parse_max_thinking_tokens(
    request: predict_v2_pb2.ModelInferRequest,
) -> Optional[int]:
    """Look at the two dashllm-recognised tensor names in priority order — see
    ``dashllm/worker/processors/model.py:265-270``."""
    for tensor_name in ("max_new_think_tokens", "max_think_length"):
        v = _parse_optional_scalar_int(request, tensor_name)
        if v is not None:
            return v
    return None


@dataclass(frozen=True)
class _GrammarFields:
    """Result of normalising a ``response_format`` / ``guided_json`` payload
    onto the four GenerateConfig grammar knobs. Any combination of fields may
    be ``None`` (the empty case for ``type=text`` or absent payload). The
    ``error`` slot, when populated, captures a (field, reason) entry for the
    ``unsupported`` bucket so the caller can WARN without raising.
    """

    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None
    error: Optional[tuple[str, str]] = None


def _normalize_response_format_payload(payload: Any) -> _GrammarFields:
    """Map a parsed ``response_format`` dict onto the four grammar strings.

    Mirrors ``openai_endpoint._apply_response_format``: same accepted ``type``
    set, same JSON encoding for nested schema, same ``json.dumps`` separators
    so the digest written into the per-request grammar cache key is identical
    across HTTP and gRPC entrypoints. Returning a ``_GrammarFields`` (instead
    of raising) lets the caller record bad payloads in ``unsupported`` and
    keep the RPC alive — same policy as the existing ``stop`` handler.
    """
    if not isinstance(payload, dict):
        return _GrammarFields(
            error=("response_format", f"unexpected_shape:{type(payload).__name__}")
        )
    rf_type = payload.get("type")
    if rf_type == "text":
        return _GrammarFields()
    if rf_type == "json_object":
        return _GrammarFields(json_schema=json.dumps({"type": "object"}))
    if rf_type == "json_schema":
        # OpenAI ResponseFormatJSONSchema shape: ``{"json_schema": {"name":...,
        # "schema": {...}, "strict":...}}``. Mirrors the assertion in
        # ``openai_endpoint._apply_response_format`` so HTTP and gRPC paths
        # accept the same wire shape — a bare schema dict goes through
        # ``guided_json`` instead.
        wrapper = payload.get("json_schema")
        if not isinstance(wrapper, dict):
            return _GrammarFields(
                error=("response_format", "json_schema_missing_wrapper")
            )
        sch = wrapper.get("schema")
        if not isinstance(sch, dict):
            return _GrammarFields(
                error=("response_format", "json_schema_missing_schema")
            )
        return _GrammarFields(
            json_schema=json.dumps(sch, ensure_ascii=False, separators=(",", ":"))
        )
    if rf_type == "regex":
        pattern = payload.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return _GrammarFields(error=("response_format", "regex_missing_pattern"))
        return _GrammarFields(regex=pattern)
    if rf_type == "ebnf":
        grammar = payload.get("grammar")
        if not isinstance(grammar, str) or not grammar:
            return _GrammarFields(error=("response_format", "ebnf_missing_grammar"))
        return _GrammarFields(ebnf=grammar)
    if rf_type == "structural_tag":
        st = payload.get("structural_tag")
        if not isinstance(st, dict):
            return _GrammarFields(
                error=("response_format", "structural_tag_missing_payload")
            )
        return _GrammarFields(
            structural_tag=json.dumps(st, ensure_ascii=False, separators=(",", ":"))
        )
    return _GrammarFields(error=("response_format", f"unknown_type:{rf_type!r}"))


def _parse_logprobs_inputs(
    request: predict_v2_pb2.ModelInferRequest,
) -> tuple[bool, int]:
    """``logprobs`` (INT32 > 0 enables) and ``top_logprobs`` (INT32, default 0).

    rtp-llm only honours the boolean (``GenerateConfig.return_all_probs``) — see
    ``rtp_llm/openai/openai_endpoint.py:202-203``. ``top_logprobs`` is parsed
    for log/observability but cannot fan out tail probabilities through the
    current GenerateConfig schema.
    """
    enable = False
    top_n = 0
    v = _parse_optional_scalar_int(request, "logprobs")
    if v is not None and v > 0:
        enable = True
    v = _parse_optional_scalar_int(request, "top_logprobs")
    if v is not None and v > 0:
        top_n = int(v)
    return enable, top_n


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def parse_dashscope_request_extras(
    request: predict_v2_pb2.ModelInferRequest,
    invocation_metadata: Optional[Iterable[tuple[str, str]]] = None,
) -> DashScopeRequestExtras:
    """Read dashscope-specific knobs off ``request`` + gRPC headers.

    All payload-level errors (malformed JSON, wrong shape) are captured into the
    returned ``unsupported`` tuple instead of propagating — letting the RPC
    finish with degraded semantics rather than 500'ing on a bad client.
    """
    unsupported: list[tuple[str, str]] = []

    # P1: stop / stop_token_ids
    stop_strs: tuple[str, ...] = ()
    try:
        stop_payload = _parse_json_param(request, "stop", request_log_tag="")
    except ValueError as exc:
        unsupported.append(("stop", str(exc)))
        stop_payload = None
    if stop_payload is not None:
        normed = _normalize_stop_strs(stop_payload)
        if normed is None:
            unsupported.append(
                ("stop", f"unexpected_shape: {type(stop_payload).__name__}")
            )
        else:
            stop_strs = normed

    extra_stop_token_groups: tuple[tuple[int, ...], ...] = ()
    try:
        stop_ids_payload = _parse_json_param(
            request, "stop_token_ids", request_log_tag=""
        )
    except ValueError as exc:
        unsupported.append(("stop_token_ids", str(exc)))
        stop_ids_payload = None
    if stop_ids_payload is not None:
        normed_ids = _normalize_stop_token_groups(stop_ids_payload)
        if normed_ids is None:
            unsupported.append(
                (
                    "stop_token_ids",
                    f"unexpected_shape: {type(stop_ids_payload).__name__}",
                )
            )
        else:
            extra_stop_token_groups = normed_ids

    # P1: x-ds-max-matched-token-num metadata gate
    suppress_cached = False
    raw_max_matched = _extract_metadata_value(
        invocation_metadata, _MAX_MATCHED_TOKEN_NUM_HEADER
    )
    if raw_max_matched is not None:
        try:
            v = int(raw_max_matched)
            if v <= 0:
                suppress_cached = True
        except (TypeError, ValueError):
            unsupported.append(
                (_MAX_MATCHED_TOKEN_NUM_HEADER, f"non_int: {raw_max_matched!r}")
            )

    # P2: incremental_output / scheduler_request_id
    incremental_output: Optional[bool] = None
    raw_inc = _int64_param(request, "incremental_output")
    if raw_inc is not None:
        incremental_output = raw_inc > 0
    scheduler_request_id = _string_param(request, "scheduler_request_id")
    if scheduler_request_id == "":
        scheduler_request_id = None

    # P2: max_thinking_tokens
    max_thinking_tokens = _parse_max_thinking_tokens(request)

    # P2: logprobs / top_logprobs
    enable_logprobs, top_logprobs = _parse_logprobs_inputs(request)

    # P3: response_format (typed) and guided_json (vLLM bare-schema convenience).
    # response_format wins when both are present; we record the duplication on the
    # unsupported bucket so misuse is observable upstream rather than silently
    # ignored. Empty/text payloads do not collide with guided_json because they
    # leave json_schema unset.
    json_schema: Optional[str] = None
    regex_str: Optional[str] = None
    ebnf_str: Optional[str] = None
    structural_tag: Optional[str] = None

    rf_payload: Any = None
    try:
        rf_payload = _parse_json_param(request, "response_format", request_log_tag="")
    except ValueError as exc:
        unsupported.append(("response_format", str(exc)))
    if rf_payload is not None:
        gf = _normalize_response_format_payload(rf_payload)
        if gf.error is not None:
            unsupported.append(gf.error)
        else:
            json_schema = gf.json_schema
            regex_str = gf.regex
            ebnf_str = gf.ebnf
            structural_tag = gf.structural_tag

    # guided_json: vLLM-style bare schema dict. Parsed only when response_format
    # didn't already set a json_schema — keeps the OpenAI-typed path authoritative.
    try:
        guided_payload = _parse_json_param(
            request, "guided_json", request_log_tag=""
        )
    except ValueError as exc:
        unsupported.append(("guided_json", str(exc)))
        guided_payload = None
    if guided_payload is not None:
        if json_schema is not None:
            unsupported.append(
                ("guided_json", "ignored_response_format_already_set")
            )
        elif isinstance(guided_payload, dict):
            json_schema = json.dumps(
                guided_payload, ensure_ascii=False, separators=(",", ":")
            )
        else:
            unsupported.append(
                ("guided_json", f"unexpected_shape:{type(guided_payload).__name__}")
            )

    # Engine-unsupported fields: detect presence (not value) — applier emits the
    # WARN. We don't try to parse them since we won't honour them anyway.
    for name in _UNSUPPORTED_PARAMETER_FIELDS:
        if name in request.parameters:
            unsupported.append((name, "engine_unsupported"))
    for name in _UNSUPPORTED_INPUT_FIELDS:
        inp, _ = _find_input_raw(request, name)
        if inp is not None:
            unsupported.append((f"input.{name}", "engine_unsupported"))

    return DashScopeRequestExtras(
        stop_strs=stop_strs,
        extra_stop_token_groups=extra_stop_token_groups,
        suppress_cached_token_num=suppress_cached,
        incremental_output=incremental_output,
        scheduler_request_id=scheduler_request_id,
        max_thinking_tokens=max_thinking_tokens,
        enable_logprobs=enable_logprobs,
        top_logprobs=top_logprobs,
        json_schema=json_schema,
        regex=regex_str,
        ebnf=ebnf_str,
        structural_tag=structural_tag,
        unsupported=tuple(unsupported),
    )


def _dedup_keep_order(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for it in items:
        # Lists can't go into a set — fall back to tuple key for stop_words_list.
        key = tuple(it) if isinstance(it, list) else it
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def apply_dashscope_extras_to_generate_config(
    gc: "GenerateConfig",
    extras: DashScopeRequestExtras,
    *,
    request_log_tag: str = "",
) -> None:
    """In-place merge ``extras`` into ``gc``. MUST be called *after*
    ``SamplingParams.to_generate_config`` so it overrides the generic defaults
    rather than getting overwritten by them.

    Stop-word merge mirrors ``rtp_llm/openai/openai_endpoint.py:183-205``:
    union with the existing config-supplied lists, then dedup. This composes
    cleanly with the per-startup ``extra_stop_word_ids`` snapshot already
    applied by :func:`rtp_llm.dash_sc.inference.servicer.iter_real_model_stream_infer`
    (the latter hits the engine before this function and seeds gc.stop_words_list).
    """
    if extras.stop_strs:
        merged = list(gc.stop_words_str) + list(extras.stop_strs)
        # ``list(set(...))`` mirrors openai_endpoint.py:186-191; ordering loss
        # is acceptable because stop matching is order-independent.
        gc.stop_words_str = list(set(merged))

    if extras.extra_stop_token_groups:
        merged_ids = list(gc.stop_words_list) + [
            list(g) for g in extras.extra_stop_token_groups
        ]
        gc.stop_words_list = _dedup_keep_order(merged_ids)

    if extras.enable_logprobs:
        gc.return_all_probs = True

    if extras.max_thinking_tokens is not None and extras.max_thinking_tokens >= 0:
        gc.max_thinking_tokens = extras.max_thinking_tokens

    if extras.incremental_output is not None:
        gc.return_incremental = extras.incremental_output

    if extras.scheduler_request_id is not None:
        # ``chat_id`` is rtp-llm's existing per-request cache scheduling key —
        # repurposing it here keeps the dashllm scheduler_request_id semantics
        # (group co-scheduled requests by a logical id) intact without adding
        # a new GenerateConfig field.
        gc.chat_id = extras.scheduler_request_id

    # P3 grammar: write only when the running engine declares the field. Until
    # the xgrammar branch lands those four fields on GenerateConfig, the WARN
    # branch fires (matches the legacy engine_unsupported observability shape).
    declared = type(gc).model_fields
    grammar_values = {
        "json_schema": extras.json_schema,
        "regex": extras.regex,
        "ebnf": extras.ebnf,
        "structural_tag": extras.structural_tag,
    }
    for name, value in grammar_values.items():
        if value is None:
            continue
        if name in declared:
            setattr(gc, name, value)
        else:
            logging.warning(
                "[DashScopeCompat] [%s] grammar field %s parsed but engine "
                "lacks support (xgrammar backend not merged); dropping",
                request_log_tag,
                name,
            )

    for field_name, reason in extras.unsupported:
        # One log line per unsupported field per request. Bounded cardinality
        # (the static lists in this module cap unique field names), so this
        # cannot blow up disk on misbehaving clients.
        logging.warning(
            "[DashScopeCompat] [%s] dropping unsupported field %s (%s)",
            request_log_tag,
            field_name,
            reason,
        )


def append_dashscope_response_extras(
    infer: predict_v2_pb2.ModelInferResponse,
    echo: DashScopeResponseEchoArgs,
) -> None:
    """Append dashllm-style response parameters onto ``infer``.

    Called from inside ``build_stream_response_from_generate_outputs`` after
    the metric int64 parameters (``prompt_token_num`` / ``prompt_cached_token_num``)
    are written, so when ``echo.suppress_cached_token_num`` is True we can
    overwrite the previously-set ``prompt_cached_token_num`` with 0 — same
    semantics as ``dashllm/worker/processors/model.py:827-829``.
    """
    if echo.suppress_cached_token_num:
        # Overwrite, not delete — predict_v2 doesn't surface a ``parameters.pop``,
        # and writing 0 is what dashllm does too.
        infer.parameters["prompt_cached_token_num"].int64_param = 0
    if echo.model_name:
        infer.parameters["model_name"].string_param = echo.model_name
    if echo.instance_ip:
        infer.parameters["model_instance_ip"].string_param = echo.instance_ip
    if echo.hostname:
        infer.parameters["model_hostname"].string_param = echo.hostname
    if echo.scheduler_request_id:
        infer.parameters["scheduler_request_id"].string_param = (
            echo.scheduler_request_id
        )

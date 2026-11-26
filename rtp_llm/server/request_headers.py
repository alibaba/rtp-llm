import json
from typing import Any, Dict, Mapping, Optional

# These fields come from DashSC gateway attributes. Keep them separate from the
# public HTTP ingress allowlist: only the gateway attests to gateway_check.
GREENNET_FIELDS = (
    "model",
    "x-real-ip",
    "x-dashscope-service",
    "x-dashscope-uid",
    "x-dashscope-subuid",
    "x-dashscope-workspaceid",
    "x-dashscope-workspace",
    "x-dashscope-loggingconsent",
    "x-dashscope-inner-loggingconsent",
    "x-dashscope-inner-batch-req",
    "x-dashscope-inner-streammode",
    "x-dashscope-inner-source",
    "x-dashscope-inner-workspace-meta",
    "x-dashscope-inner-gateway-datainspection",
    "x-dashscope-inner-datainspection",
    "x-dashscope-datainspection",
    "x-dashscope-datainspection-cipservice",
    "x-dashscope-apikeyid",
)

REQUEST_HEADER_NAMES = (
    "user_id",
    "x-dashscope-uid",
    "x-dashscope-service",
    "x-dashscope-apikeyid",
    "x-dashscope-request-id",
    "x-dashscope-requestid",
    "x-request-id",
    "dashscope-request-id",
    "trace-id",
    "traceparent",
    "x-trace-id",
    "trace_id",
    "eagleeye-traceid",
    "x-b3-traceid",
    "x-dashscope-inner-qos-level",
    "x-dashscope-datainspection",
)
CORRELATION_HEADER_NAMES = (
    "x-dashscope-request-id",
    "x-dashscope-requestid",
    "x-request-id",
    "dashscope-request-id",
)
TRACE_HEADER_NAMES = (
    "x-trace-id",
    "trace_id",
    "trace-id",
    "eagleeye-traceid",
    "x-b3-traceid",
    "traceparent",
)


def _normalize_header_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def extract_request_headers(
    headers: Optional[Mapping[str, Any]],
) -> Dict[str, str]:
    if not headers:
        return {}

    try:
        lookup: Dict[str, str] = {}
        for key, value in headers.items():
            name = str(key).lower()
            if name in REQUEST_HEADER_NAMES:
                if name == "x-dashscope-datainspection" and isinstance(value, dict):
                    value = json.dumps(value, separators=(",", ":"))
                normalized = _normalize_header_value(value)
                if normalized is not None:
                    lookup[name] = normalized
    except Exception:
        return {}

    return {name: lookup[name] for name in REQUEST_HEADER_NAMES if name in lookup}


def normalize_request_headers(headers: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    return extract_request_headers(headers)


def extract_correlation_request_id(headers: Optional[Mapping[str, Any]]) -> str:
    normalized = extract_request_headers(headers)
    for header_name in CORRELATION_HEADER_NAMES:
        value = normalized.get(header_name)
        if value:
            return value
    return ""


def extract_trace_id(headers: Optional[Mapping[str, Any]]) -> str:
    normalized = extract_request_headers(headers)
    for header_name in TRACE_HEADER_NAMES:
        value = normalized.get(header_name)
        if not value:
            continue
        if header_name == "traceparent":
            parts = value.split("-")
            if len(parts) >= 2 and parts[1]:
                return parts[1]
        return value
    return ""


def dashscope_greennet_metadata(headers: Optional[Mapping[str, Any]]) -> tuple:
    """Carry identity and the external request ID to downstream ViT / LLM RPCs."""
    normalized = extract_request_headers(headers)
    metadata = tuple(
        (name, normalized[name])
        for name in ("x-dashscope-uid", "x-dashscope-service")
        if name in normalized
    )
    request_id = extract_correlation_request_id(normalized)
    if request_id:
        metadata += (("x-dashscope-request-id", request_id),)
    return metadata


def greennet_request_fields(headers: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    fields = {}
    if not headers:
        return fields
    normalized = {str(k).lower(): v for k, v in headers.items()}
    for name in GREENNET_FIELDS:
        value = normalized.get(name)
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        if isinstance(value, str) and value.strip():
            fields[name] = value.strip()
    request_id = extract_correlation_request_id(headers)
    if request_id:
        fields["x-dashscope-requestid"] = request_id
    return fields


def write_inspection_context(message, headers) -> Optional[bool]:
    message.inspection_headers.update(greennet_request_fields(headers))
    enabled = extract_input_inspection_enabled(headers)
    if enabled is not None:
        message.input_inspection_enabled.value = enabled
    return enabled


def extract_workspace_region(ds_attrs: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Extract only region from workspace metadata in decoded ds_header_attributes.

    Missing or malformed metadata and blank/non-string regions return None.
    No alternate headers or nested request paths are consulted.
    """
    if not ds_attrs:
        return None
    raw = ds_attrs.get("x-dashscope-inner-workspace-meta")
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            meta = json.loads(raw)
        except (TypeError, ValueError):
            return None
    elif isinstance(raw, dict):
        meta = raw
    else:
        return None
    if not isinstance(meta, dict):
        return None
    region = meta.get("region")
    if isinstance(region, str) and region.strip():
        return region.strip()
    return None


_DATA_INSPECTION_HEADER = "x-dashscope-datainspection"
_INSPECTION_DISABLED_VALUES = frozenset(
    {"disable", "disabled", "false", "0", "off", "no", "n"}
)
_INSPECTION_ENABLED_VALUES = frozenset(
    {"enable", "enabled", "true", "1", "on", "yes", "y"}
)


def extract_input_inspection_enabled(
    headers: Optional[Mapping[str, Any]],
) -> Optional[bool]:
    """Use the gateway decision first, then the internal and user policies."""
    if not headers:
        return None
    normalized = {str(key).lower(): value for key, value in headers.items()}
    gateway = "x-dashscope-inner-gateway-datainspection"
    for name in (gateway, "x-dashscope-inner-datainspection", _DATA_INSPECTION_HEADER):
        raw = normalized.get(name)
        if raw is None or raw == "":
            continue
        try:
            payload = raw if isinstance(raw, dict) else json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if name == gateway and payload.get("gateway_check") is True:
            return False
        directive = payload.get("input")
        if isinstance(directive, str):
            directive = directive.strip().lower()
            if directive in _INSPECTION_DISABLED_VALUES:
                return False
            if directive in _INSPECTION_ENABLED_VALUES or directive in ("high", "cip"):
                return True
        return None
    return None


def extract_qos_priority(input: Any) -> int:
    """Use the same QoS priority for scheduling and Worker RPCs."""
    headers = getattr(input, "headers", None)
    if headers:
        value = headers.get("x-dashscope-inner-qos-level")
        if value is not None:
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                pass
    config = getattr(input, "generate_config", None)
    priority = getattr(config, "qos_priority", None)
    if priority is not None and priority > 0:
        return priority
    return 50

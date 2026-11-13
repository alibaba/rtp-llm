from typing import Any, Dict, Mapping, Optional

VIT_SOURCE_REQUEST_ID_METADATA = "x-rtp-source-request-id-bin"

REQUEST_HEADER_NAMES = (
    "user_id",
    "x-dashscope-uid",
    "x-dashscope-service",
    "x-dashscope-apikeyid",
    "x-dashscope-request-id",
    "x-request-id",
    "dashscope-request-id",
    "trace-id",
    "traceparent",
    "x-trace-id",
    "trace_id",
    "eagleeye-traceid",
    "x-b3-traceid",
    "x-dashscope-inner-qos-level",
)
CORRELATION_HEADER_NAMES = (
    "x-dashscope-request-id",
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
        lookup = {str(key).lower(): value for key, value in headers.items()}
    except Exception:
        return {}

    result: Dict[str, str] = {}
    for header_name in REQUEST_HEADER_NAMES:
        value = _normalize_header_value(lookup.get(header_name))
        if value is not None:
            result[header_name] = value
    return result


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
    """Carry request UID and service identity to downstream ViT / LLM RPCs."""
    normalized = extract_request_headers(headers)
    return tuple(
        (name, normalized[name])
        for name in ("x-dashscope-uid", "x-dashscope-service")
        if name in normalized
    )


def vit_source_request_metadata(source_request_id: str) -> tuple:
    value = source_request_id.encode("utf-8")[:256]
    return ((VIT_SOURCE_REQUEST_ID_METADATA, value),) if value else ()


def vit_forwarding_metadata(headers: Optional[Mapping[str, Any]]) -> tuple:
    """Forward only ViT identity and bounded binary request correlation."""
    metadata = dashscope_greennet_metadata(headers)
    source_id = (headers or {}).get(VIT_SOURCE_REQUEST_ID_METADATA)
    if isinstance(source_id, bytes) and source_id:
        metadata += ((VIT_SOURCE_REQUEST_ID_METADATA, source_id[:256]),)
    return metadata

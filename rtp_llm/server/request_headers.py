from typing import Any, Dict, Mapping, Optional

QOS_PRIORITY_HEADER = "x-dashscope-inner-qos-level"
DEFAULT_QOS_PRIORITY = 50
MIN_QOS_PRIORITY = 1
MAX_QOS_PRIORITY = 100

REQUEST_HEADER_NAMES = (
    "user_id",
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
    QOS_PRIORITY_HEADER,
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


def resolve_qos_priority(
    headers: Optional[Mapping[str, Any]],
    generate_config: Any = None,
) -> int:
    """Resolve the priority carried to both FlexLB and the engine.

    A valid HTTP header wins. ``generate_config.qos_priority`` is the IPC-safe
    fallback used when request headers are no longer available. Invalid or
    missing values resolve to the normal Auto-TPM priority (50), matching the
    FlexLB wire contract.
    """

    normalized_headers = extract_request_headers(headers)
    candidates = (
        normalized_headers.get(QOS_PRIORITY_HEADER),
        getattr(generate_config, "qos_priority", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            priority = int(str(candidate).strip())
        except (TypeError, ValueError):
            continue
        if MIN_QOS_PRIORITY <= priority <= MAX_QOS_PRIORITY:
            return priority
    return DEFAULT_QOS_PRIORITY

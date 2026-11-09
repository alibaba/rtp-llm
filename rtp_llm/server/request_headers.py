import json
from typing import Any, Dict, Mapping, Optional

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


def extract_workspace_region(headers: Optional[Mapping[str, Any]]) -> Optional[str]:
    """读取网关注入的 x_dashscope_inner_workspace_meta, 提取 region 字段。

    该 header 是 JSON 字符串, 形如
    '{"model_logging_policy":"default","region":"ap-southeast-1",...}'。
    与 dashscope-serving request_region.extract_request_region 一致: 解析失败、
    非 dict、region 非字符串时返回 None (等价无 workspace 信息)。
    """
    if not headers:
        return None
    raw = headers.get("x_dashscope_inner_workspace_meta")
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

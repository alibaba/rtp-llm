from typing import Any, Dict, Mapping, Optional

REQUEST_HEADER_NAMES = ("user_id", "x-dashscope-apikeyid")


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

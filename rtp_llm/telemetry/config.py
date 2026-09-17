"""Trace 启动配置：唯一入口 RTP_LLM_TRACE_CONFIG，不依赖 OpenTelemetry。"""

import ipaddress
import json
import logging
import math
import os
import re
import ssl
import threading
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Tuple
from urllib.parse import urlsplit

CONFIG_ENV = "RTP_LLM_TRACE_CONFIG"
_STRING_FIELDS = {
    "endpoint",
    "certificate",
}
_INT_DEFAULTS = {
    "max_queue_size": 2048,
    "max_export_batch_size": 512,
    "schedule_delay_ms": 5000,
    "http_timeout_ms": 3000,
}
_FIELDS = _STRING_FIELDS | set(_INT_DEFAULTS) | {"enabled", "sampler_ratio", "headers"}
_RESERVED_HEADERS = {
    "host",
    "content-length",
    "content-type",
    "connection",
    "transfer-encoding",
    "content-encoding",
    "trailer",
    "upgrade",
    "keep-alive",
    "te",
}
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_LOGGER = logging.getLogger(__name__)


class TraceConfigError(ValueError):
    """仅携带固定字段名和错误码，不保留输入或第三方异常。"""

    def __init__(self, field_name: str, code: str):
        self.field_name = field_name if field_name in _FIELDS else "config"
        self.code = code
        super().__init__(f"{self.field_name}:{code}")


@dataclass(frozen=True, repr=False)
class TraceConfig:
    enabled: bool = False
    sampler_ratio: float = 1.0
    endpoint: str = ""
    headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    certificate: str = ""
    service_name: str = ""
    scope_version: str = ""
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    schedule_delay_ms: int = 5000
    http_timeout_ms: int = 3000
    source: str = "disabled"

    def __repr__(self) -> str:
        return f"TraceConfig(enabled={self.enabled}, source={self.source})"


def _pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise TraceConfigError("config", "duplicate_key")
        result[key] = value
    return result


def _invalid_constant(_value: str) -> NoReturn:
    raise TraceConfigError("config", "invalid_json")


def _decode(raw: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            raw, object_pairs_hook=_pairs, parse_constant=_invalid_constant
        )
    except TraceConfigError:
        raise
    except (ValueError, RecursionError):
        raise TraceConfigError("config", "invalid_json") from None
    if not isinstance(value, dict):
        raise TraceConfigError("config", "expected_object")
    return value


def _lenient_bool(value: Any, name: str) -> bool:
    """接受等价写法：bool / 0、1 / "true"、"false"、"0"、"1"（大小写不敏感）。"""
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if type(value) is str:
        text = value.strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
    raise TraceConfigError(name, "expected_boolean")


def _lenient_number(value: Any, name: str) -> float:
    """接受等价写法：数字 / 数字字符串。bool 与 None 仍然拒绝。"""
    if type(value) is int or type(value) is float:
        try:
            return float(value)
        except OverflowError:
            raise TraceConfigError(name, "out_of_range") from None
    if type(value) is str:
        try:
            text = value.strip()
            if re.fullmatch(
                r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text
            ):
                return float(text)
        except ValueError:
            pass
    raise TraceConfigError(name, "expected_number")


def _lenient_int(value: Any, name: str) -> int:
    """接受等价写法：整数 / 浮点 / 数字字符串；浮点按 Java 的 coerce 行为截断。"""
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TraceConfigError(name, "out_of_range")
        return int(value)
    if type(value) is str:
        try:
            text = value.strip()
            if re.fullmatch(r"[+-]?[0-9]+", text):
                return int(text)
        except ValueError:
            pass
    raise TraceConfigError(name, "expected_integer")


def _string(value: Any, name: str) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in (int, float):
        return str(value)
    if not isinstance(value, str):
        raise TraceConfigError(name, "expected_string")
    return value.strip()


def _headers(value: Any) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise TraceConfigError("headers", "expected_object")
    result = {}
    for name, item in value.items():
        if not _HEADER_NAME.fullmatch(name) or name.lower() in _RESERVED_HEADERS:
            raise TraceConfigError("headers", "invalid_header")
        name = name.lower()
        if name in result:
            raise TraceConfigError("headers", "duplicate_header")
        if (
            not isinstance(item, str)
            or not item.strip()
            or item != item.strip()
            or any(
                ord(c) < 32 and c != "\t" or ord(c) == 127 or ord(c) > 255 for c in item
            )
        ):
            raise TraceConfigError("headers", "invalid_header")
        result[name] = item
    return MappingProxyType(result)


def _endpoint(value: str) -> None:
    try:
        url = urlsplit(value)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or "#" in value
            or any(ord(c) <= 32 or ord(c) >= 127 for c in value)
            or "\\" in value
            or re.search(r"%(?![0-9a-fA-F]{2})", value)
        ):
            raise ValueError()
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError()
        if url.netloc.endswith(":"):
            raise ValueError()
        if ":" in url.hostname:
            ipaddress.IPv6Address(url.hostname)
        else:
            host = url.hostname.removesuffix(".")
            labels = host.split(".")
            if any(
                not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
                for label in labels
            ):
                raise ValueError()
            # 与 Java URI 的 host 判定一致：点分数字必须是完整 IPv4。
            if len(labels) > 1 and labels[-1][0].isdigit():
                ipaddress.IPv4Address(host)
    except ValueError:
        raise TraceConfigError("endpoint", "invalid_endpoint") from None


def parse_trace_config(raw: Optional[str], tp_rank: int = 0) -> TraceConfig:
    """纯解析入口；关闭或非负责 rank 跳过导出配置校验。"""
    if not raw or not raw.strip():
        return TraceConfig()
    values = _decode(raw)
    if values.keys() - _FIELDS:
        raise TraceConfigError("config", "unknown_field")
    enabled = _lenient_bool(values.get("enabled", False), "enabled")
    if not enabled or tp_rank != 0:
        return TraceConfig()
    strings = {name: _string(values.get(name, ""), name) for name in _STRING_FIELDS}
    ratio = _lenient_number(values.get("sampler_ratio", 1.0), "sampler_ratio")
    if not 0 <= ratio <= 1 or not math.isfinite(ratio):
        raise TraceConfigError("sampler_ratio", "out_of_range")
    ints = {}
    for name, default in _INT_DEFAULTS.items():
        value = _lenient_int(values.get(name, default), name)
        # 三端一致的上界，避免 Java int 和时间单位转换溢出。
        if not 0 < value <= 2147483647:
            raise TraceConfigError(name, "out_of_range")
        ints[name] = value
    if ints["max_export_batch_size"] > ints["max_queue_size"]:
        raise TraceConfigError("max_export_batch_size", "batch_exceeds_queue")
    headers = _headers(values.get("headers", {}))
    endpoint, certificate = strings["endpoint"], strings["certificate"]
    source = "manual"
    if not endpoint or not headers:
        raise TraceConfigError("config", "incomplete_manual")
    _endpoint(endpoint)
    if certificate:
        try:
            ssl.create_default_context(cafile=certificate)
        except (OSError, ValueError):
            raise TraceConfigError("certificate", "invalid_certificate") from None
    return TraceConfig(
        enabled=True,
        sampler_ratio=float(ratio),
        endpoint=endpoint,
        headers=headers,
        certificate=certificate,
        source=source,
        **ints,
    )


def package_scope_version() -> str:
    try:
        from importlib.metadata import version

        return version("rtp_llm")
    except Exception:
        return ""


_cache_lock = threading.Lock()
_cache_pid: Optional[int] = None
_cache = TraceConfig()


def load_trace_config(role: str, tp_rank: int = 0) -> TraceConfig:
    """每进程只读一次；错误仅关闭 Trace，最多告警一次，不打印原输入。"""
    global _cache_pid, _cache
    if tp_rank != 0:
        return TraceConfig()
    with _cache_lock:
        if _cache_pid == os.getpid():
            return _cache
        try:
            config = parse_trace_config(os.environ.get(CONFIG_ENV), tp_rank)
            if config.enabled and not config.scope_version:
                config = replace(config, scope_version=package_scope_version())
            _cache = config
        except TraceConfigError as error:
            _LOGGER.warning(
                "Trace 已关闭 role=%s field=%s reason=%s",
                role,
                error.field_name,
                error.code,
            )
            _cache = TraceConfig()
        except Exception:
            _LOGGER.warning(
                "Trace 已关闭 role=%s field=config reason=config_unavailable", role
            )
            _cache = TraceConfig()
        _cache_pid = os.getpid()
        return _cache

"""Build an isolated KVMeta object-client config from RTP's KVCM settings.

The regular fixed-block KV cache client and the exact-size embedding client
connect to the same KVCM deployment.  This module snapshots and validates the
existing ``RECO_*`` configuration, prefixes the object-client identity with
``kve_``, and replaces only the fixed-block layout with KVMeta's one-byte
schema marker.

This module intentionally has no dependency on RTP's native extension or the
KVCM wheel.  Configuration errors therefore fail before client registration
or any storage mutation.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

KVE_INSTANCE_PREFIX = "kve_"

_MAX_CLIENT_CONFIG_BYTES = 1024 * 1024
_MAX_ADDRESSES = 64
_MAX_ADDRESS_BYTES = 1024
_MAX_INSTANCE_ID_BYTES = 512
_MAX_INSTANCE_GROUP_BYTES = 512
_MAX_USER_DATA_BYTES = 64 * 1024
_MAX_OBJECT_BYTES = 1024 * 1024 * 1024
_MAX_CALL_TIMEOUT_MS = 600_000
_MAX_WRITE_TIMEOUT_SECONDS = 1800
_MAX_INT32 = (1 << 31) - 1
_MAX_UINT32 = (1 << 32) - 1
_MIN_SDK_QUEUE_SIZE = 64
_DEFAULT_WRITE_TIMEOUT_SECONDS = 30
_KV_META_MODEL_NAME = "__kv_meta_object__"
_KV_META_DTYPE = "opaque_bytes"
_KV_META_DEPLOYMENT_EXTRA = "kv_meta_v1"


class RtpKvMetaObjectConfigError(ValueError):
    """The existing RTP KVCM settings cannot safely serve EMB objects."""


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True)
class RtpKvMetaObjectClientConfig:
    """Validated configuration consumed by :class:`RtpKvMetaObjectClient`.

    Most callers should use ``RtpKvMetaObjectClient()`` or
    ``RtpKvMetaObjectClient.from_kv_cache_config()`` instead of constructing
    this object directly.
    """

    addresses: tuple[str, ...]
    instance_id: str
    instance_group: str
    user_data: str = field(repr=False)
    transfer_client_config: str = field(repr=False)
    call_timeout_ms: int
    write_timeout_seconds: int
    max_object_bytes: int = _MAX_OBJECT_BYTES

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        vipserver_resolver: Callable[[str], Iterable[Any]] | None = None,
        max_object_bytes: int = _MAX_OBJECT_BYTES,
    ) -> RtpKvMetaObjectClientConfig:
        """Snapshot the existing split ``RECO_*`` environment or config map."""

        settings = _reco_settings_from_env(environ)
        return _config_from_reco_settings(
            settings,
            vipserver_resolver=vipserver_resolver,
            max_object_bytes=max_object_bytes,
        )

    @classmethod
    def from_kv_cache_config(
        cls,
        kv_cache_config: Any,
        *,
        vipserver_resolver: Callable[[str], Iterable[Any]] | None = None,
        max_object_bytes: int = _MAX_OBJECT_BYTES,
    ) -> RtpKvMetaObjectClientConfig:
        """Snapshot an already parsed RTP ``KVCacheConfig`` object."""

        settings = _reco_settings_from_kv_cache_config(kv_cache_config)
        return _config_from_reco_settings(
            settings,
            vipserver_resolver=vipserver_resolver,
            max_object_bytes=max_object_bytes,
        )


@dataclass(frozen=True)
class _RecoSettings:
    client_config: str
    enable_vipserver: bool
    vipserver_domain: str
    server_address: str
    instance_group: str
    instance_id_salt: str
    meta_channel_retry_time: int
    meta_channel_connection_timeout: int
    meta_channel_call_timeout: int
    storage_thread_num: int
    storage_queue_size: int
    put_timeout_ms: int
    get_timeout_ms: int
    model_sdk_config: str
    model_user_data: str


_ENV_DEFAULTS = {
    "RECO_ENABLE_VIPSERVER": "0",
    "RECO_VIPSERVER_DOMAIN": "",
    "RECO_SERVER_ADDRESS": "",
    "RECO_INSTANCE_GROUP": "default",
    "RECO_INSTANCE_ID_SALT": "",
    "RECO_META_CHANNEL_RETRY_TIME": "3",
    "RECO_META_CHANNEL_CONNECTION_TIMEOUT": "6000",
    "RECO_META_CHANNEL_CALL_TIMEOUT": "1500",
    "RECO_STORAGE_THREAD_NUM": "4",
    "RECO_STORAGE_QUEUE_SIZE": "2000",
    "RECO_PUT_TIMEOUT_MS": "12000",
    "RECO_GET_TIMEOUT_MS": "12000",
    "RECO_MODEL_SDK_CONFIG": '[{"type":"local","sdk_log_level":"DEBUG"}]',
    "RECO_MODEL_USER_DATA": "",
}


def _explicit_reco_settings(client_config: str) -> _RecoSettings:
    """Represent explicit-map precedence without reading split settings."""

    return _RecoSettings(
        client_config=client_config,
        enable_vipserver=False,
        vipserver_domain="",
        server_address="",
        instance_group="",
        instance_id_salt="",
        meta_channel_retry_time=0,
        meta_channel_connection_timeout=0,
        meta_channel_call_timeout=1,
        storage_thread_num=1,
        storage_queue_size=_MIN_SDK_QUEUE_SIZE,
        put_timeout_ms=1,
        get_timeout_ms=1,
        model_sdk_config="[]",
        model_user_data="",
    )


def _utf8_size(value: str, description: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise RtpKvMetaObjectConfigError(
            f"{description} must be valid UTF-8"
        ) from error


def _bounded_text(
    value: Any, description: str, maximum: int, *, allow_empty: bool = False
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise RtpKvMetaObjectConfigError(f"{description} must be {qualifier}")
    if _utf8_size(value, description) > maximum:
        raise RtpKvMetaObjectConfigError(f"{description} exceeds {maximum} UTF-8 bytes")
    return value


def _positive_int(value: Any, description: str, maximum: int) -> int:
    if type(value) is not int:
        raise RtpKvMetaObjectConfigError(f"{description} must be an integer")
    if not 0 < value <= maximum:
        raise RtpKvMetaObjectConfigError(f"{description} is outside its valid range")
    return value


def _non_negative_int(value: Any, description: str, maximum: int) -> int:
    if type(value) is not int:
        raise RtpKvMetaObjectConfigError(f"{description} must be an integer")
    if not 0 <= value <= maximum:
        raise RtpKvMetaObjectConfigError(f"{description} is outside its valid range")
    return value


def _required_mapping(value: Any, description: str) -> dict:
    if not isinstance(value, dict):
        raise RtpKvMetaObjectConfigError(f"{description} must be a JSON object")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey()
        result[key] = value
    return result


def _reject_json_constant(_value: str):
    raise ValueError()


def _parse_json(serialized: Any, description: str) -> Any:
    if not isinstance(serialized, str) or not serialized:
        raise RtpKvMetaObjectConfigError(f"{description} must be a non-empty string")
    if _utf8_size(serialized, description) > _MAX_CLIENT_CONFIG_BYTES:
        raise RtpKvMetaObjectConfigError(f"{description} exceeds the supported size")
    try:
        return json.loads(
            serialized,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, RecursionError):
        # Parser details can contain endpoint, identity, or backend secrets.
        raise RtpKvMetaObjectConfigError(
            f"{description} must contain valid strict JSON"
        ) from None


def _parse_shared_config(serialized: Any) -> dict:
    return _required_mapping(
        _parse_json(serialized, "RECO_CLIENT_CONFIG"), "RECO_CLIENT_CONFIG"
    )


def _parse_sdk_backend_configs(serialized: Any) -> list:
    parsed = _parse_json(serialized, "RECO_MODEL_SDK_CONFIG")
    if not isinstance(parsed, list):
        raise RtpKvMetaObjectConfigError(
            "RECO_MODEL_SDK_CONFIG must contain a JSON array"
        )
    return parsed


def _select_primary_config(config_map: dict) -> dict:
    if not config_map:
        raise RtpKvMetaObjectConfigError("RECO_CLIENT_CONFIG must not be an empty map")
    if "" in config_map:
        selected = config_map[""]
    elif len(config_map) == 1:
        selected = next(iter(config_map.values()))
    else:
        # The fixed-block C++ wrapper selects std::map::begin(). Requiring an
        # explicit primary avoids selecting a different client in Python.
        raise RtpKvMetaObjectConfigError(
            "RECO_CLIENT_CONFIG with multiple entries requires an empty-key "
            "primary client config"
        )
    return _required_mapping(selected, "primary RECO_CLIENT_CONFIG entry")


def _validate_endpoint(value: Any) -> str:
    endpoint = _bounded_text(value, "KVCM endpoint", _MAX_ADDRESS_BYTES)
    if any(character.isspace() for character in endpoint) or "\x00" in endpoint:
        raise RtpKvMetaObjectConfigError(
            "KVCM endpoint contains invalid whitespace or NUL"
        )
    return endpoint


def _validate_addresses(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise RtpKvMetaObjectConfigError("KVCM address must be a JSON array")
    if not 0 < len(values) <= _MAX_ADDRESSES:
        raise RtpKvMetaObjectConfigError(
            "KVCM address count is outside its valid range"
        )
    addresses = tuple(_validate_endpoint(value) for value in values)
    if len(set(addresses)) != len(addresses):
        raise RtpKvMetaObjectConfigError("KVCM addresses must be unique")
    return addresses


def _default_vipserver_resolver(domain: str) -> Iterable[Any]:
    # RTP's VIPServer import starts refresh machinery. Keep it lazy so static
    # address users and module imports remain side-effect free.
    from rtp_llm.vipserver import get_host_list_by_domain_now

    return get_host_list_by_domain_now(domain)


def _resolve_vipserver_addresses(
    domain: str, resolver: Callable[[str], Iterable[Any]]
) -> tuple[str, ...]:
    try:
        resolved = resolver(domain)
        if isinstance(resolved, (str, bytes)):
            raise TypeError()
        hosts = tuple(islice(iter(resolved), _MAX_ADDRESSES + 1))
    except Exception:  # noqa: BLE001 - isolate arbitrary resolver implementations.
        raise RtpKvMetaObjectConfigError(
            "failed to resolve KVCM VIPServer domain"
        ) from None
    if not 0 < len(hosts) <= _MAX_ADDRESSES:
        raise RtpKvMetaObjectConfigError(
            "KVCM VIPServer domain returned an invalid endpoint count"
        )

    addresses = []
    for host in hosts:
        try:
            ip = host.ip
            port = host.port
        except Exception:  # noqa: BLE001 - host attributes may be provider proxies.
            raise RtpKvMetaObjectConfigError(
                "KVCM VIPServer returned a malformed endpoint"
            ) from None
        if not isinstance(ip, str) or not ip or type(port) is not int:
            raise RtpKvMetaObjectConfigError(
                "KVCM VIPServer returned a malformed endpoint"
            )
        try:
            ipaddress.IPv4Address(ip)
        except ipaddress.AddressValueError:
            raise RtpKvMetaObjectConfigError(
                "KVCM VIPServer returned a malformed endpoint"
            ) from None
        if not 0 < port <= 65535:
            raise RtpKvMetaObjectConfigError(
                "KVCM VIPServer returned a malformed endpoint"
            )
        addresses.append(_validate_endpoint(f"{ip}:{port}"))
    if len(set(addresses)) != len(addresses):
        raise RtpKvMetaObjectConfigError("KVCM VIPServer returned duplicate endpoints")
    return tuple(addresses)


def _derive_identity(primary: dict, field: str, maximum: int) -> str:
    base = _bounded_text(primary.get(field), f"KVCM {field}", maximum)
    derived = KVE_INSTANCE_PREFIX + base
    _bounded_text(derived, f"KVMeta {field}", maximum)
    return derived


def _validate_sdk_config(primary: dict) -> int:
    sdk_config = _required_mapping(primary.get("sdk_config"), "KVCM sdk_config")
    _positive_int(sdk_config.get("thread_num"), "KVCM SDK thread_num", _MAX_INT32)
    queue_size = _positive_int(
        sdk_config.get("queue_size"), "KVCM SDK queue_size", _MAX_INT32
    )
    if queue_size < _MIN_SDK_QUEUE_SIZE:
        raise RtpKvMetaObjectConfigError(
            f"KVCM SDK queue_size must be at least {_MIN_SDK_QUEUE_SIZE} for KVMeta"
        )
    backend_configs = sdk_config.get("sdk_backend_configs")
    if not isinstance(backend_configs, list):
        raise RtpKvMetaObjectConfigError(
            "KVCM sdk_backend_configs must be a JSON array"
        )
    for index, backend_config in enumerate(backend_configs):
        if not isinstance(backend_config, dict):
            raise RtpKvMetaObjectConfigError(
                f"KVCM sdk_backend_configs[{index}] must be a JSON object"
            )
        _bounded_text(
            backend_config.get("type"),
            f"KVCM sdk_backend_configs[{index}].type",
            128,
        )

    timeout_config = _required_mapping(
        sdk_config.get("timeout_config"), "KVCM SDK timeout_config"
    )
    put_timeout_ms = _positive_int(
        timeout_config.get("put_timeout_ms"),
        "KVCM SDK put_timeout_ms",
        _MAX_INT32,
    )
    _positive_int(
        timeout_config.get("get_timeout_ms"),
        "KVCM SDK get_timeout_ms",
        _MAX_INT32,
    )
    return put_timeout_ms


def _parse_env_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "t"}:
        return True
    if normalized in {"0", "false", "no", "off", "f"}:
        return False
    raise RtpKvMetaObjectConfigError(f"{name} must be a boolean")


def _parse_env_int(value: str, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RtpKvMetaObjectConfigError(f"{name} must be an integer") from None


def _reco_settings_from_env(
    environ: Mapping[str, str] | None,
) -> _RecoSettings:
    source = os.environ if environ is None else environ
    if not isinstance(source, Mapping):
        raise TypeError("environ must be a mapping")
    try:
        client_config = source.get("RECO_CLIENT_CONFIG", "")
    except Exception:  # noqa: BLE001 - mappings can implement arbitrary get().
        raise RtpKvMetaObjectConfigError(
            "failed to snapshot RTP KVCM environment"
        ) from None
    if not isinstance(client_config, str):
        raise RtpKvMetaObjectConfigError("RECO_CLIENT_CONFIG must be a string")
    if client_config:
        # Match RTP's fixed-block precedence: an explicit config map owns the
        # whole client configuration, so stale split fields are not inspected.
        return _explicit_reco_settings(client_config)

    try:
        snapshot = {
            name: source.get(name, default) for name, default in _ENV_DEFAULTS.items()
        }
    except Exception:  # noqa: BLE001 - snapshot the mapping behind one boundary.
        raise RtpKvMetaObjectConfigError(
            "failed to snapshot RTP KVCM environment"
        ) from None
    for name, value in snapshot.items():
        if not isinstance(value, str):
            raise RtpKvMetaObjectConfigError(f"{name} must be a string")

    return _RecoSettings(
        client_config="",
        enable_vipserver=_parse_env_bool(
            snapshot["RECO_ENABLE_VIPSERVER"], "RECO_ENABLE_VIPSERVER"
        ),
        vipserver_domain=snapshot["RECO_VIPSERVER_DOMAIN"],
        server_address=snapshot["RECO_SERVER_ADDRESS"],
        instance_group=snapshot["RECO_INSTANCE_GROUP"],
        instance_id_salt=snapshot["RECO_INSTANCE_ID_SALT"],
        meta_channel_retry_time=_parse_env_int(
            snapshot["RECO_META_CHANNEL_RETRY_TIME"],
            "RECO_META_CHANNEL_RETRY_TIME",
        ),
        meta_channel_connection_timeout=_parse_env_int(
            snapshot["RECO_META_CHANNEL_CONNECTION_TIMEOUT"],
            "RECO_META_CHANNEL_CONNECTION_TIMEOUT",
        ),
        meta_channel_call_timeout=_parse_env_int(
            snapshot["RECO_META_CHANNEL_CALL_TIMEOUT"],
            "RECO_META_CHANNEL_CALL_TIMEOUT",
        ),
        storage_thread_num=_parse_env_int(
            snapshot["RECO_STORAGE_THREAD_NUM"], "RECO_STORAGE_THREAD_NUM"
        ),
        storage_queue_size=_parse_env_int(
            snapshot["RECO_STORAGE_QUEUE_SIZE"], "RECO_STORAGE_QUEUE_SIZE"
        ),
        put_timeout_ms=_parse_env_int(
            snapshot["RECO_PUT_TIMEOUT_MS"], "RECO_PUT_TIMEOUT_MS"
        ),
        get_timeout_ms=_parse_env_int(
            snapshot["RECO_GET_TIMEOUT_MS"], "RECO_GET_TIMEOUT_MS"
        ),
        model_sdk_config=snapshot["RECO_MODEL_SDK_CONFIG"],
        model_user_data=snapshot["RECO_MODEL_USER_DATA"],
    )


def _snapshot_attribute(value: Any, attribute: str) -> Any:
    try:
        return getattr(value, attribute)
    except Exception:  # noqa: BLE001 - config properties can execute provider code.
        raise RtpKvMetaObjectConfigError(
            f"RTP KVCacheConfig is missing a readable {attribute}"
        ) from None


def _reco_settings_from_kv_cache_config(kv_cache_config: Any) -> _RecoSettings:
    if kv_cache_config is None:
        raise TypeError("kv_cache_config must not be None")
    client_config = _snapshot_attribute(kv_cache_config, "reco_client_config")
    if not isinstance(client_config, str):
        raise RtpKvMetaObjectConfigError("RECO_CLIENT_CONFIG must be a string")
    if client_config:
        # Do not read split-field properties when the fixed-block client would
        # select the explicit map. Besides preserving precedence, this avoids
        # surprising side effects from unrelated dynamic config properties.
        return _explicit_reco_settings(client_config)
    names = (
        "reco_enable_vipserver",
        "reco_vipserver_domain",
        "reco_server_address",
        "reco_instance_group",
        "reco_instance_id_salt",
        "reco_meta_channel_retry_time",
        "reco_meta_channel_connection_timeout",
        "reco_meta_channel_call_timeout",
        "reco_storage_thread_num",
        "reco_storage_queue_size",
        "reco_put_timeout_ms",
        "reco_get_timeout_ms",
        "reco_model_sdk_config",
        "reco_model_user_data",
    )
    values = {name: _snapshot_attribute(kv_cache_config, name) for name in names}
    return _RecoSettings(
        client_config="",
        enable_vipserver=values["reco_enable_vipserver"],
        vipserver_domain=values["reco_vipserver_domain"],
        server_address=values["reco_server_address"],
        instance_group=values["reco_instance_group"],
        instance_id_salt=values["reco_instance_id_salt"],
        meta_channel_retry_time=values["reco_meta_channel_retry_time"],
        meta_channel_connection_timeout=values["reco_meta_channel_connection_timeout"],
        meta_channel_call_timeout=values["reco_meta_channel_call_timeout"],
        storage_thread_num=values["reco_storage_thread_num"],
        storage_queue_size=values["reco_storage_queue_size"],
        put_timeout_ms=values["reco_put_timeout_ms"],
        get_timeout_ms=values["reco_get_timeout_ms"],
        model_sdk_config=values["reco_model_sdk_config"],
        model_user_data=values["reco_model_user_data"],
    )


def _split_reco_primary_config(settings: _RecoSettings) -> dict:
    instance_group = _bounded_text(
        settings.instance_group,
        "RECO_INSTANCE_GROUP",
        _MAX_INSTANCE_GROUP_BYTES,
    )
    instance_id_salt = _bounded_text(
        settings.instance_id_salt,
        "RECO_INSTANCE_ID_SALT",
        _MAX_INSTANCE_ID_BYTES,
        allow_empty=True,
    )
    instance_id = instance_id_salt or instance_group

    if type(settings.enable_vipserver) is not bool:
        raise RtpKvMetaObjectConfigError("RECO_ENABLE_VIPSERVER must be a boolean")
    vipserver_domain = _bounded_text(
        settings.vipserver_domain,
        "RECO_VIPSERVER_DOMAIN",
        _MAX_ADDRESS_BYTES,
        allow_empty=True,
    )
    server_address = _bounded_text(
        settings.server_address,
        "RECO_SERVER_ADDRESS",
        _MAX_ADDRESS_BYTES,
        allow_empty=True,
    )
    addresses = [] if not server_address else [_validate_endpoint(server_address)]

    # RemoteConnector intentionally accepts zero for retry and connection
    # timeout. Preserve that established contract.
    retry_time = _non_negative_int(
        settings.meta_channel_retry_time,
        "RECO_META_CHANNEL_RETRY_TIME",
        _MAX_UINT32,
    )
    connection_timeout = _non_negative_int(
        settings.meta_channel_connection_timeout,
        "RECO_META_CHANNEL_CONNECTION_TIMEOUT",
        _MAX_UINT32,
    )
    call_timeout = _positive_int(
        settings.meta_channel_call_timeout,
        "RECO_META_CHANNEL_CALL_TIMEOUT",
        _MAX_CALL_TIMEOUT_MS,
    )
    thread_num = _positive_int(
        settings.storage_thread_num, "RECO_STORAGE_THREAD_NUM", _MAX_INT32
    )
    queue_size = _positive_int(
        settings.storage_queue_size, "RECO_STORAGE_QUEUE_SIZE", _MAX_INT32
    )
    put_timeout_ms = _positive_int(
        settings.put_timeout_ms, "RECO_PUT_TIMEOUT_MS", _MAX_INT32
    )
    get_timeout_ms = _positive_int(
        settings.get_timeout_ms, "RECO_GET_TIMEOUT_MS", _MAX_INT32
    )
    user_data = _bounded_text(
        settings.model_user_data,
        "RECO_MODEL_USER_DATA",
        _MAX_USER_DATA_BYTES,
        allow_empty=True,
    )
    return {
        "enable_vipserver": settings.enable_vipserver,
        "vipserver_domain": vipserver_domain,
        "instance_group": instance_group,
        "instance_id": instance_id,
        "address": addresses,
        "block_size": 1,
        "location_spec_infos": {"value": 1},
        "location_spec_groups": {},
        "meta_channel_config": {
            "retry_time": retry_time,
            "connection_timeout": connection_timeout,
            "call_timeout": call_timeout,
        },
        "sdk_config": {
            "thread_num": thread_num,
            "queue_size": queue_size,
            "sdk_backend_configs": _parse_sdk_backend_configs(
                settings.model_sdk_config
            ),
            "timeout_config": {
                "put_timeout_ms": put_timeout_ms,
                "get_timeout_ms": get_timeout_ms,
            },
        },
        "model_deployment": {
            "model_name": _KV_META_MODEL_NAME,
            "dtype": _KV_META_DTYPE,
            "use_mla": False,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "extra": _KV_META_DEPLOYMENT_EXTRA,
            "user_data": user_data,
        },
    }


def _derive_primary_config(
    primary: dict,
    *,
    vipserver_resolver: Callable[[str], Iterable[Any]] | None,
    max_object_bytes: int,
) -> RtpKvMetaObjectClientConfig:
    max_object_bytes = _positive_int(
        max_object_bytes, "KVMeta max_object_bytes", _MAX_OBJECT_BYTES
    )
    instance_group = _derive_identity(
        primary, "instance_group", _MAX_INSTANCE_GROUP_BYTES
    )
    instance_id = _derive_identity(primary, "instance_id", _MAX_INSTANCE_ID_BYTES)

    enable_vipserver = primary.get("enable_vipserver", False)
    if type(enable_vipserver) is not bool:
        raise RtpKvMetaObjectConfigError("KVCM enable_vipserver must be a boolean")
    if enable_vipserver:
        domain = _bounded_text(
            primary.get("vipserver_domain"),
            "KVCM vipserver_domain",
            _MAX_ADDRESS_BYTES,
        )
        addresses = _resolve_vipserver_addresses(
            domain, vipserver_resolver or _default_vipserver_resolver
        )
    else:
        addresses = _validate_addresses(primary.get("address"))

    meta_channel = _required_mapping(
        primary.get("meta_channel_config"), "KVCM meta_channel_config"
    )
    _non_negative_int(
        meta_channel.get("retry_time"),
        "KVCM metadata retry_time",
        _MAX_UINT32,
    )
    _non_negative_int(
        meta_channel.get("connection_timeout"),
        "KVCM metadata connection_timeout",
        _MAX_UINT32,
    )
    call_timeout_ms = _positive_int(
        meta_channel.get("call_timeout"),
        "KVCM metadata call_timeout",
        _MAX_CALL_TIMEOUT_MS,
    )
    put_timeout_ms = _validate_sdk_config(primary)

    model_deployment = _required_mapping(
        primary.get("model_deployment"), "KVCM model_deployment"
    )
    user_data = _bounded_text(
        model_deployment.get("user_data", ""),
        "KVCM model user_data",
        _MAX_USER_DATA_BYTES,
        allow_empty=True,
    )

    minimum_lease_ms = put_timeout_ms + 3 * call_timeout_ms
    write_timeout_seconds = max(
        _DEFAULT_WRITE_TIMEOUT_SECONDS, minimum_lease_ms // 1000 + 1
    )
    if write_timeout_seconds > _MAX_WRITE_TIMEOUT_SECONDS:
        raise RtpKvMetaObjectConfigError(
            "existing KVCM timeouts cannot fit within the KVMeta write lease limit"
        )

    transfer_config = copy.deepcopy(primary)
    transfer_config.update(
        {
            "enable_vipserver": False,
            "vipserver_domain": "",
            "instance_group": instance_group,
            "instance_id": instance_id,
            "block_size": 1,
            "location_spec_infos": {"value": 1},
            "location_spec_groups": {},
            "address": list(addresses),
        }
    )
    try:
        serialized_transfer_config = json.dumps(
            transfer_config,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError):
        raise RtpKvMetaObjectConfigError(
            "KVCM client configuration cannot be serialized safely"
        ) from None
    if (
        _utf8_size(serialized_transfer_config, "KVMeta transfer client config")
        > _MAX_CLIENT_CONFIG_BYTES
    ):
        raise RtpKvMetaObjectConfigError(
            "KVMeta transfer client config exceeds the supported size"
        )

    return RtpKvMetaObjectClientConfig(
        addresses=addresses,
        instance_id=instance_id,
        instance_group=instance_group,
        user_data=user_data,
        transfer_client_config=serialized_transfer_config,
        call_timeout_ms=call_timeout_ms,
        write_timeout_seconds=write_timeout_seconds,
        max_object_bytes=max_object_bytes,
    )


def _config_from_reco_settings(
    settings: _RecoSettings,
    *,
    vipserver_resolver: Callable[[str], Iterable[Any]] | None,
    max_object_bytes: int,
) -> RtpKvMetaObjectClientConfig:
    if settings.client_config:
        primary = _select_primary_config(_parse_shared_config(settings.client_config))
    else:
        primary = _split_reco_primary_config(settings)
    return _derive_primary_config(
        primary,
        vipserver_resolver=vipserver_resolver,
        max_object_bytes=max_object_bytes,
    )

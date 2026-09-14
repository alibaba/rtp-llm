"""Derive the multimodal KVMeta client from RTP's existing KVCM settings.

The fixed-block Meta client and the exact-object KVMeta client talk to the
same KVCM deployment.  An explicit ``RECO_CLIENT_CONFIG`` remains the highest
priority source.  When it is empty, the exact-object client consumes the same
split ``RECO_*`` fields used by ``RemoteConnector::genClientConfig`` instead
of requiring operators to duplicate them.  The exact-object client always
uses a prefixed identity and an isolated one-byte schema marker so it cannot
collide with the fixed-block data path.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any, Optional, Tuple

KVE_INSTANCE_PREFIX = "kve_"
_MM_TRANSPORT_MODE_KVCM = "kvcm"

# Keep these bounds aligned with KvMetaObjectClientConfig and MMKvcmConfig.
_MAX_CLIENT_CONFIG_BYTES = 1024 * 1024
_MAX_ADDRESSES = 64
_MAX_ADDRESS_BYTES = 1024
_MAX_INSTANCE_ID_BYTES = 512
_MAX_INSTANCE_GROUP_BYTES = 512
_MAX_USER_DATA_BYTES = 64 * 1024
_MAX_CALL_TIMEOUT_MS = 600_000
_MAX_WRITE_TIMEOUT_SECONDS = 1800
_MAX_INT32 = (1 << 31) - 1
_MAX_UINT32 = (1 << 32) - 1
_MIN_SDK_QUEUE_SIZE = 64
_DEFAULT_WRITE_TIMEOUT_SECONDS = 30
_KV_META_MODEL_NAME = "__kv_meta_object__"
_KV_META_DTYPE = "opaque_bytes"
_KV_META_DEPLOYMENT_EXTRA = "kv_meta_v1"

# These were briefly introduced for the EMB path.  Silently accepting them
# after switching to the existing RECO configuration would make operators
# believe an override is effective when it is not.
_REMOVED_CLIENT_ENV_VARS = (
    "MM_KVCM_ADDRESSES",
    "MM_KVCM_INSTANCE_ID",
    "MM_KVCM_INSTANCE_GROUP",
    "MM_KVCM_USER_DATA",
    "MM_KVCM_TRANSFER_CLIENT_CONFIG",
    "MM_KVCM_CALL_TIMEOUT_MS",
    "MM_KVCM_WRITE_TIMEOUT_SECONDS",
)


class MMKvcmConfigError(ValueError):
    """An existing KVCM client configuration cannot safely serve KVMeta."""


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True)
class DerivedMMKvcmClientConfig:
    addresses: Tuple[str, ...]
    instance_id: str
    instance_group: str
    user_data: str
    transfer_client_config: str
    call_timeout_ms: int
    write_timeout_seconds: int


def _utf8_size(value: str, description: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise MMKvcmConfigError(f"{description} must be valid UTF-8") from error


def _bounded_text(
    value: Any, description: str, maximum: int, *, allow_empty: bool = False
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise MMKvcmConfigError(f"{description} must be {qualifier}")
    if _utf8_size(value, description) > maximum:
        raise MMKvcmConfigError(f"{description} exceeds {maximum} UTF-8 bytes")
    return value


def _positive_int(value: Any, description: str, maximum: int) -> int:
    if type(value) is not int:
        raise MMKvcmConfigError(f"{description} must be an integer")
    if not 0 < value <= maximum:
        raise MMKvcmConfigError(f"{description} is outside its valid range")
    return value


def _non_negative_int(value: Any, description: str, maximum: int) -> int:
    if type(value) is not int:
        raise MMKvcmConfigError(f"{description} must be an integer")
    if not 0 <= value <= maximum:
        raise MMKvcmConfigError(f"{description} is outside its valid range")
    return value


def _required_mapping(value: Any, description: str) -> dict:
    if not isinstance(value, dict):
        raise MMKvcmConfigError(f"{description} must be a JSON object")
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


def _parse_shared_config(serialized: Any) -> dict:
    if not isinstance(serialized, str) or not serialized:
        raise MMKvcmConfigError(
            "explicit RECO_CLIENT_CONFIG must be a non-empty string"
        )
    if _utf8_size(serialized, "RECO_CLIENT_CONFIG") > _MAX_CLIENT_CONFIG_BYTES:
        raise MMKvcmConfigError("RECO_CLIENT_CONFIG exceeds the supported size")
    try:
        parsed = json.loads(
            serialized,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, RecursionError):
        # Do not include parser details: they can contain identity, endpoint,
        # model, or backend configuration fragments.
        raise MMKvcmConfigError(
            "RECO_CLIENT_CONFIG must contain valid strict JSON"
        ) from None
    return _required_mapping(parsed, "RECO_CLIENT_CONFIG")


def _parse_sdk_backend_configs(serialized: Any) -> list:
    if not isinstance(serialized, str) or not serialized:
        raise MMKvcmConfigError(
            "RECO_MODEL_SDK_CONFIG must contain a non-empty JSON array"
        )
    if _utf8_size(serialized, "RECO_MODEL_SDK_CONFIG") > _MAX_CLIENT_CONFIG_BYTES:
        raise MMKvcmConfigError("RECO_MODEL_SDK_CONFIG exceeds the supported size")
    try:
        parsed = json.loads(
            serialized,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, RecursionError):
        raise MMKvcmConfigError(
            "RECO_MODEL_SDK_CONFIG must contain valid strict JSON"
        ) from None
    if not isinstance(parsed, list):
        raise MMKvcmConfigError("RECO_MODEL_SDK_CONFIG must contain a JSON array")
    return parsed


def _select_primary_config(config_map: dict) -> dict:
    if not config_map:
        raise MMKvcmConfigError("RECO_CLIENT_CONFIG must not be an empty map")
    if "" in config_map:
        selected = config_map[""]
    elif len(config_map) == 1:
        selected = next(iter(config_map.values()))
    else:
        # The legacy C++ wrapper uses std::map::begin() for its data-plane
        # client.  Requiring an explicit empty-key primary config avoids
        # duplicating C++ string ordering in Python and selecting a different
        # client when multiple model configs are present.
        raise MMKvcmConfigError(
            "RECO_CLIENT_CONFIG with multiple entries requires an empty-key "
            "primary client config"
        )
    return _required_mapping(selected, "primary RECO_CLIENT_CONFIG entry")


def _validate_endpoint(value: Any) -> str:
    endpoint = _bounded_text(value, "KVCM endpoint", _MAX_ADDRESS_BYTES)
    if any(character.isspace() for character in endpoint) or "\x00" in endpoint:
        raise MMKvcmConfigError("KVCM endpoint contains invalid whitespace or NUL")
    return endpoint


def _validate_addresses(values: Any) -> Tuple[str, ...]:
    if not isinstance(values, list):
        raise MMKvcmConfigError("KVCM address must be a JSON array")
    if not 0 < len(values) <= _MAX_ADDRESSES:
        raise MMKvcmConfigError("KVCM address count is outside its valid range")
    addresses = tuple(_validate_endpoint(value) for value in values)
    if len(set(addresses)) != len(addresses):
        raise MMKvcmConfigError("KVCM addresses must be unique")
    return addresses


def _default_vipserver_resolver(domain: str) -> Iterable[Any]:
    # Importing RTP's VIPServer client starts its refresh machinery.  Keep it
    # completely outside grpc/rdma and direct-address KVCM startup paths.
    from rtp_llm.vipserver import get_host_list_by_domain_now

    return get_host_list_by_domain_now(domain)


def _resolve_vipserver_addresses(
    domain: str, resolver: Callable[[str], Iterable[Any]]
) -> Tuple[str, ...]:
    try:
        resolved = resolver(domain)
        if isinstance(resolved, (str, bytes)):
            raise TypeError()
        hosts = tuple(islice(iter(resolved), _MAX_ADDRESSES + 1))
    except Exception:
        raise MMKvcmConfigError("failed to resolve KVCM VIPServer domain") from None
    if not 0 < len(hosts) <= _MAX_ADDRESSES:
        raise MMKvcmConfigError(
            "KVCM VIPServer domain returned an invalid endpoint count"
        )

    addresses = []
    for host in hosts:
        try:
            ip = host.ip
            port = host.port
        except Exception:
            raise MMKvcmConfigError(
                "KVCM VIPServer returned a malformed endpoint"
            ) from None
        if not isinstance(ip, str) or not ip or type(port) is not int:
            raise MMKvcmConfigError("KVCM VIPServer returned a malformed endpoint")
        try:
            ipaddress.IPv4Address(ip)
        except ipaddress.AddressValueError:
            raise MMKvcmConfigError(
                "KVCM VIPServer returned a malformed endpoint"
            ) from None
        if not 0 < port <= 65535:
            # RTP VIPServer currently supplies IPv4 hosts.  Validate the
            # provider response before constructing a static gRPC target.
            raise MMKvcmConfigError("KVCM VIPServer returned a malformed endpoint")
        addresses.append(_validate_endpoint(f"{ip}:{port}"))
    if len(set(addresses)) != len(addresses):
        raise MMKvcmConfigError("KVCM VIPServer returned duplicate endpoints")
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
        raise MMKvcmConfigError(
            f"KVCM SDK queue_size must be at least {_MIN_SDK_QUEUE_SIZE} for KVMeta"
        )
    backend_configs = sdk_config.get("sdk_backend_configs")
    if not isinstance(backend_configs, list):
        raise MMKvcmConfigError("KVCM sdk_backend_configs must be a JSON array")
    for index, backend_config in enumerate(backend_configs):
        if not isinstance(backend_config, dict):
            raise MMKvcmConfigError(
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


def _split_reco_primary_config(py_env_configs: Any) -> dict:
    """Build a transfer-only ClientConfig from the established RECO fields.

    The regular RemoteConnector computes its final instance id later, after
    model/cache topology initialization.  That value is intentionally not
    reproduced here: doing so would couple Python startup to the main C++ data
    path.  ``RECO_INSTANCE_ID_SALT`` is already the stable identity input when
    operators provide one; otherwise the shared instance group is the only
    identity that is both configured and guaranteed to match across E/P/D
    processes.
    """

    kv_cache = py_env_configs.kv_cache_config
    instance_group = _bounded_text(
        kv_cache.reco_instance_group,
        "RECO_INSTANCE_GROUP",
        _MAX_INSTANCE_GROUP_BYTES,
    )
    instance_id_salt = _bounded_text(
        kv_cache.reco_instance_id_salt,
        "RECO_INSTANCE_ID_SALT",
        _MAX_INSTANCE_ID_BYTES,
        allow_empty=True,
    )
    instance_id = instance_id_salt or instance_group

    enable_vipserver = kv_cache.reco_enable_vipserver
    if type(enable_vipserver) is not bool:
        raise MMKvcmConfigError("RECO_ENABLE_VIPSERVER must be a boolean")

    vipserver_domain = _bounded_text(
        kv_cache.reco_vipserver_domain,
        "RECO_VIPSERVER_DOMAIN",
        _MAX_ADDRESS_BYTES,
        allow_empty=True,
    )
    server_address = _bounded_text(
        kv_cache.reco_server_address,
        "RECO_SERVER_ADDRESS",
        _MAX_ADDRESS_BYTES,
        allow_empty=True,
    )
    addresses = [] if not server_address else [_validate_endpoint(server_address)]

    # RemoteConnector/GrpcStub deliberately accept zero for these two fields:
    # retry_time is clamped to one attempt and connection_timeout still gets
    # the stub's minimum per-attempt wait. Preserve that established contract.
    retry_time = _non_negative_int(
        kv_cache.reco_meta_channel_retry_time,
        "RECO_META_CHANNEL_RETRY_TIME",
        _MAX_UINT32,
    )
    connection_timeout = _non_negative_int(
        kv_cache.reco_meta_channel_connection_timeout,
        "RECO_META_CHANNEL_CONNECTION_TIMEOUT",
        _MAX_UINT32,
    )
    call_timeout = _positive_int(
        kv_cache.reco_meta_channel_call_timeout,
        "RECO_META_CHANNEL_CALL_TIMEOUT",
        _MAX_CALL_TIMEOUT_MS,
    )
    thread_num = _positive_int(
        kv_cache.reco_storage_thread_num,
        "RECO_STORAGE_THREAD_NUM",
        _MAX_INT32,
    )
    queue_size = _positive_int(
        kv_cache.reco_storage_queue_size,
        "RECO_STORAGE_QUEUE_SIZE",
        _MAX_INT32,
    )
    put_timeout_ms = _positive_int(
        kv_cache.reco_put_timeout_ms,
        "RECO_PUT_TIMEOUT_MS",
        _MAX_INT32,
    )
    get_timeout_ms = _positive_int(
        kv_cache.reco_get_timeout_ms,
        "RECO_GET_TIMEOUT_MS",
        _MAX_INT32,
    )
    user_data = _bounded_text(
        kv_cache.reco_model_user_data,
        "RECO_MODEL_USER_DATA",
        _MAX_USER_DATA_BYTES,
        allow_empty=True,
    )
    return {
        "enable_vipserver": enable_vipserver,
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
                kv_cache.reco_model_sdk_config
            ),
            "timeout_config": {
                "put_timeout_ms": put_timeout_ms,
                "get_timeout_ms": get_timeout_ms,
            },
        },
        # These fields are required by the existing transfer ClientConfig JSON
        # parser, but KVMeta registration owns the canonical object deployment
        # metadata.  Use the same constants as KvMetaManager rather than
        # pretending the one-byte schema describes the main model topology.
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


def derive_mm_kvcm_client_config(
    shared_client_config: str,
    *,
    vipserver_resolver: Optional[Callable[[str], Iterable[Any]]] = None,
) -> DerivedMMKvcmClientConfig:
    """Build an exact-object client config without mutating the shared JSON."""

    config_map = _parse_shared_config(shared_client_config)
    primary = _select_primary_config(config_map)

    return _derive_primary_config(primary, vipserver_resolver=vipserver_resolver)


def _derive_primary_config(
    primary: dict,
    *,
    vipserver_resolver: Optional[Callable[[str], Iterable[Any]]] = None,
) -> DerivedMMKvcmClientConfig:
    """Validate and isolate one existing KVCM client configuration."""

    instance_group = _derive_identity(
        primary, "instance_group", _MAX_INSTANCE_GROUP_BYTES
    )
    instance_id = _derive_identity(primary, "instance_id", _MAX_INSTANCE_ID_BYTES)

    enable_vipserver = primary.get("enable_vipserver", False)
    if type(enable_vipserver) is not bool:
        raise MMKvcmConfigError("KVCM enable_vipserver must be a boolean")
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
        raise MMKvcmConfigError(
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
        raise MMKvcmConfigError(
            "KVCM client configuration cannot be serialized safely"
        ) from None
    if (
        _utf8_size(serialized_transfer_config, "KVMeta transfer client config")
        > _MAX_CLIENT_CONFIG_BYTES
    ):
        raise MMKvcmConfigError(
            "KVMeta transfer client config exceeds the supported size"
        )

    return DerivedMMKvcmClientConfig(
        addresses=addresses,
        instance_id=instance_id,
        instance_group=instance_group,
        user_data=user_data,
        transfer_client_config=serialized_transfer_config,
        call_timeout_ms=call_timeout_ms,
        write_timeout_seconds=write_timeout_seconds,
    )


def configure_mm_kvcm_client(
    py_env_configs: Any,
    *,
    environ: Optional[Mapping[str, str]] = None,
    vipserver_resolver: Optional[Callable[[str], Iterable[Any]]] = None,
) -> None:
    """Populate the internal MM KVMeta config only when KVCM is selected."""

    transport = py_env_configs.vit_config.output_transport
    if transport.mode != _MM_TRANSPORT_MODE_KVCM:
        return

    environment = os.environ if environ is None else environ
    removed = [name for name in _REMOVED_CLIENT_ENV_VARS if name in environment]
    if removed:
        raise MMKvcmConfigError(
            "MM KVCM client settings now come from the existing RECO KVCM "
            "configuration; remove " + ", ".join(removed)
        )

    shared_client_config = py_env_configs.kv_cache_config.reco_client_config
    if not isinstance(shared_client_config, str):
        raise MMKvcmConfigError("RECO_CLIENT_CONFIG must be a string")
    if shared_client_config:
        derived = derive_mm_kvcm_client_config(
            shared_client_config,
            vipserver_resolver=vipserver_resolver,
        )
    else:
        derived = _derive_primary_config(
            _split_reco_primary_config(py_env_configs),
            vipserver_resolver=vipserver_resolver,
        )

    # Apply only after the complete KVCM configuration has passed validation.
    # MMKvcmConfig is a private startup object with plain attributes, so these
    # assignments cannot invoke user code or perform I/O.
    kvcm = transport.kvcm
    kvcm.addresses = list(derived.addresses)
    kvcm.instance_id = derived.instance_id
    kvcm.instance_group = derived.instance_group
    kvcm.user_data = derived.user_data
    kvcm.transfer_client_config = derived.transfer_client_config
    kvcm.call_timeout_ms = derived.call_timeout_ms
    kvcm.write_timeout_seconds = derived.write_timeout_seconds

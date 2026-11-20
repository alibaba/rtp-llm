from typing import NamedTuple


class KVCacheEventEnvCase(NamedTuple):
    env_name: str
    field_name: str
    raw_value: str
    expected_value: object


KV_CACHE_EVENT_FIELD_VALUES = {
    "kv_cache_event_publisher_type": "kvcm",
    "kv_cache_event_manager_endpoint": "http://kvcm-meta:56020",
    "kv_cache_event_instance_group": "test-group",
    "kv_cache_event_instance_id": "test-instance",
    "kv_cache_event_host_ip_port": "127.0.0.1:18000",
    "kv_cache_event_snapshot_interval_ms": 45000,
}

KV_CACHE_EVENT_ENV_CASES = tuple(
    KVCacheEventEnvCase(field_name.upper(), field_name, str(value), value)
    for field_name, value in KV_CACHE_EVENT_FIELD_VALUES.items()
    if field_name in (
        "kv_cache_event_publisher_type",
        "kv_cache_event_instance_id",
        "kv_cache_event_snapshot_interval_ms",
    )
)

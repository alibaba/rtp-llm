from typing import NamedTuple


class KVCacheEventEnvCase(NamedTuple):
    env_name: str
    field_name: str
    raw_value: str
    expected_value: object


KV_CACHE_EVENT_ENV_CASES = (
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_PUBLISHER_TYPE",
        "kv_cache_event_publisher_type",
        "kvcm",
        "kvcm",
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_MANAGER_ENDPOINT",
        "kv_cache_event_manager_endpoint",
        "http://kvcm-meta:56020",
        "http://kvcm-meta:56020",
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_INSTANCE_GROUP",
        "kv_cache_event_instance_group",
        "test-group",
        "test-group",
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_INSTANCE_ID",
        "kv_cache_event_instance_id",
        "test-instance",
        "test-instance",
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_HOST_IP_PORT",
        "kv_cache_event_host_ip_port",
        "127.0.0.1:18000",
        "127.0.0.1:18000",
    ),
)

KV_CACHE_EVENT_FIELD_VALUES = {
    case.field_name: case.expected_value for case in KV_CACHE_EVENT_ENV_CASES
}

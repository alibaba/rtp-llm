import re
from pathlib import Path
from typing import NamedTuple, Union

from rtp_llm.config.test.kv_cache_config_state_layout import (
    CURRENT_STATE_SIZE,
    DISK_STATE_SIZE,
    EVENT_STATE_OFFSET,
    KV_CACHE_EVENT_FIELDS,
    LEGACY_BASE_STATE_SIZE,
)


class KVCacheEventEnvCase(NamedTuple):
    env_name: str
    field_name: str
    raw_value: str
    expected_value: Union[str, int]


class KVCacheEventValidationCase(NamedTuple):
    target: str
    expected_valid: bool
    value: str


def _load_validation_cases() -> tuple[KVCacheEventValidationCase, ...]:
    pattern = re.compile(
        r'^KV_CACHE_EVENT_VALIDATION_CASE\((ENDPOINT|HOST|IDENTITY), (true|false), R"KV\((.*)\)KV"\)$'
    )
    cases = []
    source = Path(__file__).with_name("kv_cache_event_validation_cases.inc")
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line or line.startswith("//"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid validation case at {source}:{line_number}")
        cases.append(
            KVCacheEventValidationCase(
                target=match.group(1).lower(),
                expected_valid=match.group(2) == "true",
                value=match.group(3),
            )
        )
    if not cases:
        raise ValueError(f"no KV cache event validation cases found in {source}")
    return tuple(cases)


KV_CACHE_EVENT_VALIDATION_CASES = _load_validation_cases()


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
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_QUEUE_CAPACITY",
        "kv_cache_event_queue_capacity",
        "12345",
        12345,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_REPORT_BATCH_SIZE",
        "kv_cache_event_report_batch_size",
        "321",
        321,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_FLUSH_INTERVAL_MS",
        "kv_cache_event_flush_interval_ms",
        "17",
        17,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_HEARTBEAT_INTERVAL_MS",
        "kv_cache_event_heartbeat_interval_ms",
        "1017",
        1017,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_REQUEST_TIMEOUT_MS",
        "kv_cache_event_request_timeout_ms",
        "1517",
        1517,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_SNAPSHOT_TIMEOUT_MS",
        "kv_cache_event_snapshot_timeout_ms",
        "3017",
        3017,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_RETRY_INTERVAL_MS",
        "kv_cache_event_retry_interval_ms",
        "517",
        517,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_SNAPSHOT_INTERVAL_MS",
        "kv_cache_event_snapshot_interval_ms",
        "300017",
        300017,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_LOG_MAX_KEYS",
        "kv_cache_event_log_max_keys",
        "13",
        13,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_SNAPSHOT_MAX_KEYS",
        "kv_cache_event_snapshot_max_keys",
        "999983",
        999983,
    ),
    KVCacheEventEnvCase(
        "KV_CACHE_EVENT_SNAPSHOT_MAX_BYTES",
        "kv_cache_event_snapshot_max_bytes",
        "268435439",
        268435439,
    ),
)

KV_CACHE_EVENT_FIELD_VALUES = {
    case.field_name: case.expected_value for case in KV_CACHE_EVENT_ENV_CASES
}

KV_CACHE_EVENT_DEFAULTS = {
    "kv_cache_event_publisher_type": "none",
    "kv_cache_event_manager_endpoint": "",
    "kv_cache_event_instance_group": "",
    "kv_cache_event_instance_id": "",
    "kv_cache_event_host_ip_port": "",
    "kv_cache_event_queue_capacity": 100000,
    "kv_cache_event_report_batch_size": 1000,
    "kv_cache_event_flush_interval_ms": 20,
    "kv_cache_event_heartbeat_interval_ms": 1000,
    "kv_cache_event_request_timeout_ms": 1500,
    "kv_cache_event_snapshot_timeout_ms": 30000,
    "kv_cache_event_retry_interval_ms": 500,
    "kv_cache_event_snapshot_interval_ms": 300000,
    "kv_cache_event_log_max_keys": 8,
    "kv_cache_event_snapshot_max_keys": 1000000,
    "kv_cache_event_snapshot_max_bytes": 256 * 1024 * 1024,
}

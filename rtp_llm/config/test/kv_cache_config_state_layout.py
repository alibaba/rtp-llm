"""Single test-side declaration of KVCacheConfig's published pickle layout."""

LEGACY_BASE_STATE_SIZE = 43

DISK_CACHE_FIELDS = (
    "enable_memory_cache_disk",
    "memory_cache_disk_paths",
    "memory_cache_disk_size_mb",
    "memory_cache_disk_buffered_io",
    "memory_cache_disk_sync_timeout_ms",
    "enable_gpu_prefix_tree",
    "enable_prefix_tree_memory_cache",
    "enable_legacy_memory_connector_fallback",
    "prefix_tree_memory_state_swa_pool_ratio",
    "enable_independent_group_eviction",
    "load_cache_retry_times",
)

# Distinct, non-default probes make order mistakes observable after a round
# trip. Keep them beside the field order they validate.
DISK_FIELD_VALUES = {
    "enable_memory_cache_disk": True,
    "memory_cache_disk_paths": "/tmp/cache",
    "memory_cache_disk_size_mb": 4096,
    "memory_cache_disk_buffered_io": False,
    "memory_cache_disk_sync_timeout_ms": 1234,
    "enable_gpu_prefix_tree": True,
    "enable_prefix_tree_memory_cache": True,
    "enable_legacy_memory_connector_fallback": False,
    "prefix_tree_memory_state_swa_pool_ratio": 37,
    "enable_independent_group_eviction": True,
    "load_cache_retry_times": 7,
}

KV_CACHE_EVENT_FIELDS = (
    "kv_cache_event_publisher_type",
    "kv_cache_event_manager_endpoint",
    "kv_cache_event_instance_group",
    "kv_cache_event_instance_id",
    "kv_cache_event_host_ip_port",
    "kv_cache_event_queue_capacity",
    "kv_cache_event_report_batch_size",
    "kv_cache_event_flush_interval_ms",
    "kv_cache_event_heartbeat_interval_ms",
    "kv_cache_event_request_timeout_ms",
    "kv_cache_event_snapshot_timeout_ms",
    "kv_cache_event_retry_interval_ms",
    "kv_cache_event_snapshot_interval_ms",
    "kv_cache_event_log_max_keys",
    "kv_cache_event_snapshot_max_keys",
    "kv_cache_event_snapshot_max_bytes",
)

DISK_STATE_SIZE = LEGACY_BASE_STATE_SIZE + len(DISK_CACHE_FIELDS)
EVENT_STATE_OFFSET = DISK_STATE_SIZE
CURRENT_STATE_SIZE = EVENT_STATE_OFFSET + len(KV_CACHE_EVENT_FIELDS)

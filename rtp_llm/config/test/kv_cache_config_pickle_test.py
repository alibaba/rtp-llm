import pickle
from unittest import TestCase, main

from rtp_llm.ops import KVCacheConfig

EVENT_FIELDS = {
    "kv_cache_event_publisher_type": "kvcm",
    "kv_cache_event_manager_endpoint": "http://kvcm-meta:56020",
    "kv_cache_event_instance_group": "test-group",
    "kv_cache_event_instance_id": "test-instance",
    "kv_cache_event_host_ip_port": "127.0.0.1:18000",
    "kv_cache_event_queue_capacity": 12345,
    "kv_cache_event_report_batch_size": 321,
    "kv_cache_event_flush_interval_ms": 17,
    "kv_cache_event_heartbeat_interval_ms": 1017,
    "kv_cache_event_request_timeout_ms": 1517,
    "kv_cache_event_snapshot_timeout_ms": 3017,
    "kv_cache_event_retry_interval_ms": 517,
    "kv_cache_event_snapshot_interval_ms": 300017,
    "kv_cache_event_log_max_keys": 13,
}


class KVCacheConfigPickleTest(TestCase):
    def test_event_fields_round_trip_in_current_state(self):
        config = KVCacheConfig()
        for name, value in EVENT_FIELDS.items():
            setattr(config, name, value)

        self.assertEqual(68, len(config.__getstate__()))
        restored = pickle.loads(pickle.dumps(config))

        for name, value in EVENT_FIELDS.items():
            self.assertEqual(value, getattr(restored, name), name)

    def test_legacy_54_element_state_uses_event_defaults(self):
        source = KVCacheConfig()
        source.enable_memory_cache_disk = True
        source.memory_cache_disk_paths = "/tmp/cache"
        source.load_cache_retry_times = 7
        legacy_state = source.__getstate__()[:54]
        self.assertEqual(54, len(legacy_state))

        restored = KVCacheConfig()
        restored.__setstate__(legacy_state)

        self.assertTrue(restored.enable_memory_cache_disk)
        self.assertEqual("/tmp/cache", restored.memory_cache_disk_paths)
        self.assertEqual(7, restored.load_cache_retry_times)
        defaults = KVCacheConfig()
        for name in EVENT_FIELDS:
            self.assertEqual(getattr(defaults, name), getattr(restored, name), name)

    def test_unreleased_56_and_57_element_states_are_rejected(self):
        current_state = KVCacheConfig().__getstate__()
        for size in (56, 57):
            with self.subTest(size=size), self.assertRaises(RuntimeError):
                KVCacheConfig().__setstate__(current_state[:size])


if __name__ == "__main__":
    main()

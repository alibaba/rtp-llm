import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import (
    KV_CACHE_EVENT_FIELD_VALUES,
)
from rtp_llm.ops import KVCacheConfig

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


class KVCacheConfigPickleTest(TestCase):
    def test_event_fields_round_trip_in_current_state(self):
        config = KVCacheConfig()
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            setattr(config, name, value)

        self.assertEqual(68, len(config.__getstate__()))
        restored = pickle.loads(pickle.dumps(config))

        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            self.assertEqual(value, getattr(restored, name), name)

    def test_legacy_54_element_state_uses_event_defaults(self):
        source = KVCacheConfig()
        source.enable_memory_cache_disk = True
        source.memory_cache_disk_paths = "/tmp/cache"
        source.load_cache_retry_times = 7
        legacy_state = source.__getstate__()[:54]
        self.assertEqual(54, len(legacy_state))

        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(legacy_state)

        self.assertTrue(restored.enable_memory_cache_disk)
        self.assertEqual("/tmp/cache", restored.memory_cache_disk_paths)
        self.assertEqual(7, restored.load_cache_retry_times)
        defaults = KVCacheConfig()
        for name in KV_CACHE_EVENT_FIELD_VALUES:
            self.assertEqual(getattr(defaults, name), getattr(restored, name), name)

    def test_legacy_43_element_state_uses_disk_and_event_defaults(self):
        source = KVCacheConfig()
        source.enable_memory_cache_disk = True
        source.memory_cache_disk_paths = "/tmp/cache"
        source.load_cache_retry_times = 7
        source.kv_cache_event_publisher_type = "kvcm"
        legacy_state = source.__getstate__()[:43]
        self.assertEqual(43, len(legacy_state))

        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(legacy_state)

        defaults = KVCacheConfig()
        for name in (*DISK_CACHE_FIELDS, *KV_CACHE_EVENT_FIELD_VALUES):
            self.assertEqual(getattr(defaults, name), getattr(restored, name), name)

    def test_unreleased_56_and_57_element_states_are_rejected(self):
        current_state = KVCacheConfig().__getstate__()
        for size in (56, 57):
            with self.subTest(size=size), self.assertRaisesRegex(
                RuntimeError, "Invalid state"
            ):
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(current_state[:size])


if __name__ == "__main__":
    main()

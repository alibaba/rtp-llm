import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import KV_CACHE_EVENT_FIELD_VALUES
from rtp_llm.ops import CacheOffloadPolicy, KVCacheConfig, KVCacheSpecDesc

EVENT_PICKLE_FIELDS = tuple(KV_CACHE_EVENT_FIELD_VALUES)


class KVCacheConfigPickleTest(TestCase):
    def test_event_fields_round_trip_with_block_tree_configuration(self):
        config = KVCacheConfig()
        config.block_tree_transfer_worker_count = 7
        config.enable_disk_cache = True
        config.disk_cache_paths = "/tmp/cache"
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            setattr(config, name, value)
        state = config.__getstate__()
        self.assertEqual(state[:2], ("KVCacheConfig", 8))
        self.assertEqual(state[64:69], tuple(KV_CACHE_EVENT_FIELD_VALUES.values()))
        # setstate must normalize its private slice, not mutate the supplied
        # tuple retained by another Python reference.
        original_state = tuple(list(state))
        direct = KVCacheConfig.__new__(KVCacheConfig)
        direct.__setstate__(state)
        self.assertEqual(state, original_state)
        self.assertEqual(direct.__getstate__(), original_state)
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.block_tree_transfer_worker_count, 7)
        self.assertTrue(restored.enable_disk_cache)
        self.assertEqual(restored.disk_cache_paths, "/tmp/cache")
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            self.assertEqual(getattr(restored, name), value, name)

    def test_previous_block_tree_state_uses_event_defaults(self):
        source = KVCacheConfig()
        source.dsv4_fixed_pool_blocks = 17
        source.block_tree_transfer_worker_count = 7
        state = source.__getstate__()
        previous = (state[0], 6, *state[2:64])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.dsv4_fixed_pool_blocks, 17)
        self.assertEqual(restored.block_tree_transfer_worker_count, 7)
        defaults = KVCacheConfig()
        for name in EVENT_PICKLE_FIELDS:
            self.assertEqual(getattr(restored, name), getattr(defaults, name), name)

    def test_malformed_event_states_are_rejected(self):
        state = KVCacheConfig().__getstate__()
        for malformed in (
            state[:-1],
            state + ("extra",),
            (state[0], 99, *state[2:]),
            ("OtherConfig", *state[1:]),
        ):
            with self.subTest(state=malformed), self.assertRaisesRegex(
                RuntimeError, "invalid KVCacheConfig state"
            ):
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(malformed)

    def test_offload_round_trip_and_previous_event_state(self):
        source = KVCacheConfig()
        values = {
            "enable_sparse_kv_offload": True,
            "sparse_kv_offload_host_mb": 1024,
            "sparse_kv_offload_test_hbm_blocks": 2,
        }
        for name, value in values.items():
            setattr(source, name, value)
        restored = pickle.loads(pickle.dumps(source))
        for name, value in values.items():
            self.assertEqual(getattr(restored, name), value)
        state = source.__getstate__()
        previous = (state[0], 7, *state[2:69])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        defaults = KVCacheConfig()
        for name in values:
            self.assertEqual(getattr(restored, name), getattr(defaults, name))
        self.assertFalse(restored.enable_sparse_kv_offload)

    def test_descriptor_offload_round_trip_and_legacy_state(self):
        desc = KVCacheSpecDesc()
        desc.tag = "default"
        policy = CacheOffloadPolicy()
        policy.enabled = True
        policy.max_selected_rows = 256
        desc.offload = policy
        desc.offload_runtime_group = 7
        restored = pickle.loads(pickle.dumps(desc))
        self.assertEqual(restored.tag, "default")
        self.assertTrue(restored.offload.enabled)
        self.assertEqual(restored.offload.max_selected_rows, 256)
        self.assertEqual(restored.offload_runtime_group, 7)
        previous = KVCacheSpecDesc.__new__(KVCacheSpecDesc)
        previous.__setstate__(desc.__getstate__()[:21])
        self.assertTrue(previous.offload.enabled)
        self.assertIsNone(previous.offload_runtime_group)
        legacy = KVCacheSpecDesc.__new__(KVCacheSpecDesc)
        legacy.__setstate__(desc.__getstate__()[:20])
        self.assertEqual(legacy.tag, "default")
        self.assertIsNone(legacy.offload)


if __name__ == "__main__":
    main()

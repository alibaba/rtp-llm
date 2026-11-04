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
        for enable_crc in (False, True):
            with self.subTest(enable_crc=enable_crc):
                config.enable_crc = enable_crc
                state = config.__getstate__()
                self.assertEqual(len(state), 76)
                self.assertEqual(state[:2], ("KVCacheConfig", 11))
                self.assertEqual(
                    state[64:69], tuple(KV_CACHE_EVENT_FIELD_VALUES.values())
                )
                self.assertIs(state[69], enable_crc)
                self.assertEqual(state[70:72], (0, 0))
                self.assertEqual(state[72:75], (False, 0, 0))
                self.assertEqual(state[75], 2048)
                # Normalizing old fields must not mutate the supplied tuple.
                original_state = tuple(list(state))
                direct = KVCacheConfig.__new__(KVCacheConfig)
                direct.__setstate__(state)
                self.assertEqual(state, original_state)
                self.assertEqual(direct.__getstate__(), original_state)
                restored = pickle.loads(pickle.dumps(config))
                self.assertIs(restored.enable_crc, enable_crc)
                self.assertEqual(restored.block_tree_transfer_worker_count, 7)
                self.assertTrue(restored.enable_disk_cache)
                self.assertEqual(restored.disk_cache_paths, "/tmp/cache")
                for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
                    self.assertEqual(getattr(restored, name), value, name)
                self.assertIn(f"enable_crc: {int(enable_crc)}\n", restored.to_string())

    def test_version_seven_preserves_events_and_enables_crc_by_default(self):
        source = KVCacheConfig()
        source.enable_crc = False
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            setattr(source, name, value)
        state = source.__getstate__()
        previous = (state[0], 7, *state[2:69])
        original_state = tuple(list(previous))
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(previous, original_state)
        self.assertTrue(restored.enable_crc)
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            self.assertEqual(getattr(restored, name), value, name)

    def test_previous_block_tree_state_uses_event_defaults(self):
        source = KVCacheConfig()
        source.enable_crc = False
        source.dsv4_fixed_pool_blocks = 17
        source.block_tree_transfer_worker_count = 7
        state = source.__getstate__()
        previous = (state[0], 6, *state[2:64])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.dsv4_fixed_pool_blocks, 17)
        self.assertEqual(restored.block_tree_transfer_worker_count, 7)
        self.assertTrue(restored.enable_crc)
        defaults = KVCacheConfig()
        for name in EVENT_PICKLE_FIELDS:
            self.assertEqual(getattr(restored, name), getattr(defaults, name), name)

    def test_tail_and_pool_configuration_survives_worker_round_trips(self):
        config = KVCacheConfig()
        config.reuse_cache_tail_blocks = 13
        config.linear_pool_blocks = 31
        config.swa_pool_blocks = 17
        for _ in range(2):
            config = pickle.loads(pickle.dumps(config))
            self.assertEqual(config.reuse_cache_tail_blocks, 13)
            self.assertEqual(config.linear_pool_blocks, 31)
            self.assertEqual(config.swa_pool_blocks, 17)
        self.assertFalse(hasattr(config, "linear_step"))
        self.assertFalse(hasattr(config, "linear_cache_tail_blocks"))

    def test_version_nine_uses_offload_defaults(self):
        source = KVCacheConfig()
        source.enable_crc = False
        source.reuse_cache_tail_blocks = 13
        source.linear_pool_blocks = 31
        source.swa_pool_blocks = 17
        source.enable_sparse_kv_offload = True
        source.sparse_kv_offload_host_mb = 1024
        source.sparse_kv_offload_test_hbm_blocks = 2
        state = source.__getstate__()
        previous = (state[0], 9, *state[2:72])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.reuse_cache_tail_blocks, 13)
        self.assertEqual(restored.linear_pool_blocks, 31)
        self.assertEqual(restored.swa_pool_blocks, 17)
        self.assertFalse(restored.enable_crc)
        self.assertFalse(restored.enable_sparse_kv_offload)

    def test_version_eight_uses_new_retention_defaults(self):
        source = KVCacheConfig()
        source.enable_crc = False
        source.reuse_cache_tail_blocks = 13
        state = list(source.__getstate__()[:70])
        state[1] = 8
        state[10] = 4  # Historical checkpoint interval, not a tail count.
        previous = tuple(state)
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(previous, tuple(state))
        self.assertEqual(restored.reuse_cache_tail_blocks, 2147483647)
        self.assertEqual(restored.linear_pool_blocks, 0)
        self.assertEqual(restored.swa_pool_blocks, 0)
        self.assertFalse(restored.enable_crc)

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

    def test_hot_rows_round_trip_and_version_ten_default(self):
        source = KVCacheConfig()
        source.sparse_kv_offload_hot_rows = 4096
        restored = pickle.loads(pickle.dumps(source))
        self.assertEqual(restored.sparse_kv_offload_hot_rows, 4096)
        state = source.__getstate__()
        previous = (state[0], 10, *state[2:75])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.sparse_kv_offload_hot_rows, 2048)

    def test_version_eight_preserves_legacy_offload_fields(self):
        source = KVCacheConfig()
        source.enable_crc = False
        source.enable_sparse_kv_offload = True
        source.sparse_kv_offload_host_mb = 1024
        source.sparse_kv_offload_test_hbm_blocks = 2
        state = source.__getstate__()
        previous = (state[0], 8, *state[2:69], *state[72:75])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertTrue(restored.enable_sparse_kv_offload)
        self.assertEqual(restored.sparse_kv_offload_host_mb, 1024)
        self.assertEqual(restored.sparse_kv_offload_test_hbm_blocks, 2)
        self.assertTrue(restored.enable_crc)

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

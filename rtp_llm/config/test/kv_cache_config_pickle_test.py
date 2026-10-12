import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import KV_CACHE_EVENT_FIELD_VALUES
from rtp_llm.ops import DataType, KVCacheConfig, LinearAttentionConfig

EVENT_PICKLE_FIELDS = tuple(KV_CACHE_EVENT_FIELD_VALUES)


class LinearAttentionConfigBindingsTest(TestCase):
    def test_reclaim_capability_defaults_off_with_previous_constructor(self):
        config = LinearAttentionConfig(
            4, 128, 16, 48, 128, DataType.TYPE_FP32, DataType.TYPE_BF16
        )
        self.assertFalse(config.supports_history_state_reclaim)
        self.assertEqual(config.ssm_state_dtype, DataType.TYPE_FP32)

    def test_model_can_declare_reclaim_capability(self):
        config = LinearAttentionConfig(supports_history_state_reclaim=True)
        self.assertTrue(config.supports_history_state_reclaim)
        config.supports_history_state_reclaim = False
        self.assertFalse(config.supports_history_state_reclaim)


class KVCacheConfigPickleTest(TestCase):
    def test_linear_cache_reclaim_defaults_off_and_round_trips(self):
        config = KVCacheConfig()
        self.assertFalse(config.enable_linear_cache_reclaim)
        config.enable_linear_cache_reclaim = True
        restored = pickle.loads(pickle.dumps(config))
        self.assertTrue(restored.enable_linear_cache_reclaim)

    def test_previous_event_state_defaults_linear_reclaim_off(self):
        config = KVCacheConfig()
        config.enable_linear_cache_reclaim = True
        state = config.__getstate__()
        previous = (state[0], 7, *state[2:-1])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertFalse(restored.enable_linear_cache_reclaim)

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
                config.enable_linear_cache_reclaim = True
                state = config.__getstate__()
                self.assertEqual(len(state), 73)
                self.assertEqual(state[:2], ("KVCacheConfig", 10))
                self.assertEqual(state[64:69], tuple(KV_CACHE_EVENT_FIELD_VALUES.values()))
                self.assertIs(state[69], enable_crc)
                self.assertTrue(state[70])
                self.assertEqual(state[71], 0)
                self.assertEqual(state[72], 0)
                # setstate must normalize its private slice, not mutate the supplied
                # version-10 tuple retained by another Python reference.
                original_state = tuple(list(state))
                direct = KVCacheConfig.__new__(KVCacheConfig)
                direct.__setstate__(state)
                self.assertEqual(state, original_state)
                self.assertEqual(direct.__getstate__(), original_state)
                restored = pickle.loads(pickle.dumps(config))
                self.assertIs(restored.enable_crc, enable_crc)
                self.assertTrue(restored.enable_linear_cache_reclaim)
                self.assertEqual(restored.block_tree_transfer_worker_count, 7)
                self.assertTrue(restored.enable_disk_cache)
                self.assertEqual(restored.disk_cache_paths, "/tmp/cache")
                for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
                    self.assertEqual(getattr(restored, name), value, name)
                self.assertIn(f"enable_crc: {int(enable_crc)}\n", restored.to_string())
                # A mainline version-8 state (crc tail, no reclaim) must still
                # decode with crc preserved and reclaim defaulted off.
                crc_only = (state[0], 8, *state[2:70])
                restored_crc_only = KVCacheConfig.__new__(KVCacheConfig)
                restored_crc_only.__setstate__(crc_only)
                self.assertIs(restored_crc_only.enable_crc, enable_crc)
                self.assertFalse(restored_crc_only.enable_linear_cache_reclaim)

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


if __name__ == "__main__":
    main()

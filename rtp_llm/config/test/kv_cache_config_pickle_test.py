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
    def test_pool_block_weights_binding_defaults_empty_and_preserves_raw_json(self):
        config = KVCacheConfig()
        self.assertEqual(config.kv_cache_pool_block_weights, "")
        raw = '{"full": 8, "linear": 1, "draft-linear": 2}'
        config.kv_cache_pool_block_weights = raw
        self.assertEqual(config.kv_cache_pool_block_weights, raw)
        self.assertIn(f"kv_cache_pool_block_weights: {raw}\n", config.to_string())

    def test_pool_block_weights_round_trip_preserves_other_cache_policies(self):
        config = KVCacheConfig()
        config.kv_cache_pool_block_weights = '{"full":7,"linear":2}'
        config.enable_linear_cache_reclaim = True
        config.reserve_block_ratio = 9
        config.kv_cache_event_instance_group = "weighted-pools"
        state = config.__getstate__()
        self.assertEqual(state[:2], ("KVCacheConfig", 9))
        self.assertEqual(len(state), 71)
        self.assertEqual(state[70], config.kv_cache_pool_block_weights)

        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.__getstate__(), state)
        self.assertEqual(
            restored.kv_cache_pool_block_weights, config.kv_cache_pool_block_weights
        )
        self.assertTrue(restored.enable_linear_cache_reclaim)
        self.assertEqual(restored.reserve_block_ratio, 9)
        self.assertEqual(restored.kv_cache_event_instance_group, "weighted-pools")

    def test_version_eight_state_keeps_reclaim_and_defaults_pool_weights_empty(self):
        config = KVCacheConfig()
        config.enable_linear_cache_reclaim = True
        config.kv_cache_pool_block_weights = '{"full":8,"linear":1}'
        state = config.__getstate__()
        previous = (state[0], 8, *state[2:70])
        original = tuple(list(previous))
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(previous, original)
        self.assertTrue(restored.enable_linear_cache_reclaim)
        self.assertEqual(restored.kv_cache_pool_block_weights, "")

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
        previous = (state[0], 7, *state[2:69])
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
        state = config.__getstate__()
        self.assertEqual(state[:2], ("KVCacheConfig", 9))
        self.assertEqual(state[64:69], tuple(KV_CACHE_EVENT_FIELD_VALUES.values()))
        # setstate must normalize its private slice, not mutate the supplied
        # version-9 tuple retained by another Python reference.
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
        self.assertEqual(restored.kv_cache_pool_block_weights, "")
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


if __name__ == "__main__":
    main()

import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import KV_CACHE_EVENT_FIELD_VALUES
from rtp_llm.ops import KVCacheConfig

BASE_EVENT_FIELD_VALUES = {
    name: value for name, value in KV_CACHE_EVENT_FIELD_VALUES.items()
    if name != "kv_cache_event_snapshot_interval_ms"
}
EVENT_PICKLE_FIELDS = tuple(BASE_EVENT_FIELD_VALUES)


class KVCacheConfigPickleTest(TestCase):
    def test_event_fields_round_trip_with_block_tree_configuration(self):
        config = KVCacheConfig()
        config.block_tree_transfer_worker_count = 7
        config.enable_disk_cache = True
        config.disk_cache_paths = "/tmp/cache"
        for name, value in BASE_EVENT_FIELD_VALUES.items():
            setattr(config, name, value)
        for enable_crc in (False, True):
            with self.subTest(enable_crc=enable_crc):
                config.enable_crc = enable_crc
                state = config.__getstate__()
                self.assertEqual(len(state), 85)
                self.assertEqual(state[:2], ("KVCacheConfig", 13))
                self.assertEqual(
                    state[64:69], tuple(BASE_EVENT_FIELD_VALUES.values())
                )
                self.assertIs(state[69], enable_crc)
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
                for name, value in BASE_EVENT_FIELD_VALUES.items():
                    self.assertEqual(getattr(restored, name), value, name)
                self.assertIn(f"enable_crc: {int(enable_crc)}\n", restored.to_string())

    def test_version_seven_preserves_events_and_enables_crc_by_default(self):
        source = KVCacheConfig()
        source.enable_crc = False
        for name, value in BASE_EVENT_FIELD_VALUES.items():
            setattr(source, name, value)
        state = source.__getstate__()
        previous = (state[0], 7, *state[2:69])
        original_state = tuple(list(previous))
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(previous, original_state)
        self.assertTrue(restored.enable_crc)
        for name, value in BASE_EVENT_FIELD_VALUES.items():
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

    def test_linear_pool_avg_prompt_tokens_survives_worker_spawn(self):
        config = KVCacheConfig()
        self.assertEqual(config.linear_pool_avg_prompt_tokens, -1)
        config.linear_pool_avg_prompt_tokens = 100_000
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.linear_pool_avg_prompt_tokens, 100_000)
        state = config.__getstate__()
        legacy = KVCacheConfig.__new__(KVCacheConfig)
        legacy.__setstate__((state[0], 12, *state[2:84]))
        self.assertEqual(legacy.linear_pool_avg_prompt_tokens, -1)
        disabled = KVCacheConfig.__new__(KVCacheConfig)
        disabled.__setstate__((*state[:84], 0))
        self.assertEqual(disabled.linear_pool_avg_prompt_tokens, -1)

    def test_pool_settings_round_trip_and_version10_defaults(self):
        config = KVCacheConfig()
        config.memory_cache_swa_block_ratio = 0.5
        config.dsv41_decoder_swa_pool_blocks = 512
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.memory_cache_swa_block_ratio, 0.5)
        self.assertEqual(restored.dsv41_decoder_swa_pool_blocks, 512)
        state = config.__getstate__()
        previous = (state[0], 10, *state[2:78])
        legacy = KVCacheConfig.__new__(KVCacheConfig)
        legacy.__setstate__(previous)
        self.assertEqual(legacy.memory_cache_swa_block_ratio, 1.0)
        self.assertEqual(legacy.dsv41_decoder_swa_pool_blocks, 0)
        self.assertEqual(previous[1], 10)

    def test_backend_and_snapshot_fields_survive_worker_round_trips(self):
        config = KVCacheConfig()
        values = {
            "kvcm_task_timeout_ms": 12345,
            "kvcm_read_backend_type": 2,
            "kvcm_min_replica_count": 3,
            "kv_cache_event_snapshot_interval_ms": 45000,
        }
        for name, value in values.items():
            setattr(config, name, value)
        for _ in range(2):
            config = pickle.loads(pickle.dumps(config))
            for name, value in values.items():
                self.assertEqual(getattr(config, name), value, name)
        state = config.__getstate__()
        legacy_state = (state[0], 11, *state[2:80])
        legacy = KVCacheConfig.__new__(KVCacheConfig)
        legacy.__setstate__(legacy_state)
        defaults = KVCacheConfig()
        for name in values:
            self.assertEqual(getattr(legacy, name), getattr(defaults, name), name)

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

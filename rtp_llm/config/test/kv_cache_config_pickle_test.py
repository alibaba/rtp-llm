import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import KV_CACHE_EVENT_FIELD_VALUES
from rtp_llm.ops import KVCacheConfig

EVENT_PICKLE_FIELDS = tuple(KV_CACHE_EVENT_FIELD_VALUES)
KVCM_PICKLE_FIELD_VALUES = {
    "kvcm_read_backend_type": 3,
    "kvcm_min_replica_count": 2,
}


class KVCacheConfigPickleTest(TestCase):
    def test_event_fields_round_trip_with_block_tree_configuration(self):
        config = KVCacheConfig()
        config.block_tree_transfer_worker_count = 7
        config.enable_disk_cache = True
        config.disk_cache_paths = "/tmp/cache"
        field_values = {**KV_CACHE_EVENT_FIELD_VALUES, **KVCM_PICKLE_FIELD_VALUES}
        for name, value in field_values.items():
            setattr(config, name, value)
        for enable_crc in (False, True):
            with self.subTest(enable_crc=enable_crc):
                config.enable_crc = enable_crc
                state = config.__getstate__()
                self.assertEqual(len(state), 74)
                self.assertEqual(state[:2], ("KVCacheConfig", 10))
                self.assertEqual(
                    state[64:69], tuple(KV_CACHE_EVENT_FIELD_VALUES.values())
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
                for name, value in field_values.items():
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

    def test_previous_states_preserve_fields_and_use_defaults(self):
        source = KVCacheConfig()
        source.enable_crc = False
        source.dsv4_fixed_pool_blocks = 17
        source.block_tree_transfer_worker_count = 7
        source.memory_cache_swa_block_ratio = 0.5
        source.dsv41_decoder_swa_pool_blocks = 512
        for name, value in {
            **KV_CACHE_EVENT_FIELD_VALUES,
            **KVCM_PICKLE_FIELD_VALUES,
        }.items():
            setattr(source, name, value)
        state = source.__getstate__()
        defaults = KVCacheConfig()
        for version, fields, has_pools, has_kvcm, enable_crc in (
            (6, state[2:64], False, False, True),
            (7, state[2:69], False, False, True),
            (8, state[2:71], True, False, True),
            (8, (*state[2:69], *state[72:74]), False, True, True),
            (9, state[2:72], True, False, False),
            (9, (*state[2:71], True), True, False, True),
            (9, (*state[2:71], *state[72:74]), True, True, True),
        ):
            with self.subTest(
                version=version,
                has_pools=has_pools,
                has_kvcm=has_kvcm,
                enable_crc=enable_crc,
            ):
                previous = (state[0], version, *fields)
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)
                self.assertEqual(previous, (state[0], version, *fields))
                self.assertEqual(restored.dsv4_fixed_pool_blocks, 17)
                self.assertEqual(restored.block_tree_transfer_worker_count, 7)
                self.assertIs(restored.enable_crc, enable_crc)
                expected_pools = source if has_pools else defaults
                self.assertEqual(
                    restored.memory_cache_swa_block_ratio,
                    expected_pools.memory_cache_swa_block_ratio,
                )
                self.assertEqual(
                    restored.dsv41_decoder_swa_pool_blocks,
                    expected_pools.dsv41_decoder_swa_pool_blocks,
                )
                for name in EVENT_PICKLE_FIELDS:
                    expected = source if version >= 7 else defaults
                    self.assertEqual(getattr(restored, name), getattr(expected, name), name)
                for name in KVCM_PICKLE_FIELD_VALUES:
                    expected = source if has_kvcm else defaults
                    self.assertEqual(getattr(restored, name), getattr(expected, name), name)

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

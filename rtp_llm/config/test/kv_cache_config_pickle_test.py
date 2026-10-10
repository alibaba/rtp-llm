import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import KV_CACHE_EVENT_FIELD_VALUES
from rtp_llm.ops import KVCacheConfig

EVENT_PICKLE_FIELDS = tuple(KV_CACHE_EVENT_FIELD_VALUES)
KVCM_PICKLE_FIELD_VALUES = {
    "kvcm_default_query_type": 3,
    "kvcm_query_type": 1,
    "kvcm_sw_size": 8,
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
        state = config.__getstate__()
        self.assertEqual(len(state), 74)
        self.assertEqual(state[:2], ("KVCacheConfig", 8))
        self.assertEqual(state[64:69], tuple(KV_CACHE_EVENT_FIELD_VALUES.values()))
        self.assertEqual(state[69:74], tuple(KVCM_PICKLE_FIELD_VALUES.values()))
        # setstate must normalize its private slice, not mutate the supplied
        # version-8 tuple retained by another Python reference.
        original_state = tuple(list(state))
        direct = KVCacheConfig.__new__(KVCacheConfig)
        direct.__setstate__(state)
        self.assertEqual(state, original_state)
        self.assertEqual(direct.__getstate__(), original_state)
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.block_tree_transfer_worker_count, 7)
        self.assertTrue(restored.enable_disk_cache)
        self.assertEqual(restored.disk_cache_paths, "/tmp/cache")
        for name, value in field_values.items():
            self.assertEqual(getattr(restored, name), value, name)

    def test_previous_states_preserve_fields_and_use_defaults(self):
        source = KVCacheConfig()
        source.dsv4_fixed_pool_blocks = 17
        source.block_tree_transfer_worker_count = 7
        for name, value in {
            **KV_CACHE_EVENT_FIELD_VALUES,
            **KVCM_PICKLE_FIELD_VALUES,
        }.items():
            setattr(source, name, value)
        state = source.__getstate__()
        defaults = KVCacheConfig()
        for version, field_count in ((6, 64), (7, 69)):
            with self.subTest(version=version):
                previous = (state[0], version, *state[2:field_count])
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)
                self.assertEqual(previous, (state[0], version, *state[2:field_count]))
                self.assertEqual(restored.dsv4_fixed_pool_blocks, 17)
                self.assertEqual(restored.block_tree_transfer_worker_count, 7)
                for name in EVENT_PICKLE_FIELDS:
                    expected = source if version == 7 else defaults
                    self.assertEqual(getattr(restored, name), getattr(expected, name), name)
                for name in KVCM_PICKLE_FIELD_VALUES:
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

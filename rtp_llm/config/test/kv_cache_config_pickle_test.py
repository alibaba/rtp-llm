import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import KV_CACHE_EVENT_FIELD_VALUES
from rtp_llm.ops import KVCacheConfig

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
        self.assertEqual(state[:2], ("KVCacheConfig", 9))
        self.assertEqual(state[-7:-2], tuple(KV_CACHE_EVENT_FIELD_VALUES.values()))
        self.assertFalse(state[-2])
        self.assertFalse(state[-1])
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

        config.enable_prefill_partial_tail_reuse = True
        self.assertTrue(
            pickle.loads(pickle.dumps(config)).enable_prefill_partial_tail_reuse
        )

    def test_recovery_and_tail_switches_round_trip_independently(self):
        for tail in (False, True):
            for recovery in (False, True):
                with self.subTest(tail=tail, recovery=recovery):
                    config = KVCacheConfig()
                    config.enable_prefill_partial_tail_reuse = tail
                    config.enable_prefill_common_prefix_recovery = recovery
                    restored = pickle.loads(pickle.dumps(config))
                    self.assertEqual(restored.enable_prefill_partial_tail_reuse, tail)
                    self.assertEqual(
                        restored.enable_prefill_common_prefix_recovery, recovery
                    )

    def test_previous_block_tree_state_uses_event_defaults(self):
        source = KVCacheConfig()
        source.dsv4_fixed_pool_blocks = 17
        source.block_tree_transfer_worker_count = 7
        source.enable_prefill_partial_tail_reuse = True
        state = source.__getstate__()
        previous = (state[0], 6, *state[2:-7])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.dsv4_fixed_pool_blocks, 17)
        self.assertEqual(restored.block_tree_transfer_worker_count, 7)
        defaults = KVCacheConfig()
        for name in EVENT_PICKLE_FIELDS:
            self.assertEqual(getattr(restored, name), getattr(defaults, name), name)
        self.assertFalse(restored.enable_prefill_partial_tail_reuse)
        self.assertFalse(restored.enable_prefill_common_prefix_recovery)

        previous_event = (state[0], 7, *state[2:-2])
        restored_event = KVCacheConfig.__new__(KVCacheConfig)
        restored_event.__setstate__(previous_event)
        self.assertFalse(restored_event.enable_prefill_partial_tail_reuse)
        self.assertFalse(restored_event.enable_prefill_common_prefix_recovery)

        previous_tail = (state[0], 8, *state[2:-1])
        restored_tail = KVCacheConfig.__new__(KVCacheConfig)
        restored_tail.__setstate__(previous_tail)
        self.assertTrue(restored_tail.enable_prefill_partial_tail_reuse)
        self.assertFalse(restored_tail.enable_prefill_common_prefix_recovery)

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

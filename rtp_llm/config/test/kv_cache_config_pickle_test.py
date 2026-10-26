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
        for enable_crc in (False, True):
            for tail_reuse in (False, True):
                for recovery in (False, True):
                    with self.subTest(
                        enable_crc=enable_crc, tail_reuse=tail_reuse, recovery=recovery
                    ):
                        config.enable_crc = enable_crc
                        config.enable_prefill_partial_tail_reuse = tail_reuse
                        config.enable_prefill_common_prefix_recovery = recovery
                        state = config.__getstate__()
                        self.assertEqual(len(state), 74)
                        self.assertEqual(state[:2], ("KVCacheConfig", 11))
                        self.assertEqual(
                            state[64:69], tuple(KV_CACHE_EVENT_FIELD_VALUES.values())
                        )
                        self.assertIs(state[69], enable_crc)
                        self.assertIs(state[70], tail_reuse)
                        self.assertIs(state[71], recovery)
                        # Normalizing old fields must not mutate the supplied tuple.
                        original_state = tuple(list(state))
                        direct = KVCacheConfig.__new__(KVCacheConfig)
                        direct.__setstate__(state)
                        self.assertEqual(state, original_state)
                        self.assertEqual(direct.__getstate__(), original_state)
                        restored = pickle.loads(pickle.dumps(config))
                        self.assertIs(restored.enable_crc, enable_crc)
                        self.assertIs(restored.enable_prefill_partial_tail_reuse, tail_reuse)
                        self.assertIs(restored.enable_prefill_common_prefix_recovery, recovery)
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
        self.assertFalse(restored.enable_prefill_partial_tail_reuse)
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            self.assertEqual(getattr(restored, name), value, name)

    def test_mainline_version_eight_treats_last_field_as_crc(self):
        source = KVCacheConfig()
        source.enable_crc = False
        source.enable_prefill_partial_tail_reuse = True
        state = source.__getstate__()
        previous = (state[0], 8, *state[2:70])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertFalse(restored.enable_crc)
        self.assertFalse(restored.enable_prefill_partial_tail_reuse)

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
        source.enable_crc = False
        source.dsv4_fixed_pool_blocks = 17
        source.block_tree_transfer_worker_count = 7
        source.enable_prefill_partial_tail_reuse = True
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
        self.assertFalse(restored.enable_prefill_partial_tail_reuse)
        self.assertFalse(restored.enable_prefill_common_prefix_recovery)

        previous_event = (state[0], 7, *state[2:69])
        restored_event = KVCacheConfig.__new__(KVCacheConfig)
        restored_event.__setstate__(previous_event)
        self.assertFalse(restored_event.enable_prefill_partial_tail_reuse)
        self.assertFalse(restored_event.enable_prefill_common_prefix_recovery)

        previous_legacy = (state[0], 9, *state[2:69], state[70], state[71])
        restored_legacy = KVCacheConfig.__new__(KVCacheConfig)
        restored_legacy.__setstate__(previous_legacy)
        self.assertTrue(restored_legacy.enable_prefill_partial_tail_reuse)
        self.assertFalse(restored_legacy.enable_prefill_common_prefix_recovery)
        self.assertTrue(restored_legacy.enable_crc)
        previous_legacy_with_recovery = (*previous_legacy[:-1], True)
        restored_legacy_with_recovery = KVCacheConfig.__new__(KVCacheConfig)
        restored_legacy_with_recovery.__setstate__(previous_legacy_with_recovery)
        self.assertTrue(
            restored_legacy_with_recovery.enable_prefill_common_prefix_recovery
        )

        previous_crc_tail = (state[0], 10, *state[2:71])
        restored_crc_tail = KVCacheConfig.__new__(KVCacheConfig)
        restored_crc_tail.__setstate__(previous_crc_tail)
        self.assertFalse(restored_crc_tail.enable_crc)
        self.assertTrue(restored_crc_tail.enable_prefill_partial_tail_reuse)
        self.assertFalse(restored_crc_tail.enable_prefill_common_prefix_recovery)

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

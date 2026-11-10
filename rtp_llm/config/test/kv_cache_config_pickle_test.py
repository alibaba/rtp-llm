import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import KV_CACHE_EVENT_FIELD_VALUES
from rtp_llm.ops import KVCacheConfig

KVCM_PICKLE_FIELD_VALUES = {
    "kvcm_read_backend_type": 3,
    "kvcm_min_replica_count": 2,
}


class KVCacheConfigPickleTest(TestCase):
    def test_event_fields_round_trip_with_block_tree_configuration(self):
        config = KVCacheConfig()
        config.block_tree_transfer_worker_count = 7
        config.btc_crc_slot_count = 3
        config.btc_transfer_engine_async = True
        config.disk_cache_read_thread_count = 3
        config.disk_cache_write_thread_count = 5
        config.enable_disk_cache = True
        config.memory_cache_swa_block_ratio = 0.5
        config.dsv41_decoder_swa_pool_blocks = 512
        config.disk_cache_paths = "/tmp/cache"
        config.disk_cache_sync_timeout_ms = 200
        config.disk_cache_evict_timeout_ms = 45000
        config.kvcm_task_timeout_ms = 23000
        config.block_pool_frozen_abort_lower_ratio = 0.45
        config.block_pool_frozen_abort_upper_ratio = 0.8
        field_values = {**KV_CACHE_EVENT_FIELD_VALUES, **KVCM_PICKLE_FIELD_VALUES}
        for name, value in field_values.items():
            setattr(config, name, value)
        for enable_crc in (False, True):
            with self.subTest(enable_crc=enable_crc):
                config.enable_crc = enable_crc
                state = config.__getstate__()
                self.assertEqual(len(state), 91)
                self.assertEqual(state[:2], ("KVCacheConfig", 18))
                self.assertEqual(
                    state[64:69], tuple(list(KV_CACHE_EVENT_FIELD_VALUES.values())[:5])
                )
                self.assertEqual(state[74], config.kv_cache_event_snapshot_interval_ms)
                self.assertIs(state[71], enable_crc)
                self.assertEqual(state[72:74], tuple(KVCM_PICKLE_FIELD_VALUES.values()))
                self.assertEqual(state[83], 23000)
                self.assertEqual(state[84], 45000)
                self.assertEqual(state[85], 0.45)
                self.assertEqual(state[90], 0.8)
                self.assertEqual(state[86:90], (3, True, 3, 5))
                # Restoring configuration must not mutate the supplied tuple.
                original_state = tuple(list(state))
                direct = KVCacheConfig.__new__(KVCacheConfig)
                direct.__setstate__(state)
                self.assertEqual(state, original_state)
                self.assertEqual(direct.__getstate__(), original_state)
                restored = pickle.loads(pickle.dumps(config))
                self.assertIs(restored.enable_crc, enable_crc)
                self.assertEqual(restored.block_tree_transfer_worker_count, 7)
                self.assertEqual(restored.btc_crc_slot_count, 3)
                self.assertTrue(restored.btc_transfer_engine_async)
                self.assertEqual(restored.disk_cache_read_thread_count, 3)
                self.assertEqual(restored.disk_cache_write_thread_count, 5)
                self.assertTrue(restored.enable_disk_cache)
                self.assertEqual(restored.memory_cache_swa_block_ratio, 0.5)
                self.assertEqual(restored.dsv41_decoder_swa_pool_blocks, 512)
                self.assertEqual(restored.disk_cache_paths, "/tmp/cache")
                self.assertEqual(restored.disk_cache_sync_timeout_ms, 200)
                self.assertEqual(restored.disk_cache_evict_timeout_ms, 45000)
                self.assertEqual(restored.kvcm_task_timeout_ms, 23000)
                self.assertEqual(restored.block_pool_frozen_abort_lower_ratio, 0.45)
                self.assertEqual(
                    restored.block_pool_frozen_abort_upper_ratio, 0.8
                )
                for name, value in field_values.items():
                    self.assertEqual(getattr(restored, name), value, name)
                self.assertIn(f"enable_crc: {int(enable_crc)}\n", restored.to_string())
                self.assertIn(
                    "block_pool_frozen_abort_lower_ratio: 0.45\n", restored.to_string()
                )
                self.assertIn(
                    "block_pool_frozen_abort_upper_ratio: 0.8\n",
                    restored.to_string(),
                )

    def test_async_defaults_and_explicit_values_round_trip(self):
        self.assertTrue(KVCacheConfig().btc_transfer_engine_async)
        self.assertEqual(KVCacheConfig().btc_crc_slot_count, 8)
        self.assertEqual(KVCacheConfig().block_tree_transfer_worker_count, 1)
        self.assertEqual(KVCacheConfig().disk_cache_read_thread_count, 16)
        self.assertEqual(KVCacheConfig().disk_cache_write_thread_count, 16)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = KVCacheConfig()
                config.btc_transfer_engine_async = enabled
                state = config.__getstate__()
                self.assertEqual(state[:2], ("KVCacheConfig", 18))
                self.assertEqual(len(state), 91)
                self.assertIs(state[87], enabled)
                self.assertIs(
                    pickle.loads(pickle.dumps(config)).btc_transfer_engine_async,
                    enabled,
                )

    def test_upstream_versions_preserve_fields_and_default_async_configuration(self):
        source = KVCacheConfig()
        source.block_tree_transfer_worker_count = 7
        source.kvcm_task_timeout_ms = 23000
        source.disk_cache_evict_timeout_ms = 45000
        source.block_pool_frozen_abort_lower_ratio = 0.35
        source.btc_crc_slot_count = 3
        source.btc_transfer_engine_async = False
        source.disk_cache_read_thread_count = 3
        source.disk_cache_write_thread_count = 5
        state = source.__getstate__()
        for version, size in ((13, 83), (14, 84), (15, 85), (16, 86), (17, 90)):
            with self.subTest(version=version):
                previous = (state[0], version, *state[2:size])
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)
                self.assertEqual(restored.block_tree_transfer_worker_count, 7)
                self.assertEqual(
                    restored.kvcm_task_timeout_ms, 20000 if version == 13 else 23000
                )
                self.assertEqual(
                    restored.disk_cache_evict_timeout_ms, -1 if version < 15 else 45000
                )
                self.assertEqual(
                    restored.block_pool_frozen_abort_lower_ratio,
                    0.3 if version < 16 else 0.35,
                )
                self.assertEqual(
                    restored.block_pool_frozen_abort_upper_ratio, 0.4
                )
                self.assertEqual(restored.btc_crc_slot_count, 8 if version < 17 else 3)
                self.assertEqual(restored.btc_transfer_engine_async, version < 17)
                self.assertEqual(
                    restored.disk_cache_read_thread_count, 16 if version < 17 else 3
                )
                self.assertEqual(
                    restored.disk_cache_write_thread_count, 16 if version < 17 else 5
                )

    def test_version_thirteen_uses_default_remote_budget(self):
        source = KVCacheConfig()
        source.kvcm_task_timeout_ms = 23000
        source.enable_crc = False
        source.kvcm_read_backend_type = 3
        source.kvcm_min_replica_count = 2
        state = source.__getstate__()
        previous = (state[0], 13, *state[2:83])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.kvcm_task_timeout_ms, 20000)
        self.assertFalse(restored.enable_crc)
        self.assertEqual(restored.kvcm_read_backend_type, 3)
        self.assertEqual(restored.kvcm_min_replica_count, 2)
        self.assertEqual(restored.block_pool_frozen_abort_lower_ratio, 0.3)
        self.assertEqual(restored.__getstate__()[:2], ("KVCacheConfig", 18))

    def test_disk_timeout_defaults_and_overrides_survive_worker_round_trips(self):
        self.assertEqual(KVCacheConfig().disk_cache_sync_timeout_ms, 30000)
        self.assertEqual(KVCacheConfig().disk_cache_evict_timeout_ms, -1)
        for eviction_timeout in (-1, 45000):
            with self.subTest(eviction_timeout=eviction_timeout):
                config = KVCacheConfig()
                config.disk_cache_sync_timeout_ms = 200
                config.disk_cache_evict_timeout_ms = eviction_timeout
                config.kvcm_task_timeout_ms = 23000
                for _ in range(2):
                    config = pickle.loads(pickle.dumps(config))
                    self.assertEqual(config.disk_cache_sync_timeout_ms, 200)
                    self.assertEqual(
                        config.disk_cache_evict_timeout_ms, eviction_timeout
                    )
                    self.assertEqual(config.kvcm_task_timeout_ms, 23000)

    def test_older_versions_inherit_the_disk_load_timeout_for_eviction(self):
        source = KVCacheConfig()
        source.disk_cache_sync_timeout_ms = 4321
        source.disk_cache_evict_timeout_ms = 45000
        source.kvcm_task_timeout_ms = 23000
        state = source.__getstate__()
        for version, size in ((13, 83), (14, 84)):
            with self.subTest(version=version):
                previous = (state[0], version, *state[2:size])
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)
                self.assertEqual(restored.disk_cache_sync_timeout_ms, 4321)
                self.assertEqual(restored.disk_cache_evict_timeout_ms, -1)
                self.assertEqual(
                    restored.kvcm_task_timeout_ms, 20000 if version == 13 else 23000
                )

    def test_older_versions_use_default_frozen_abort_ratio(self):
        source = KVCacheConfig()
        source.kvcm_task_timeout_ms = 23000
        source.disk_cache_evict_timeout_ms = 45000
        source.block_pool_frozen_abort_lower_ratio = 0.35
        state = source.__getstate__()
        for version, size in ((14, 84), (15, 85)):
            with self.subTest(version=version):
                previous = (state[0], version, *state[2:size])
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)
                self.assertEqual(restored.kvcm_task_timeout_ms, 23000)
                self.assertEqual(
                    restored.disk_cache_evict_timeout_ms, -1 if version == 14 else 45000
                )
                self.assertEqual(restored.block_pool_frozen_abort_lower_ratio, 0.3)

    def test_frozen_abort_range_defaults_and_boundaries_round_trip(self):
        config = KVCacheConfig()
        self.assertEqual(config.block_pool_frozen_abort_lower_ratio, 0.3)
        self.assertEqual(config.block_pool_frozen_abort_upper_ratio, 0.4)
        for lower, upper in (
            (0.0, 0.0),
            (0.0, 0.4),
            (0.3, 0.8),
            (0.5, 0.5),
            (1.0, 1.0),
        ):
            with self.subTest(lower=lower, upper=upper):
                config.block_pool_frozen_abort_lower_ratio = lower
                config.block_pool_frozen_abort_upper_ratio = upper
                restored = pickle.loads(pickle.dumps(config))
                self.assertEqual(restored.block_pool_frozen_abort_lower_ratio, lower)
                self.assertEqual(
                    restored.block_pool_frozen_abort_upper_ratio, upper
                )

    def test_invalid_frozen_abort_ranges_are_rejected(self):
        state = KVCacheConfig().__getstate__()
        invalid_values = (-0.1, 1.1, float("nan"), float("inf"), float("-inf"))
        invalid_ranges = (
            [(value, 0.4) for value in invalid_values]
            + [(0.3, value) for value in invalid_values]
            + [(0.5, 0.4), (1.0, 0.4)]
        )
        for lower, upper in invalid_ranges:
            with self.subTest(lower=lower, upper=upper), self.assertRaisesRegex(
                RuntimeError, "invalid KVCacheConfig block pool frozen abort ratio range"
            ):
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__((*state[:85], lower, *state[86:90], upper))

    def test_old_frozen_abort_lower_above_default_upper_is_rejected(self):
        state = KVCacheConfig().__getstate__()
        for version, size in ((16, 86), (17, 90)):
            previous = (state[0], version, *state[2:85], 0.45, *state[86:size])
            with self.subTest(version=version), self.assertRaisesRegex(
                RuntimeError, "invalid KVCacheConfig block pool frozen abort ratio range"
            ):
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)

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

    def test_malformed_event_states_are_rejected(self):
        state = KVCacheConfig().__getstate__()
        for malformed in (
            state[:-1],
            state + ("extra",),
            (state[0], 99, *state[2:]),
            (state[0], 13, *state[2:]),
            (state[0], 14, *state[2:]),
            (state[0], 15, *state[2:]),
            (state[0], 16, *state[2:]),
            (state[0], 17, *state[2:]),
            state[:88],
            (state[0], 11, *state[2:75]),
            ("OtherConfig", *state[1:]),
        ):
            with self.subTest(state=malformed), self.assertRaisesRegex(
                RuntimeError, "invalid KVCacheConfig state"
            ):
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(malformed)


if __name__ == "__main__":
    main()

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
        config.enable_disk_cache = True
        config.memory_cache_swa_block_ratio = 0.5
        config.dsv41_decoder_swa_pool_blocks = 512
        config.disk_cache_paths = "/tmp/cache"
        config.disk_cache_read_thread_count = 3
        config.disk_cache_write_thread_count = 5
        config.kvcm_task_timeout_ms = 23000
        field_values = {**KV_CACHE_EVENT_FIELD_VALUES, **KVCM_PICKLE_FIELD_VALUES}
        for name, value in field_values.items():
            setattr(config, name, value)
        for enable_crc in (False, True):
            with self.subTest(enable_crc=enable_crc):
                config.enable_crc = enable_crc
                state = config.__getstate__()
                self.assertEqual(len(state), 88)
                self.assertEqual(state[:2], ("KVCacheConfig", 17))
                self.assertEqual(
                    state[64:69], tuple(list(KV_CACHE_EVENT_FIELD_VALUES.values())[:5])
                )
                self.assertEqual(state[74], config.kv_cache_event_snapshot_interval_ms)
                self.assertIs(state[71], enable_crc)
                self.assertEqual(state[72:74], tuple(KVCM_PICKLE_FIELD_VALUES.values()))
                self.assertEqual(state[83], 23000)
                self.assertEqual(state[84], 3)
                self.assertEqual(state[86:88], (3, 5))
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
                self.assertTrue(restored.enable_disk_cache)
                self.assertEqual(restored.memory_cache_swa_block_ratio, 0.5)
                self.assertEqual(restored.dsv41_decoder_swa_pool_blocks, 512)
                self.assertEqual(restored.disk_cache_paths, "/tmp/cache")
                self.assertEqual(restored.disk_cache_read_thread_count, 3)
                self.assertEqual(restored.disk_cache_write_thread_count, 5)
                self.assertEqual(restored.kvcm_task_timeout_ms, 23000)
                for name, value in field_values.items():
                    self.assertEqual(getattr(restored, name), value, name)
                self.assertIn(f"enable_crc: {int(enable_crc)}\n", restored.to_string())

    def test_async_defaults_and_explicit_values_keep_the_existing_schema(self):
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
                self.assertEqual(state[:2], ("KVCacheConfig", 17))
                self.assertEqual(len(state), 88)
                self.assertIs(state[85], enabled)
                self.assertIs(
                    pickle.loads(pickle.dumps(config)).btc_transfer_engine_async,
                    enabled,
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
        self.assertEqual(restored.btc_crc_slot_count, 8)
        self.assertTrue(restored.btc_transfer_engine_async)
        self.assertFalse(restored.enable_crc)
        self.assertEqual(restored.kvcm_read_backend_type, 3)
        self.assertEqual(restored.kvcm_min_replica_count, 2)
        self.assertEqual(restored.__getstate__()[:2], ("KVCacheConfig", 17))

    def test_version_fourteen_preserves_remote_budget_and_defaults_crc_slots(
        self,
    ):
        source = KVCacheConfig()
        source.kvcm_task_timeout_ms = 23000
        source.btc_crc_slot_count = 3
        state = source.__getstate__()
        previous = (state[0], 14, *state[2:84])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.kvcm_task_timeout_ms, 23000)
        self.assertEqual(restored.btc_crc_slot_count, 8)
        self.assertTrue(restored.btc_transfer_engine_async)
        self.assertEqual(restored.__getstate__()[:2], ("KVCacheConfig", 17))

    def test_version_fifteen_preserves_slots_and_defaults_to_async(self):
        source = KVCacheConfig()
        source.btc_crc_slot_count = 3
        source.btc_transfer_engine_async = False
        state = source.__getstate__()
        previous = (state[0], 15, *state[2:85])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.btc_crc_slot_count, 3)
        self.assertTrue(restored.btc_transfer_engine_async)

    def test_version_sixteen_preserves_workers_and_defaults_disk_threads(self):
        source = KVCacheConfig()
        source.block_tree_transfer_worker_count = 2
        source.btc_transfer_engine_async = False
        source.disk_cache_read_thread_count = 3
        source.disk_cache_write_thread_count = 5
        state = source.__getstate__()
        previous = (state[0], 16, *state[2:86])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.block_tree_transfer_worker_count, 2)
        self.assertFalse(restored.btc_transfer_engine_async)
        self.assertEqual(restored.disk_cache_read_thread_count, 16)
        self.assertEqual(restored.disk_cache_write_thread_count, 16)

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

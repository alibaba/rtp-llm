import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import KV_CACHE_EVENT_FIELD_VALUES
from rtp_llm.ops import DataType, KVCacheConfig, LinearAttentionConfig

BASE_EVENT_FIELD_VALUES = {
    name: value
    for name, value in KV_CACHE_EVENT_FIELD_VALUES.items()
    if name != "kv_cache_event_snapshot_interval_ms"
}
EVENT_PICKLE_FIELDS = tuple(BASE_EVENT_FIELD_VALUES)


def _version_twenty_two_state(config):
    state = config.__getstate__()
    # The former host-placement option was between pool counts and scan interval.
    # Drop the v24 tail reclaim flag so older states keep their mainline shapes.
    return (state[0], 22, *state[2:53], True, *state[53:79])


def _version_twenty_state(config):
    state = _version_twenty_two_state(config)
    return (
        state[0],
        20,
        *state[2:72],
        # FULL watermarks
        0.90,
        0.95,
        0.94,
        0.97,
        0.96,
        0.98,
        *state[72:75],
        # KVCM backend and replica count
        2,
        3,
        state[75],
        # CRC slots, async transfer, disk read and write threads
        3,
        False,
        3,
        5,
        state[76],
        # Frozen abort bounds
        0.35,
        0.45,
        state[77],
        # Frozen recovery interval and query timeout
        2500,
        750,
        *state[78:],
    )


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
        # A mainline version-7 state ends at the event fields; indices past
        # that (including the v24 tail reclaim flag) must not reach it.
        previous = (state[0], 7, *state[2:69])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertFalse(restored.enable_linear_cache_reclaim)

    def test_event_fields_round_trip_with_block_tree_configuration(self):
        config = KVCacheConfig()
        config.block_tree_task_pool_size = 6
        config.block_tree_transfer_worker_count = 7
        config.disk_cache_evict_max_pending_bytes = 80 * 1024 * 1024
        config.enable_disk_cache = True
        config.disk_cache_paths = "/tmp/cache"
        for name, value in BASE_EVENT_FIELD_VALUES.items():
            setattr(config, name, value)
        for enable_crc in (False, True):
            with self.subTest(enable_crc=enable_crc):
                config.enable_crc = enable_crc
                config.enable_linear_cache_reclaim = True
                state = config.__getstate__()
                self.assertEqual(len(state), 80)
                self.assertEqual(state[:2], ("KVCacheConfig", 24))
                self.assertEqual(state[63:68], tuple(BASE_EVENT_FIELD_VALUES.values()))
                self.assertIs(state[68], enable_crc)
                self.assertEqual(state[77], 6)
                # v24 appends the reclaim flag after main's v23 fields.
                self.assertTrue(state[79])
                # Restoring fields must not mutate the supplied tuple.
                original_state = tuple(list(state))
                direct = KVCacheConfig.__new__(KVCacheConfig)
                direct.__setstate__(state)
                self.assertEqual(state, original_state)
                self.assertEqual(direct.__getstate__(), original_state)
                restored = pickle.loads(pickle.dumps(config))
                self.assertIs(restored.enable_crc, enable_crc)
                self.assertEqual(restored.block_tree_task_pool_size, 6)
                self.assertTrue(restored.enable_linear_cache_reclaim)
                self.assertEqual(restored.block_tree_transfer_worker_count, 7)
                self.assertEqual(
                    restored.disk_cache_evict_max_pending_bytes, 80 * 1024 * 1024
                )
                self.assertTrue(restored.enable_disk_cache)
                self.assertEqual(restored.disk_cache_paths, "/tmp/cache")
                for name, value in BASE_EVENT_FIELD_VALUES.items():
                    self.assertEqual(getattr(restored, name), value, name)
                self.assertIn(f"enable_crc: {int(enable_crc)}\n", restored.to_string())
                self.assertIn("block_tree_task_pool_size: 6\n", restored.to_string())
                # A mainline version-8 state (crc tail, no reclaim) must still
                # decode with crc preserved and reclaim defaulted off.
                crc_only = (state[0], 8, *state[2:70])
                restored_crc_only = KVCacheConfig.__new__(KVCacheConfig)
                restored_crc_only.__setstate__(crc_only)
                self.assertIs(restored_crc_only.enable_crc, enable_crc)
                self.assertFalse(restored_crc_only.enable_linear_cache_reclaim)

    def test_previous_versions_use_default_pending_budget(self):
        source = KVCacheConfig()
        source.disk_cache_evict_max_pending_bytes = 80 * 1024 * 1024
        state = _version_twenty_state(source)
        for version, size in (
            (6, 64),
            (7, 69),
            (8, 70),
            (9, 72),
            (10, 78),
            (11, 80),
            (12, 84),
            (13, 88),
            (14, 89),
            (15, 90),
            (16, 91),
        ):
            with self.subTest(version=version):
                previous = (state[0], version, *state[2:size])
                original_state = tuple(list(previous))
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)
                self.assertEqual(previous, original_state)
                self.assertEqual(
                    restored.disk_cache_evict_max_pending_bytes, 100 * 1024 * 1024
                )
                self.assertFalse(restored.enable_linear_cache_reclaim)

    def test_version_seven_preserves_events_and_enables_crc_by_default(self):
        source = KVCacheConfig()
        source.enable_crc = False
        for name, value in BASE_EVENT_FIELD_VALUES.items():
            setattr(source, name, value)
        state = _version_twenty_two_state(source)
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
        state = _version_twenty_two_state(source)
        previous = (state[0], 6, *state[2:64])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.dsv4_fixed_pool_blocks, 17)
        self.assertEqual(restored.block_tree_transfer_worker_count, 7)
        self.assertTrue(restored.enable_crc)
        defaults = KVCacheConfig()
        for name in EVENT_PICKLE_FIELDS:
            self.assertEqual(getattr(restored, name), getattr(defaults, name), name)

    def test_version_twelve_preserves_snapshot_and_task_timeout(self):
        source = KVCacheConfig()
        source.enable_crc = False
        source.block_tree_transfer_worker_count = 7
        source.kvcm_task_timeout_ms = 23000
        source.kv_cache_event_snapshot_interval_ms = 45000
        state = _version_twenty_state(source)
        previous = (state[0], 12, *state[2:84])
        original = tuple(list(previous))
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(previous, original)
        self.assertEqual(restored.__getstate__()[:75], source.__getstate__()[:75])

    def test_legacy_states_ignore_retired_options(self):
        source = KVCacheConfig()
        source.memory_cache_swa_block_ratio = 0.5
        source.dsv41_decoder_swa_pool_blocks = 512
        source.kvcm_task_timeout_ms = 23000
        source.kv_cache_event_snapshot_interval_ms = 45000
        source.disk_cache_evict_timeout_ms = 5000
        source.disk_cache_evict_max_pending_bytes = 80 * 1024 * 1024
        source.block_tree_task_pool_size = 6
        source.block_tree_frozen_force_release_timeout_ms = 120000

        state = source.__getstate__()
        v22_state = _version_twenty_two_state(source)
        for previous in (
            _version_twenty_state(source),
            (state[0], 21, *v22_state[2:77], 0.35, 0.45, *v22_state[77:]),
        ):
            with self.subTest(version=previous[1]):
                original = tuple(list(previous))
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)
                self.assertEqual(previous, original)
                self.assertEqual(restored.__getstate__(), state)
                self.assertFalse(
                    hasattr(restored, "block_pool_frozen_abort_lower_ratio")
                )
                self.assertFalse(
                    hasattr(restored, "block_pool_frozen_abort_upper_ratio")
                )

    def test_version_twenty_two_ignores_fixed_pool_host_placement(self):
        config = KVCacheConfig()
        config.dsv4_fixed_pool_blocks = 512
        config.dsv4_hca_state_pool_blocks = 256
        config.block_tree_full_prefix_scan_interval_ms = 1500
        config.block_tree_transfer_worker_count = 5
        config.block_tree_frozen_force_release_timeout_ms = 120000

        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(_version_twenty_two_state(config))
        self.assertEqual(restored.__getstate__(), config.__getstate__())

    def test_disk_timeout_defaults_and_overrides_survive_worker_round_trips(self):
        self.assertEqual(KVCacheConfig().disk_cache_sync_timeout_ms, 3000)
        self.assertEqual(KVCacheConfig().disk_cache_evict_timeout_ms, 3000)
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
        state = _version_twenty_state(source)
        for version, size in ((12, 84), (13, 88)):
            with self.subTest(version=version):
                previous = (state[0], version, *state[2:size])
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)
                self.assertEqual(restored.disk_cache_sync_timeout_ms, 4321)
                self.assertEqual(restored.disk_cache_evict_timeout_ms, -1)
                self.assertEqual(restored.kvcm_task_timeout_ms, 23000)

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
        state = list(_version_twenty_two_state(source)[:70])
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

    def test_pool_settings_round_trip_and_version10_defaults(self):
        config = KVCacheConfig()
        config.memory_cache_swa_block_ratio = 0.5
        config.dsv41_decoder_swa_pool_blocks = 512
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.memory_cache_swa_block_ratio, 0.5)
        self.assertEqual(restored.dsv41_decoder_swa_pool_blocks, 512)
        state = _version_twenty_state(config)
        previous = (state[0], 10, *state[2:78])
        legacy = KVCacheConfig.__new__(KVCacheConfig)
        legacy.__setstate__(previous)
        self.assertEqual(legacy.memory_cache_swa_block_ratio, 1.0)
        self.assertEqual(legacy.dsv41_decoder_swa_pool_blocks, 0)
        self.assertEqual(previous[1], 10)

    def test_remote_timeout_and_snapshot_fields_survive_worker_round_trips(self):
        config = KVCacheConfig()
        values = {
            "kvcm_task_timeout_ms": 12345,
            "kv_cache_event_snapshot_interval_ms": 45000,
        }
        for name, value in values.items():
            setattr(config, name, value)
        for _ in range(2):
            config = pickle.loads(pickle.dumps(config))
            for name, value in values.items():
                self.assertEqual(getattr(config, name), value, name)
        state = _version_twenty_state(config)
        legacy_state = (state[0], 11, *state[2:80])
        legacy = KVCacheConfig.__new__(KVCacheConfig)
        legacy.__setstate__(legacy_state)
        defaults = KVCacheConfig()
        for name in values:
            self.assertEqual(getattr(legacy, name), getattr(defaults, name), name)

    def test_frozen_force_release_timeout_defaults_boundaries_and_validation(self):
        name = "block_tree_frozen_force_release_timeout_ms"
        config = KVCacheConfig()
        self.assertEqual(getattr(config, name), 300000)
        for timeout in (0, 1, 120000, 1200000, 2**31 - 1):
            with self.subTest(timeout=timeout):
                setattr(config, name, timeout)
                for _ in range(2):
                    config = pickle.loads(pickle.dumps(config))
                    self.assertEqual(getattr(config, name), timeout)
                    self.assertEqual(config.__getstate__()[-1], timeout)
                    self.assertIn(f"{name}: {timeout}\n", config.to_string())
        for timeout in (-1, 2**31):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(
                RuntimeError, "invalid KVCacheConfig frozen recovery timing"
            ):
                state = KVCacheConfig().__getstate__()
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__((*state[:-1], timeout))

    def test_previous_versions_use_default_frozen_force_release_timeout(self):
        source = KVCacheConfig()
        source.block_tree_task_pool_size = 6
        source.block_tree_transfer_worker_count = 7
        source.block_tree_frozen_force_release_timeout_ms = 120000
        state = _version_twenty_state(source)
        for version, size in (
            (6, 64),
            (7, 69),
            (8, 70),
            (9, 72),
            (10, 78),
            (11, 80),
            (12, 84),
            (13, 88),
            (14, 89),
            (15, 90),
            (16, 91),
            (17, 92),
            (18, 94),
            (19, 95),
        ):
            with self.subTest(version=version):
                previous = (state[0], version, *state[2:size])
                original = tuple(list(previous))
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(previous)
                self.assertEqual(previous, original)
                self.assertEqual(
                    restored.block_tree_frozen_force_release_timeout_ms, 300000
                )
                self.assertEqual(
                    restored.block_tree_task_pool_size, 6 if version == 19 else 4
                )
                self.assertEqual(restored.block_tree_transfer_worker_count, 7)

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

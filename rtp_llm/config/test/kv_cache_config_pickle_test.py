import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_config_state_layout import (
    CURRENT_STATE_SIZE,
    DISK_CACHE_FIELDS,
    DISK_FIELD_VALUES,
    DISK_STATE_SIZE,
    KV_CACHE_EVENT_FIELDS,
    LEGACY_BASE_STATE_SIZE,
)
from rtp_llm.config.test.kv_cache_event_test_values import (
    KV_CACHE_EVENT_DEFAULTS,
    KV_CACHE_EVENT_ENV_CASES,
    KV_CACHE_EVENT_FIELD_VALUES,
)
from rtp_llm.ops import KVCacheConfig


class KVCacheConfigPickleTest(TestCase):
    def test_shared_field_tables_are_consistent(self):
        self.assertEqual(
            DISK_STATE_SIZE, LEGACY_BASE_STATE_SIZE + len(DISK_CACHE_FIELDS)
        )
        self.assertEqual(set(DISK_CACHE_FIELDS), set(DISK_FIELD_VALUES))
        self.assertEqual(
            CURRENT_STATE_SIZE, DISK_STATE_SIZE + len(KV_CACHE_EVENT_FIELDS)
        )
        self.assertEqual(set(KV_CACHE_EVENT_FIELDS), set(KV_CACHE_EVENT_DEFAULTS))
        self.assertEqual(
            KV_CACHE_EVENT_FIELDS,
            tuple(case.field_name for case in KV_CACHE_EVENT_ENV_CASES),
        )

    def test_cpp_defaults_match_shared_defaults(self):
        defaults = KVCacheConfig()
        for name, value in KV_CACHE_EVENT_DEFAULTS.items():
            with self.subTest(name=name):
                self.assertEqual(value, getattr(defaults, name))

    def test_event_fields_round_trip_in_current_state(self):
        config = KVCacheConfig()
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            setattr(config, name, value)

        self.assertEqual(CURRENT_STATE_SIZE, len(config.__getstate__()))
        restored = pickle.loads(pickle.dumps(config))

        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            self.assertEqual(value, getattr(restored, name), name)

    def test_event_pickle_block_follows_declaration_order(self):
        config = KVCacheConfig()
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            setattr(config, name, value)

        self.assertEqual(
            tuple(KV_CACHE_EVENT_FIELD_VALUES[name] for name in KV_CACHE_EVENT_FIELDS),
            config.__getstate__()[DISK_STATE_SIZE:],
        )

    def test_disk_pickle_block_follows_declaration_order(self):
        config = KVCacheConfig()
        for name, value in DISK_FIELD_VALUES.items():
            setattr(config, name, value)

        self.assertEqual(
            tuple(DISK_FIELD_VALUES[name] for name in DISK_CACHE_FIELDS),
            config.__getstate__()[LEGACY_BASE_STATE_SIZE:DISK_STATE_SIZE],
        )

    def test_legacy_54_element_state_uses_event_defaults(self):
        source = KVCacheConfig()
        for name, value in DISK_FIELD_VALUES.items():
            setattr(source, name, value)
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            setattr(source, name, value)
        legacy_state = source.__getstate__()[:DISK_STATE_SIZE]
        self.assertEqual(DISK_STATE_SIZE, len(legacy_state))

        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(legacy_state)

        for name, value in DISK_FIELD_VALUES.items():
            self.assertEqual(value, getattr(restored, name), name)
        defaults = KVCacheConfig()
        for name in KV_CACHE_EVENT_FIELD_VALUES:
            self.assertEqual(getattr(defaults, name), getattr(restored, name), name)

    def test_legacy_43_element_state_uses_disk_and_event_defaults(self):
        source = KVCacheConfig()
        for name, value in DISK_FIELD_VALUES.items():
            setattr(source, name, value)
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            setattr(source, name, value)
        legacy_state = source.__getstate__()[:LEGACY_BASE_STATE_SIZE]
        self.assertEqual(LEGACY_BASE_STATE_SIZE, len(legacy_state))

        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(legacy_state)

        defaults = KVCacheConfig()
        for name in (*DISK_CACHE_FIELDS, *KV_CACHE_EVENT_FIELD_VALUES):
            self.assertEqual(getattr(defaults, name), getattr(restored, name), name)

    def test_only_published_pickle_layouts_are_accepted(self):
        current_state = KVCacheConfig().__getstate__()
        oversized_state = current_state + (None, None)
        accepted_sizes = {
            LEGACY_BASE_STATE_SIZE,
            DISK_STATE_SIZE,
            CURRENT_STATE_SIZE,
        }
        for size in range(CURRENT_STATE_SIZE + 2):
            if size in accepted_sizes:
                continue
            with self.subTest(size=size), self.assertRaisesRegex(
                RuntimeError, "Invalid state"
            ):
                restored = KVCacheConfig.__new__(KVCacheConfig)
                # Slice from an actually oversized tuple so sizes above the
                # current layout do not silently clamp back to CURRENT_STATE_SIZE.
                restored.__setstate__(oversized_state[:size])


if __name__ == "__main__":
    main()

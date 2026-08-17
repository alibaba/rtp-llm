import pickle
import unittest

from rtp_llm.ops import GrammarConfig, KVCacheConfig

_ARITY_TRIPWIRE_MSG = (
    "config pickle state arity changed: after adding or removing a field, update "
    "the accepted t.size() list in that config's __setstate__ in "
    "rtp_llm/cpp/pybind/ConfigInit.cc, keep the previous arity accepted so older "
    "states still load, and append new fields at the end so existing indices stay stable"
)
_CURRENT_KV_CACHE_CONFIG_ARITY = 56
_KV_CACHE_CONFIG_LEGACY_STATE_43 = (
    True,
    "legacy-task",
    "legacy-prompt",
    {"legacy": [3, 5]},
    7,
    8,
    901,
    902,
    9,
    1,
    1234,
    16,
    32,
    11,
    1,
    True,
    False,
    True,
    False,
    True,
    False,
    12,
    True,
    "legacy-domain",
    "legacy-address",
    "legacy-group",
    101,
    102,
    103,
    4,
    5,
    104,
    105,
    "legacy-sdk",
    "legacy-user-data",
    "legacy-extra-info",
    "legacy-salt",
    6,
    7,
    106,
    107,
    "legacy-client-config",
    "fp16",
)
_KV_CACHE_CONFIG_LEGACY_STATE_54 = _KV_CACHE_CONFIG_LEGACY_STATE_43 + (
    True,
    "/legacy/cache",
    2048,
    False,
    3000,
    True,
    False,
    True,
    25,
    True,
    3,
)
_KV_CACHE_CONFIG_LEGACY_STATES = {
    43: _KV_CACHE_CONFIG_LEGACY_STATE_43,
    54: _KV_CACHE_CONFIG_LEGACY_STATE_54,
}
_ACCEPTED_KV_CACHE_CONFIG_ARITIES = tuple(sorted(_KV_CACHE_CONFIG_LEGACY_STATES)) + (
    _CURRENT_KV_CACHE_CONFIG_ARITY,
)


def _new_grammar_config():
    return GrammarConfig.__new__(GrammarConfig)


class _LegacyGrammarConfig:
    def __reduce__(self):
        legacy_state = ("xgrammar", True, 3, "tokenizer-info", [7, 11])
        return _new_grammar_config, (), legacy_state


class _PreviousGrammarConfig:
    def __reduce__(self):
        previous_state = (True, 4, "previous-tokenizer-info", [5, 9], 2048)
        return _new_grammar_config, (), previous_state


class _PreviousSixTupleGrammarConfig:
    def __reduce__(self):
        previous_state = (True, 6, "six-tokenizer-info", [13, 17], 4096, True)
        return _new_grammar_config, (), previous_state


class GrammarConfigPickleTest(unittest.TestCase):
    def test_current_format_round_trip(self):
        config = GrammarConfig()
        config.constrained_json_disable_any_whitespace = True
        config.num_workers = 5
        config.tokenizer_info_json = "current-tokenizer-info"
        config.compiler_cache_bytes = 1024
        config.terminate_without_stop_token = True

        restored = pickle.loads(pickle.dumps(config))

        self.assertTrue(restored.constrained_json_disable_any_whitespace)
        self.assertEqual(restored.num_workers, 5)
        self.assertEqual(restored.tokenizer_info_json, "current-tokenizer-info")
        self.assertEqual(restored.compiler_cache_bytes, 1024)
        self.assertTrue(restored.terminate_without_stop_token)
        self.assertFalse(hasattr(restored, "override_stop_tokens"))

    def test_legacy_five_tuple_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_LegacyGrammarConfig()))

        self.assertTrue(restored.constrained_json_disable_any_whitespace)
        self.assertEqual(restored.num_workers, 3)
        self.assertEqual(restored.tokenizer_info_json, "tokenizer-info")
        self.assertEqual(restored.compiler_cache_bytes, 512 * 1024 * 1024)
        self.assertFalse(restored.terminate_without_stop_token)
        self.assertFalse(hasattr(restored, "override_stop_tokens"))

    def test_previous_five_tuple_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_PreviousGrammarConfig()))

        self.assertTrue(restored.constrained_json_disable_any_whitespace)
        self.assertEqual(restored.num_workers, 4)
        self.assertEqual(restored.tokenizer_info_json, "previous-tokenizer-info")
        self.assertEqual(restored.compiler_cache_bytes, 2048)
        self.assertFalse(restored.terminate_without_stop_token)
        self.assertFalse(hasattr(restored, "override_stop_tokens"))

    def test_previous_six_tuple_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_PreviousSixTupleGrammarConfig()))

        self.assertTrue(restored.constrained_json_disable_any_whitespace)
        self.assertEqual(restored.num_workers, 6)
        self.assertEqual(restored.tokenizer_info_json, "six-tokenizer-info")
        self.assertEqual(restored.compiler_cache_bytes, 4096)
        self.assertTrue(restored.terminate_without_stop_token)
        self.assertFalse(hasattr(restored, "override_stop_tokens"))

    def test_fabricated_short_layouts_are_rejected(self):
        for state in ((True, 3, 1024), (True, 3, [7, 11], 1024)):
            with (
                self.subTest(state=state),
                self.assertRaisesRegex(RuntimeError, "Invalid state"),
            ):
                config = _new_grammar_config()
                config.__setstate__(state)


class KVCacheConfigPickleTest(unittest.TestCase):
    def test_current_arity_and_runtime_memory_fields_round_trip(self):
        config = KVCacheConfig()
        config.runtime_mem_safety_ratio = 0.08
        config.runtime_mem_no_warmup_floor_mb = 3072

        self.assertEqual(
            len(config.__getstate__()),
            _CURRENT_KV_CACHE_CONFIG_ARITY,
            msg=_ARITY_TRIPWIRE_MSG,
        )
        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.runtime_mem_safety_ratio, 0.08)
        self.assertEqual(restored.runtime_mem_no_warmup_floor_mb, 3072)

    def test_legacy_arity_keeps_runtime_memory_defaults(self):
        defaults = KVCacheConfig()

        for arity, state in _KV_CACHE_CONFIG_LEGACY_STATES.items():
            with self.subTest(arity=arity):
                self.assertEqual(len(state), arity)
                legacy = KVCacheConfig.__new__(KVCacheConfig)
                legacy.__setstate__(state)

                self.assertTrue(legacy.reuse_cache)
                self.assertEqual(legacy.multi_task_prompt, "legacy-task")
                self.assertEqual(legacy.kv_cache_mem_mb, 1234)
                self.assertEqual(legacy.reco_client_config, "legacy-client-config")
                if arity == 54:
                    self.assertEqual(legacy.memory_cache_disk_paths, "/legacy/cache")
                    self.assertEqual(legacy.load_cache_retry_times, 3)
                self.assertEqual(
                    legacy.runtime_mem_safety_ratio,
                    defaults.runtime_mem_safety_ratio,
                )
                self.assertEqual(
                    legacy.runtime_mem_no_warmup_floor_mb,
                    defaults.runtime_mem_no_warmup_floor_mb,
                )

        # Prove the C++ setter rejects both sides of the current arity and the gap before 54.
        for invalid_arity, state in (
            (42, _KV_CACHE_CONFIG_LEGACY_STATE_43[:42]),
            (53, _KV_CACHE_CONFIG_LEGACY_STATE_54[:53]),
            (55, _KV_CACHE_CONFIG_LEGACY_STATE_54 + (0,)),
            (57, _KV_CACHE_CONFIG_LEGACY_STATE_54 + (0, 0, 0)),
        ):
            with self.subTest(invalid_arity=invalid_arity):
                self.assertEqual(len(state), invalid_arity)
                invalid = KVCacheConfig.__new__(KVCacheConfig)
                with self.assertRaisesRegex(RuntimeError, "Invalid state"):
                    invalid.__setstate__(state)


if __name__ == "__main__":
    unittest.main()

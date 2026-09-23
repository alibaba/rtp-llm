import pickle
import unittest

from rtp_llm.ops import GrammarConfig, KVCacheConfig


def _new_grammar_config():
    return GrammarConfig.__new__(GrammarConfig)


def _new_kv_cache_config():
    return KVCacheConfig.__new__(KVCacheConfig)


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


class _Previous57TupleKVCacheConfig:
    def __init__(self, hca_blocks: int):
        self.hca_blocks = hca_blocks

    def __reduce__(self):
        state = list(KVCacheConfig().__getstate__())
        if len(state) != 58:
            raise AssertionError(f"expected current 58-field state, got {len(state)}")
        state[55] = self.hca_blocks
        return _new_kv_cache_config, (), tuple(state[:57])


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
    def test_current_format_round_trip_preserves_explicit_clear(self):
        config = KVCacheConfig()
        config.dsv4_hca_state_pool_blocks = -1
        config.dsv4_hca_state_pool_clear = True

        self.assertEqual(len(config.__getstate__()), 58)
        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.dsv4_hca_state_pool_blocks, -1)
        self.assertTrue(restored.dsv4_hca_state_pool_clear)

    def test_previous_zero_means_unset(self):
        restored = pickle.loads(pickle.dumps(_Previous57TupleKVCacheConfig(0)))

        self.assertEqual(restored.dsv4_hca_state_pool_blocks, -1)
        self.assertFalse(restored.dsv4_hca_state_pool_clear)

    def test_previous_positive_override_is_preserved(self):
        restored = pickle.loads(pickle.dumps(_Previous57TupleKVCacheConfig(96)))

        self.assertEqual(restored.dsv4_hca_state_pool_blocks, 96)
        self.assertFalse(restored.dsv4_hca_state_pool_clear)


if __name__ == "__main__":
    unittest.main()

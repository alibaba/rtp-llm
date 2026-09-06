import pickle
import unittest

from rtp_llm.ops import CPRotateMethod, GrammarConfig, PrefillCPConfig


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


def _new_prefill_cp_config():
    return PrefillCPConfig.__new__(PrefillCPConfig)


class _LegacyPrefillCPConfig:
    def __reduce__(self):
        return _new_prefill_cp_config, (), (CPRotateMethod.ALL_GATHER, 1024)


class _PreviousPrefillCPConfig:
    def __reduce__(self):
        return _new_prefill_cp_config, (), (
            CPRotateMethod.ALL_GATHER_WITH_OVERLAP,
            2048,
            True,
            4,
        )


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


class PrefillCPConfigPickleTest(unittest.TestCase):
    def test_current_format_round_trip(self):
        config = PrefillCPConfig()
        config.method = CPRotateMethod.ALLTOALL
        config.comm_buffer_size = 4096
        config.kv_cache_sharded = True
        config.prefill_cp_size = 8
        config.segment_size_alignment = 64

        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.method, CPRotateMethod.ALLTOALL)
        self.assertEqual(restored.comm_buffer_size, 4096)
        self.assertTrue(restored.kv_cache_sharded)
        self.assertEqual(restored.prefill_cp_size, 8)
        self.assertEqual(restored.segment_size_alignment, 64)

    def test_legacy_two_tuple_uses_new_defaults(self):
        restored = pickle.loads(pickle.dumps(_LegacyPrefillCPConfig()))

        self.assertEqual(restored.method, CPRotateMethod.ALL_GATHER)
        self.assertEqual(restored.comm_buffer_size, 1024)
        self.assertFalse(restored.kv_cache_sharded)
        self.assertEqual(restored.prefill_cp_size, 0)
        self.assertEqual(restored.segment_size_alignment, 1)

    def test_previous_four_tuple_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_PreviousPrefillCPConfig()))

        self.assertEqual(restored.method, CPRotateMethod.ALL_GATHER_WITH_OVERLAP)
        self.assertEqual(restored.comm_buffer_size, 2048)
        self.assertTrue(restored.kv_cache_sharded)
        self.assertEqual(restored.prefill_cp_size, 4)
        self.assertEqual(restored.segment_size_alignment, 1)

    def test_invalid_layouts_are_rejected(self):
        for state in ((CPRotateMethod.ALL_GATHER,), (CPRotateMethod.ALL_GATHER, 1, False)):
            with (
                self.subTest(state=state),
                self.assertRaisesRegex(RuntimeError, "Invalid state"),
            ):
                config = _new_prefill_cp_config()
                config.__setstate__(state)


if __name__ == "__main__":
    unittest.main()

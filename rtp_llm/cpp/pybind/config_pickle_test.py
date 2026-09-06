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
        expected = {
            "method": CPRotateMethod.ALL_GATHER,
            "comm_buffer_size": 1024,
            "kv_cache_sharded": True,
            "prefill_cp_size": 4,
            "segment_size_alignment": 64,
        }
        for name, value in expected.items():
            setattr(config, name, value)

        restored = pickle.loads(pickle.dumps(config))
        for name, value in expected.items():
            self.assertEqual(getattr(restored, name), value)

    def test_legacy_states_are_loaded(self):
        for state in (
            (CPRotateMethod.ALLTOALL, 128),
            (CPRotateMethod.ALLTOALL, 192, True),
            (CPRotateMethod.ALL_GATHER, 256, True, 2),
        ):
            with self.subTest(state=state):
                restored = PrefillCPConfig.__new__(PrefillCPConfig)
                restored.__setstate__(state)
                expected = (
                    state[0],
                    state[1],
                    state[2] if len(state) >= 3 else False,
                    state[3] if len(state) >= 4 else 0,
                    1,
                )
                self.assertEqual(
                    (
                        restored.method,
                        restored.comm_buffer_size,
                        restored.kv_cache_sharded,
                        restored.prefill_cp_size,
                        restored.segment_size_alignment,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()

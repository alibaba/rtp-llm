import pickle
import unittest

from rtp_llm.ops import GrammarConfig, MoeConfig, PDSepConfig, RoleType


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


def _new_moe_config():
    return MoeConfig.__new__(MoeConfig)


class _LegacyMoeConfig:
    def __reduce__(self):
        legacy_state = (
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            7,
            8192,
            True,
            128,
            "legacy",
        )
        return _new_moe_config, (), legacy_state


def _new_pd_sep_config():
    return PDSepConfig.__new__(PDSepConfig)


class _MalformedPDSepConfig:
    def __reduce__(self):
        return _new_pd_sep_config, (), (RoleType.PDFUSION,) * 20


class _InvalidTypePDSepConfig:
    def __reduce__(self):
        state = [
            RoleType.PDFUSION,
            True,
            "not-an-int",
        ] + [0] * 18
        return _new_pd_sep_config, (), tuple(state)


class _LegacyPDSepConfig:
    def __reduce__(self):
        legacy_state = (
            RoleType.PDFUSION,
            True,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            True,
            118,
        )
        return _new_pd_sep_config, (), legacy_state


class MoeConfigPickleTest(unittest.TestCase):
    def test_non_default_fp4_moe_op_round_trip(self):
        config = MoeConfig()
        config.fp4_moe_op = "cutlass"

        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.fp4_moe_op, "cutlass")

    def test_legacy_twelve_tuple_defaults_fp4_moe_op(self):
        restored = pickle.loads(pickle.dumps(_LegacyMoeConfig()))

        self.assertEqual(restored.moe_strategy, "legacy")
        self.assertEqual(restored.fp4_moe_op, "auto")


class PDSepConfigPickleTest(unittest.TestCase):
    def test_current_format_round_trip(self):
        config = PDSepConfig()
        config.prefill_stop_stream_wait_timeout_ms = 118
        config.prefill_prepare_resource_pool_size = 119

        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.prefill_stop_stream_wait_timeout_ms, 118)
        self.assertEqual(restored.prefill_prepare_resource_pool_size, 119)

    def test_legacy_format_defaults_new_field(self):
        restored = pickle.loads(pickle.dumps(_LegacyPDSepConfig()))

        self.assertEqual(restored.prefill_stop_stream_wait_timeout_ms, 118)
        self.assertEqual(restored.prefill_prepare_resource_pool_size, 0)

    def test_rejects_invalid_state_length(self):
        with self.assertRaisesRegex(RuntimeError, "expected 21 or 22 fields"):
            pickle.loads(pickle.dumps(_MalformedPDSepConfig()))

    def test_rejects_invalid_state_type_with_context(self):
        with self.assertRaisesRegex(RuntimeError, "PDSepConfig unpickle error"):
            pickle.loads(pickle.dumps(_InvalidTypePDSepConfig()))


class GrammarConfigPickleTest(unittest.TestCase):
    def test_current_format_round_trip(self):
        config = GrammarConfig()
        config.constrained_json_disable_any_whitespace = True
        config.num_workers = 5
        config.tokenizer_info_json = "current-tokenizer-info"
        config.compiler_cache_bytes = 1024
        config.terminate_without_stop_token = True
        config.compile_timeout_ms = 1234
        config.compile_concurrency = 3
        config.compile_queue_size = 5

        restored = pickle.loads(pickle.dumps(config))

        self.assertTrue(restored.constrained_json_disable_any_whitespace)
        self.assertEqual(restored.num_workers, 5)
        self.assertEqual(restored.tokenizer_info_json, "current-tokenizer-info")
        self.assertEqual(restored.compiler_cache_bytes, 1024)
        self.assertTrue(restored.terminate_without_stop_token)
        self.assertEqual(restored.compile_timeout_ms, 1234)
        self.assertEqual(restored.compile_concurrency, 3)
        self.assertEqual(restored.compile_queue_size, 5)
        self.assertFalse(hasattr(restored, "override_stop_tokens"))

    def test_legacy_five_tuple_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_LegacyGrammarConfig()))

        self.assertTrue(restored.constrained_json_disable_any_whitespace)
        self.assertEqual(restored.num_workers, 3)
        self.assertEqual(restored.tokenizer_info_json, "tokenizer-info")
        self.assertEqual(restored.compiler_cache_bytes, 2 * 1024 * 1024 * 1024)
        self.assertFalse(restored.terminate_without_stop_token)
        self.assertEqual(restored.compile_timeout_ms, 2000)
        self.assertEqual(restored.compile_concurrency, 1)
        self.assertEqual(restored.compile_queue_size, 2)
        self.assertFalse(hasattr(restored, "override_stop_tokens"))

    def test_previous_five_tuple_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_PreviousGrammarConfig()))

        self.assertTrue(restored.constrained_json_disable_any_whitespace)
        self.assertEqual(restored.num_workers, 4)
        self.assertEqual(restored.tokenizer_info_json, "previous-tokenizer-info")
        self.assertEqual(restored.compiler_cache_bytes, 2048)
        self.assertFalse(restored.terminate_without_stop_token)
        self.assertEqual(restored.compile_timeout_ms, 2000)
        self.assertEqual(restored.compile_concurrency, 1)
        self.assertEqual(restored.compile_queue_size, 2)
        self.assertFalse(hasattr(restored, "override_stop_tokens"))

    def test_previous_six_tuple_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_PreviousSixTupleGrammarConfig()))

        self.assertTrue(restored.constrained_json_disable_any_whitespace)
        self.assertEqual(restored.num_workers, 6)
        self.assertEqual(restored.tokenizer_info_json, "six-tokenizer-info")
        self.assertEqual(restored.compiler_cache_bytes, 4096)
        self.assertTrue(restored.terminate_without_stop_token)
        self.assertEqual(restored.compile_timeout_ms, 2000)
        self.assertEqual(restored.compile_concurrency, 1)
        self.assertEqual(restored.compile_queue_size, 2)
        self.assertFalse(hasattr(restored, "override_stop_tokens"))

    def test_fabricated_short_layouts_are_rejected(self):
        for state in ((True, 3, 1024), (True, 3, [7, 11], 1024)):
            with (
                self.subTest(state=state),
                self.assertRaisesRegex(RuntimeError, "Invalid state"),
            ):
                config = _new_grammar_config()
                config.__setstate__(state)


if __name__ == "__main__":
    unittest.main()

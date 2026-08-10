import pickle
import unittest

from rtp_llm.ops import FMHAConfig, GrammarConfig


def _new_grammar_config():
    return GrammarConfig.__new__(GrammarConfig)


def _new_fmha_config():
    return FMHAConfig.__new__(FMHAConfig)


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


class _LegacyFMHAConfig:
    def __reduce__(self):
        legacy_state = (
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            2049,
            False,
        )
        return _new_fmha_config, (), legacy_state


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


class FMHAConfigPickleTest(unittest.TestCase):
    def test_current_fourteen_tuple_round_trip(self):
        config = FMHAConfig()
        config.enable_fmha = False
        config.enable_flashinfer_trt_fmha_v2 = True
        config.enable_paged_flashinfer_trt_fmha_v2 = False
        config.enable_open_source_fmha = True
        config.enable_paged_open_source_fmha = False
        config.disable_flashinfer_native = True
        config.enable_xqa = False
        config.use_aiter_pa = True
        config.use_asm_pa = False
        config.use_triton_pa = True
        config.absorb_opt_len = 4097
        config.enable_flashinfer_trtllm_gen = False
        config.enable_flashinfer_fa2_target_verify = False
        config.enable_fa4_target_verify = True

        state = config.__getstate__()
        self.assertEqual(len(state), 14)
        self.assertEqual(state[10:], (4097, False, False, True))

        restored = pickle.loads(pickle.dumps(config))

        self.assertFalse(restored.enable_fmha)
        self.assertTrue(restored.enable_flashinfer_trt_fmha_v2)
        self.assertFalse(restored.enable_paged_flashinfer_trt_fmha_v2)
        self.assertTrue(restored.enable_open_source_fmha)
        self.assertFalse(restored.enable_paged_open_source_fmha)
        self.assertTrue(restored.disable_flashinfer_native)
        self.assertFalse(restored.enable_xqa)
        self.assertTrue(restored.use_aiter_pa)
        self.assertFalse(restored.use_asm_pa)
        self.assertTrue(restored.use_triton_pa)
        self.assertEqual(restored.absorb_opt_len, 4097)
        self.assertFalse(restored.enable_flashinfer_trtllm_gen)
        self.assertFalse(restored.enable_flashinfer_fa2_target_verify)
        self.assertTrue(restored.enable_fa4_target_verify)

    def test_legacy_twelve_tuple_uses_defaults_for_new_gates(self):
        restored = pickle.loads(pickle.dumps(_LegacyFMHAConfig()))

        self.assertFalse(restored.enable_fmha)
        self.assertEqual(restored.absorb_opt_len, 2049)
        self.assertFalse(restored.enable_flashinfer_trtllm_gen)
        self.assertTrue(restored.enable_flashinfer_fa2_target_verify)
        self.assertTrue(restored.enable_fa4_target_verify)

    def test_thirteen_tuple_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid state"):
            config = _new_fmha_config()
            config.__setstate__((False,) * 13)


if __name__ == "__main__":
    unittest.main()

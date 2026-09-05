import pickle
import unittest

from rtp_llm.ops import GrammarConfig, HWKernelConfig


def _new_grammar_config():
    return GrammarConfig.__new__(GrammarConfig)


def _new_hw_kernel_config():
    return HWKernelConfig.__new__(HWKernelConfig)


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


class _LegacyHWKernelConfig:
    def __reduce__(self):
        legacy_state = (
            11,
            True,
            False,
            False,
            "legacy.csv",
            True,
            True,
            True,
            True,
            37,
            [64, 128],
            [1, 8],
            True,
            True,
        )
        return _new_hw_kernel_config, (), legacy_state


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


class HWKernelConfigPickleTest(unittest.TestCase):
    def test_current_format_round_trip(self):
        config = HWKernelConfig()
        config.deep_gemm_num_sm = 7
        config.arm_gemm_use_kai = True
        config.enable_multi_block_mode = False
        config.ft_disable_custom_ar = False
        config.rocm_hipblaslt_config = "current.csv"
        config.use_swizzleA = True
        config.enable_cuda_graph = True
        config.enable_cuda_graph_debug_mode = True
        config.enable_prefill_cuda_graph = True
        config.prefill_cuda_graph_max_requests = 5
        config.prefill_cuda_graph_capture_seq_lens = [32, 64, 96]
        config.enable_native_cuda_graph = True
        config.num_native_cuda_graph = 41
        config.prefill_capture_seq_lens = [17, 23]
        config.decode_capture_batch_sizes = [2, 7]
        config.disable_dpc_random = True
        config.rocm_disable_custom_ag = True

        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.deep_gemm_num_sm, 7)
        self.assertTrue(restored.arm_gemm_use_kai)
        self.assertFalse(restored.enable_multi_block_mode)
        self.assertFalse(restored.ft_disable_custom_ar)
        self.assertEqual(restored.rocm_hipblaslt_config, "current.csv")
        self.assertTrue(restored.use_swizzleA)
        self.assertTrue(restored.enable_cuda_graph)
        self.assertTrue(restored.enable_cuda_graph_debug_mode)
        self.assertTrue(restored.enable_prefill_cuda_graph)
        self.assertEqual(restored.prefill_cuda_graph_max_requests, 5)
        self.assertEqual(restored.prefill_cuda_graph_capture_seq_lens, [32, 64, 96])
        self.assertTrue(restored.enable_native_cuda_graph)
        self.assertEqual(restored.num_native_cuda_graph, 41)
        self.assertEqual(restored.prefill_capture_seq_lens, [17, 23])
        self.assertEqual(restored.decode_capture_batch_sizes, [2, 7])
        self.assertTrue(restored.disable_dpc_random)
        self.assertTrue(restored.rocm_disable_custom_ag)

    def test_legacy_14_tuple_uses_prefill_cuda_graph_defaults(self):
        restored = pickle.loads(pickle.dumps(_LegacyHWKernelConfig()))

        self.assertFalse(restored.enable_prefill_cuda_graph)
        self.assertEqual(restored.prefill_cuda_graph_max_requests, 8)
        self.assertEqual(restored.prefill_capture_seq_lens, [64, 128])
        self.assertEqual(restored.decode_capture_batch_sizes, [1, 8])
        self.assertEqual(restored.num_native_cuda_graph, 37)
        self.assertEqual(
            restored.prefill_cuda_graph_capture_seq_lens,
            HWKernelConfig().prefill_cuda_graph_capture_seq_lens,
        )

    def test_unsupported_tuple_sizes_are_rejected(self):
        for size in (15, 16, 18):
            with self.subTest(size=size), self.assertRaisesRegex(
                RuntimeError, "Invalid state"
            ):
                config = _new_hw_kernel_config()
                config.__setstate__(tuple(range(size)))

    def test_current_layout_rejects_wrong_field_type(self):
        malformed_state = (
            11,
            True,
            False,
            False,
            "legacy.csv",
            True,
            True,
            True,
            True,
            37,
            [64, 128],
            [1, 8],
            True,
            True,
            True,
            "not-an-integer",
            [32, 64],
        )
        with self.assertRaisesRegex(RuntimeError, "HWKernelConfig unpickle error"):
            config = _new_hw_kernel_config()
            config.__setstate__(malformed_state)


if __name__ == "__main__":
    unittest.main()

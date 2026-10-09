import pickle
import unittest

from rtp_llm.ops import (
    CacheCapacityPolicyDesc,
    CacheGroupType,
    CacheReusePolicyDesc,
    GrammarConfig,
    HWKernelConfig,
    KVCacheSpecDesc,
    KVCacheSpecType,
    RuntimeConfig,
)


def _new_grammar_config():
    return GrammarConfig.__new__(GrammarConfig)


def _new_hw_kernel_config():
    return HWKernelConfig.__new__(HWKernelConfig)


def _new_kv_cache_spec_desc():
    return KVCacheSpecDesc.__new__(KVCacheSpecDesc)


def _new_cache_capacity_policy_desc():
    return CacheCapacityPolicyDesc.__new__(CacheCapacityPolicyDesc)


class _LegacyCacheCapacityPolicyDesc:
    def __reduce__(self):
        legacy_state = (True, 17, False)
        return _new_cache_capacity_policy_desc, (), legacy_state


class _LegacyKVCacheSpecDesc:
    def __reduce__(self):
        desc = KVCacheSpecDesc()
        desc.tag = "legacy"
        desc.cache_type = KVCacheSpecType.MHA
        desc.entry_elems = 17
        desc.explicit_entry_count = 23
        desc.block_stride_alignment_min_entries = 29
        desc.group_type = CacheGroupType.FULL
        reuse = CacheReusePolicyDesc()
        reuse.enable_prefix_reuse = True
        desc.reuse = reuse
        current_state = desc.__getstate__()
        legacy_state = current_state[:4] + current_state[6:]
        return _new_kv_cache_spec_desc, (), legacy_state


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


class CacheCapacityPolicyDescPickleTest(unittest.TestCase):
    def test_current_format_round_trip(self):
        capacity = CacheCapacityPolicyDesc()
        capacity.reservable = True
        capacity.explicit_block_num = 23
        capacity.charge_to_paged_budget = False
        capacity.bounded_by_active_tail = True

        restored = pickle.loads(pickle.dumps(capacity))

        self.assertTrue(restored.reservable)
        self.assertEqual(restored.explicit_block_num, 23)
        self.assertFalse(restored.charge_to_paged_budget)
        self.assertTrue(restored.bounded_by_active_tail)

    def test_legacy_three_tuple_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_LegacyCacheCapacityPolicyDesc()))

        self.assertTrue(restored.reservable)
        self.assertEqual(restored.explicit_block_num, 17)
        self.assertFalse(restored.charge_to_paged_budget)
        self.assertIsNone(restored.bounded_by_active_tail)

    def test_unsupported_tuple_size_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeError, "Invalid CacheCapacityPolicyDesc state"
        ):
            capacity = _new_cache_capacity_policy_desc()
            capacity.__setstate__((True,))


class KVCacheSpecDescPickleTest(unittest.TestCase):
    def test_current_format_round_trip(self):
        desc = KVCacheSpecDesc()
        desc.tag = "full"
        desc.cache_type = KVCacheSpecType.MHA
        desc.kv_head_num = 2
        desc.size_per_head = 512
        desc.entry_elems = 31
        desc.explicit_entry_count = 37
        desc.block_stride_alignment_min_entries = 41
        desc.group_type = CacheGroupType.FULL
        reuse = CacheReusePolicyDesc()
        reuse.enable_prefix_reuse = True
        desc.reuse = reuse
        capacity = CacheCapacityPolicyDesc()
        capacity.bounded_by_active_tail = True
        desc.capacity = capacity

        self.assertEqual(len(desc.__getstate__()), 22)
        restored = pickle.loads(pickle.dumps(desc))

        self.assertEqual(restored.tag, "full")
        self.assertEqual(restored.cache_type, KVCacheSpecType.MHA)
        self.assertEqual(restored.kv_head_num, 2)
        self.assertEqual(restored.size_per_head, 512)
        self.assertEqual(restored.entry_elems, 31)
        self.assertEqual(restored.explicit_entry_count, 37)
        self.assertEqual(restored.block_stride_alignment_min_entries, 41)
        self.assertEqual(restored.group_type, CacheGroupType.FULL)
        self.assertTrue(restored.reuse.enable_prefix_reuse)
        self.assertTrue(restored.capacity.bounded_by_active_tail)

    def test_legacy_twenty_tuple_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_LegacyKVCacheSpecDesc()))

        self.assertEqual(restored.tag, "legacy")
        self.assertEqual(restored.cache_type, KVCacheSpecType.MHA)
        self.assertIsNone(restored.kv_head_num)
        self.assertIsNone(restored.size_per_head)
        self.assertEqual(restored.entry_elems, 17)
        self.assertEqual(restored.explicit_entry_count, 23)
        self.assertEqual(restored.block_stride_alignment_min_entries, 29)
        self.assertEqual(restored.group_type, CacheGroupType.FULL)
        self.assertTrue(restored.reuse.enable_prefix_reuse)

    def test_unsupported_tuple_size_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid KVCacheSpecDesc state"):
            desc = _new_kv_cache_spec_desc()
            desc.__setstate__(tuple(range(21)))


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
        config.generation_prefill_cuda_graph_max_requests = 5
        config.generation_prefill_capture_token_buckets = [32, 64, 96]
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
        self.assertEqual(restored.generation_prefill_cuda_graph_max_requests, 5)
        self.assertEqual(
            restored.generation_prefill_capture_token_buckets, [32, 64, 96]
        )
        self.assertTrue(restored.enable_native_cuda_graph)
        self.assertEqual(restored.num_native_cuda_graph, 41)
        self.assertEqual(restored.prefill_capture_seq_lens, [17, 23])
        self.assertEqual(restored.decode_capture_batch_sizes, [2, 7])
        self.assertTrue(restored.disable_dpc_random)
        self.assertTrue(restored.rocm_disable_custom_ag)

    def test_legacy_14_tuple_uses_generation_prefill_cuda_graph_defaults(self):
        restored = pickle.loads(pickle.dumps(_LegacyHWKernelConfig()))

        self.assertEqual(restored.generation_prefill_cuda_graph_max_requests, 1)
        self.assertEqual(restored.prefill_capture_seq_lens, [64, 128])
        self.assertEqual(restored.decode_capture_batch_sizes, [1, 8])
        self.assertEqual(restored.num_native_cuda_graph, 37)
        self.assertEqual(
            restored.generation_prefill_capture_token_buckets,
            HWKernelConfig().generation_prefill_capture_token_buckets,
        )

    def test_unsupported_tuple_sizes_are_rejected(self):
        for size in (15, 17, 18):
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
            "not-an-integer",
            [32, 64],
        )
        with self.assertRaisesRegex(RuntimeError, "HWKernelConfig unpickle error"):
            config = _new_hw_kernel_config()
            config.__setstate__(malformed_state)


class RuntimeConfigPickleTest(unittest.TestCase):
    def test_output_dispatcher_worker_count_round_trip(self):
        config = RuntimeConfig()
        config.output_dispatcher_worker_count = 3

        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.output_dispatcher_worker_count, 3)

    def test_previous_formats_default_to_serial_dispatch(self):
        config = RuntimeConfig()
        config.output_dispatcher_worker_count = 3
        config.model_warm_up = False
        state = config.__getstate__()
        for size in (12, 13):
            with self.subTest(size=size):
                restored = RuntimeConfig.__new__(RuntimeConfig)
                restored.__setstate__(state[:size])
                self.assertEqual(restored.output_dispatcher_worker_count, 0)
                self.assertEqual(restored.model_warm_up, size == 12)


if __name__ == "__main__":
    unittest.main()

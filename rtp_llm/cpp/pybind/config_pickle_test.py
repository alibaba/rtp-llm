import pickle
import unittest

from rtp_llm.ops import (
    CacheCapacityPolicyDesc,
    CacheCpPolicyDesc,
    CacheTailPolicyDesc,
    CpBlockMappingMode,
    CpBlockSliceMode,
    CpPrefillSliceLayout,
    GrammarConfig,
    KVCacheConfig,
    KVCacheSpecDesc,
    KVCacheSpecType,
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


class CacheConfigPickleTest(unittest.TestCase):
    def test_kv_cache_config_round_trip_preserves_dsv4_adjacent_fields(self):
        config = KVCacheConfig()
        config.load_cache_retry_times = 17
        config.dsv4_fixed_pool_blocks = 101
        config.dsv4_hca_state_pool_blocks = 203

        restored = pickle.loads(pickle.dumps(config))

        self.assertIs(type(restored), KVCacheConfig)
        self.assertEqual(restored.load_cache_retry_times, 17)
        self.assertEqual(restored.dsv4_fixed_pool_blocks, 101)
        self.assertEqual(restored.dsv4_hca_state_pool_blocks, 203)
        with self.assertRaises(AttributeError):
            _ = restored.dsv4_fixed_pool_use_memory

    def test_capacity_policy_round_trip_preserves_current_fields(self):
        capacity = CacheCapacityPolicyDesc()
        capacity.reservable = False
        capacity.explicit_block_num = 307

        restored = pickle.loads(pickle.dumps(capacity))

        self.assertIs(type(restored), CacheCapacityPolicyDesc)
        self.assertIs(restored.reservable, False)
        self.assertEqual(restored.explicit_block_num, 307)

    def test_kv_cache_spec_round_trip_preserves_adjacent_policy_fields(self):
        capacity = CacheCapacityPolicyDesc()
        capacity.reservable = True
        capacity.explicit_block_num = 409

        tail = CacheTailPolicyDesc()
        tail.active_tail_blocks = 3
        tail.validate_tail_blocks = False

        cp = CacheCpPolicyDesc()
        cp.mapping = CpBlockMappingMode.COMPACT_LAST_RANK
        cp.slice = CpBlockSliceMode.PAYLOAD_BYTES
        cp.scale_seq_size = True
        cp.align_payload = False
        cp.prefill_slice_layout = CpPrefillSliceLayout.BLOCK_STRIDE

        desc = KVCacheSpecDesc()
        desc.tag = "pickle-policy"
        desc.cache_type = KVCacheSpecType.OPAQUE_STATE
        desc.capacity = capacity
        desc.tail = tail
        desc.cp = cp

        restored = pickle.loads(pickle.dumps(desc))

        self.assertIs(type(restored), KVCacheSpecDesc)
        self.assertEqual(restored.tag, "pickle-policy")
        self.assertEqual(restored.cache_type, KVCacheSpecType.OPAQUE_STATE)
        self.assertIs(type(restored.capacity), CacheCapacityPolicyDesc)
        self.assertIs(restored.capacity.reservable, True)
        self.assertEqual(restored.capacity.explicit_block_num, 409)
        self.assertIs(type(restored.tail), CacheTailPolicyDesc)
        self.assertEqual(restored.tail.active_tail_blocks, 3)
        self.assertIs(restored.tail.validate_tail_blocks, False)
        self.assertIs(type(restored.cp), CacheCpPolicyDesc)
        self.assertEqual(restored.cp.mapping, CpBlockMappingMode.COMPACT_LAST_RANK)
        self.assertEqual(restored.cp.slice, CpBlockSliceMode.PAYLOAD_BYTES)
        self.assertIs(restored.cp.scale_seq_size, True)
        self.assertIs(restored.cp.align_payload, False)
        self.assertEqual(
            restored.cp.prefill_slice_layout, CpPrefillSliceLayout.BLOCK_STRIDE
        )


if __name__ == "__main__":
    unittest.main()

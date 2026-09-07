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


def _new_kv_cache_config():
    return KVCacheConfig.__new__(KVCacheConfig)


def _new_cache_capacity_policy_desc():
    return CacheCapacityPolicyDesc.__new__(CacheCapacityPolicyDesc)


def _new_cache_cp_policy_desc():
    return CacheCpPolicyDesc.__new__(CacheCpPolicyDesc)


def _new_kv_cache_spec_desc():
    return KVCacheSpecDesc.__new__(KVCacheSpecDesc)


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


class _PreviousKVCacheConfig:
    def __reduce__(self):
        config = KVCacheConfig()
        config.load_cache_retry_times = 17
        config.dsv4_fixed_pool_blocks = 101
        config.dsv4_hca_state_pool_blocks = 203
        previous_state = config.__getstate__() + (True,)
        return _new_kv_cache_config, (), previous_state


class _PreviousCacheCapacityPolicyDesc:
    def __reduce__(self):
        # Previous layout: reservable, explicit_block_num, charge_to_paged_budget.
        return _new_cache_capacity_policy_desc, (), (False, 509, True)


class _PreviousCacheCpPolicyDesc:
    def __reduce__(self):
        # Previous layout: mapping, slice, scale_seq_size, align_payload,
        # prefill_slice_layout.
        return (
            _new_cache_cp_policy_desc,
            (),
            (
                CpBlockMappingMode.COMPACT_LAST_RANK,
                CpBlockSliceMode.PAYLOAD_BYTES,
                True,
                False,
                CpPrefillSliceLayout.BLOCK_STRIDE,
            ),
        )


class _PreviousKVCacheSpecDesc:
    def __init__(self, memory=None):
        self.memory = memory

    def __reduce__(self):
        desc = KVCacheSpecDesc()
        desc.tag = "previous-pickle-policy"
        desc.cache_type = KVCacheSpecType.OPAQUE_STATE
        tail = CacheTailPolicyDesc()
        tail.active_tail_blocks = 7
        tail.validate_tail_blocks = True
        desc.tail = tail

        # Drop the newly restored alignment suffix to reproduce the preceding
        # 19-item layout, then insert the removed memory policy field.
        current_state = list(desc.__getstate__())[:-1]
        current_state[16] = _PreviousCacheCapacityPolicyDesc()
        current_state[18] = _PreviousCacheCpPolicyDesc()
        # Previous 20-item layout inserted memory between capacity and tail.
        previous_state = tuple(current_state[:17] + [self.memory] + current_state[17:])
        return _new_kv_cache_spec_desc, (), previous_state


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
    def test_kv_cache_sequence_block_size_round_trip(self):
        default_config = pickle.loads(pickle.dumps(KVCacheConfig()))
        self.assertEqual(default_config.seq_size_per_block, 0)

        explicit_config = KVCacheConfig()
        explicit_config.seq_size_per_block = 64
        restored = pickle.loads(pickle.dumps(explicit_config))
        self.assertEqual(restored.seq_size_per_block, 64)

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

    def test_previous_57_item_format_is_loaded(self):
        restored = pickle.loads(pickle.dumps(_PreviousKVCacheConfig()))

        self.assertIs(type(restored), KVCacheConfig)
        self.assertEqual(restored.load_cache_retry_times, 17)
        self.assertEqual(restored.dsv4_fixed_pool_blocks, 101)
        self.assertEqual(restored.dsv4_hca_state_pool_blocks, 203)
        with self.assertRaises(AttributeError):
            _ = restored.dsv4_fixed_pool_use_memory

    def test_unknown_kv_cache_config_layout_is_rejected(self):
        config = KVCacheConfig()
        for state in (
            config.__getstate__()[:-1],
            config.__getstate__() + (True, False),
        ):
            with (
                self.subTest(size=len(state)),
                self.assertRaisesRegex(RuntimeError, "Invalid state"),
            ):
                candidate = _new_kv_cache_config()
                candidate.__setstate__(state)

    def test_capacity_policy_round_trip_preserves_current_fields(self):
        capacity = CacheCapacityPolicyDesc()
        capacity.reservable = False
        capacity.explicit_block_num = 307

        restored = pickle.loads(pickle.dumps(capacity))

        self.assertIs(type(restored), CacheCapacityPolicyDesc)
        self.assertIs(restored.reservable, False)
        self.assertEqual(restored.explicit_block_num, 307)

    def test_previous_capacity_policy_format_is_loaded(self):
        restored = pickle.loads(
            pickle.dumps(_PreviousCacheCapacityPolicyDesc(), protocol=4)
        )

        self.assertIs(type(restored), CacheCapacityPolicyDesc)
        self.assertIs(restored.reservable, False)
        self.assertEqual(restored.explicit_block_num, 509)

    def test_previous_cp_policy_format_uses_shifted_field_indices(self):
        restored = pickle.loads(pickle.dumps(_PreviousCacheCpPolicyDesc(), protocol=4))

        self.assertIs(type(restored), CacheCpPolicyDesc)
        self.assertEqual(restored.mapping, CpBlockMappingMode.COMPACT_LAST_RANK)
        self.assertEqual(restored.slice, CpBlockSliceMode.PAYLOAD_BYTES)
        self.assertIs(restored.align_payload, False)
        self.assertEqual(
            restored.prefill_slice_layout, CpPrefillSliceLayout.BLOCK_STRIDE
        )

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
        cp.align_payload = False
        cp.prefill_slice_layout = CpPrefillSliceLayout.BLOCK_STRIDE

        desc = KVCacheSpecDesc()
        desc.tag = "pickle-policy"
        desc.cache_type = KVCacheSpecType.OPAQUE_STATE
        desc.kernel_tokens_per_block_alignment = 128
        desc.capacity = capacity
        desc.tail = tail
        desc.cp = cp

        restored = pickle.loads(pickle.dumps(desc))

        self.assertIs(type(restored), KVCacheSpecDesc)
        self.assertEqual(restored.tag, "pickle-policy")
        self.assertEqual(restored.cache_type, KVCacheSpecType.OPAQUE_STATE)
        self.assertEqual(restored.kernel_tokens_per_block_alignment, 128)
        self.assertIs(type(restored.capacity), CacheCapacityPolicyDesc)
        self.assertIs(restored.capacity.reservable, True)
        self.assertEqual(restored.capacity.explicit_block_num, 409)
        self.assertIs(type(restored.tail), CacheTailPolicyDesc)
        self.assertEqual(restored.tail.active_tail_blocks, 3)
        self.assertIs(restored.tail.validate_tail_blocks, False)
        self.assertIs(type(restored.cp), CacheCpPolicyDesc)
        self.assertEqual(restored.cp.mapping, CpBlockMappingMode.COMPACT_LAST_RANK)
        self.assertEqual(restored.cp.slice, CpBlockSliceMode.PAYLOAD_BYTES)
        self.assertIs(restored.cp.align_payload, False)
        self.assertEqual(
            restored.cp.prefill_slice_layout, CpPrefillSliceLayout.BLOCK_STRIDE
        )

    def test_current_policy_writers_keep_compact_layouts(self):
        self.assertEqual(len(CacheCapacityPolicyDesc().__getstate__()), 2)
        self.assertEqual(len(CacheCpPolicyDesc().__getstate__()), 4)
        self.assertEqual(len(KVCacheSpecDesc().__getstate__()), 20)

    def test_previous_kv_cache_spec_without_alignment_is_loaded(self):
        state = KVCacheSpecDesc().__getstate__()[:-1]
        restored = _new_kv_cache_spec_desc()
        restored.__setstate__(state)

        self.assertEqual(restored.kernel_tokens_per_block_alignment, 1)

    def test_previous_kv_cache_spec_with_empty_memory_is_loaded(self):
        restored = pickle.loads(
            pickle.dumps(_PreviousKVCacheSpecDesc(memory=None), protocol=4)
        )

        self.assertIs(type(restored), KVCacheSpecDesc)
        self.assertEqual(restored.tag, "previous-pickle-policy")
        self.assertEqual(restored.cache_type, KVCacheSpecType.OPAQUE_STATE)
        self.assertIs(type(restored.capacity), CacheCapacityPolicyDesc)
        self.assertIs(restored.capacity.reservable, False)
        self.assertEqual(restored.capacity.explicit_block_num, 509)
        self.assertIs(type(restored.tail), CacheTailPolicyDesc)
        self.assertEqual(restored.tail.active_tail_blocks, 7)
        self.assertIs(restored.tail.validate_tail_blocks, True)
        self.assertIs(type(restored.cp), CacheCpPolicyDesc)
        self.assertEqual(restored.cp.mapping, CpBlockMappingMode.COMPACT_LAST_RANK)
        self.assertIs(restored.cp.align_payload, False)

    def test_previous_kv_cache_spec_with_memory_policy_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeError, "convert the pickle offline with the previous RTP-LLM version"
        ):
            pickle.loads(
                pickle.dumps(
                    _PreviousKVCacheSpecDesc(memory="legacy-memory-policy"),
                    protocol=4,
                )
            )

    def test_unknown_policy_layouts_are_rejected(self):
        invalid_cases = (
            (CacheCapacityPolicyDesc, (True,)),
            (CacheCapacityPolicyDesc, (True, 1, False, None)),
            (CacheCpPolicyDesc, (None, None, None)),
            (CacheCpPolicyDesc, (None, None, None, None, None, None)),
            (KVCacheSpecDesc, KVCacheSpecDesc().__getstate__()[:-2]),
            (KVCacheSpecDesc, KVCacheSpecDesc().__getstate__() + (None, None)),
        )
        for config_type, state in invalid_cases:
            with (
                self.subTest(config_type=config_type, size=len(state)),
                self.assertRaisesRegex(RuntimeError, "Invalid .* state"),
            ):
                candidate = config_type.__new__(config_type)
                candidate.__setstate__(state)


if __name__ == "__main__":
    unittest.main()

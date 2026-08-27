import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_sibling_module(name: str, relative_path: str):
    module_path = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


grouping = _load_sibling_module("glm53_indexer_grouping", "indexer_grouping.py")
compressor = _load_sibling_module("glm53_indexer_compressor", "indexer_compressor.py")
hadamard = _load_sibling_module("glm53_hadamard", "../base/cuda/hadamard.py")


class IndexerGroupingTest(unittest.TestCase):
    def test_legacy_geometry_preserves_per_token_topk(self):
        config = SimpleNamespace(
            indexer_topk=2048,
            indexer_compress_ratio=1,
            sparse_attention_topk=0,
        )
        actual = grouping.IndexerGroupingGeometry.from_attention_config(config)
        self.assertEqual(actual.selection_topk, 2048)
        self.assertEqual(actual.group_size, 1)
        self.assertEqual(actual.attention_topk, 2048)

    def test_glm53_geometry_splits_history_groups_and_tail(self):
        config = SimpleNamespace(
            indexer_topk=512,
            indexer_compress_ratio=4,
            sparse_attention_topk=2051,
        )
        actual = grouping.IndexerGroupingGeometry.from_attention_config(config)
        self.assertEqual(actual.selection_topk, 512)
        self.assertEqual(actual.group_size, 4)
        self.assertEqual(actual.attention_topk, 2051)
        self.assertEqual(actual.tail_size, 3)

    def test_glm53_geometry_derives_tail_width(self):
        config = SimpleNamespace(
            indexer_topk=512,
            indexer_compress_ratio=4,
            sparse_attention_topk=0,
        )
        actual = grouping.IndexerGroupingGeometry.from_attention_config(config)
        self.assertEqual(actual.attention_topk, 2051)

    def test_geometry_rejects_mismatched_attention_width(self):
        config = SimpleNamespace(
            indexer_topk=512,
            indexer_compress_ratio=4,
            sparse_attention_topk=2048,
        )
        with self.assertRaisesRegex(ValueError, "incomplete tail"):
            grouping.IndexerGroupingGeometry.from_attention_config(config)

    def test_expand_preserves_score_then_lane_order_and_invalids(self):
        group_ids = torch.tensor([[2, 0, -1]], dtype=torch.int32)
        actual = grouping.expand_indexer_group_indices(group_ids, 4)
        expected = torch.tensor(
            [[8, 9, 10, 11, 0, 1, 2, 3, -1, -1, -1, -1]],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)

    def test_expand_masks_partial_tail_with_raw_length(self):
        group_ids = torch.tensor([[2]], dtype=torch.int64)
        lengths = torch.tensor([10], dtype=torch.int64)
        actual = grouping.expand_indexer_group_indices(
            group_ids, 4, raw_sequence_lengths=lengths
        )
        torch.testing.assert_close(
            actual, torch.tensor([[8, 9, -1, -1]], dtype=torch.int64)
        )

    def test_append_incomplete_tail_uses_fixed_three_lane_width(self):
        history = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        actual = grouping.append_incomplete_tail_indices(
            history, torch.tensor([6], dtype=torch.int32), 4
        )
        torch.testing.assert_close(
            actual, torch.tensor([[0, 1, 2, 3, 4, 5, -1]], dtype=torch.int32)
        )

    def test_append_tail_covers_all_remainders_without_future_tokens(self):
        history = torch.full((4, 4), -1, dtype=torch.int32)
        lengths = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
        actual = grouping.append_incomplete_tail_indices(history, lengths, 4)
        expected_tail = torch.tensor(
            [
                [-1, -1, -1],
                [4, -1, -1],
                [4, 5, -1],
                [4, 5, 6],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual[:, -3:], expected_tail)

    def test_expand_rejects_invalid_dtype_and_length_shape(self):
        with self.assertRaisesRegex(TypeError, "int32 or int64"):
            grouping.expand_indexer_group_indices(torch.zeros(1, 2), 4)
        with self.assertRaisesRegex(ValueError, "shape must match"):
            grouping.expand_indexer_group_indices(
                torch.zeros(2, 2, dtype=torch.int32),
                4,
                raw_sequence_lengths=torch.ones(1, dtype=torch.int32),
            )


class IndexerCompressorTest(unittest.TestCase):
    def test_state_ring_includes_current_group_and_mtp_slack(self):
        self.assertEqual(compressor.compressor_state_ring_entries(4, 0, 0), 4)
        self.assertEqual(compressor.compressor_state_ring_entries(4, 0, 3), 8)

    def test_cache_layout_matches_fp8_ratio4_contract(self):
        layout = compressor.IndexerCompressorCacheLayout()
        self.assertEqual(layout.kv_entry_bytes, 132)
        self.assertEqual(layout.state_width, 256)
        self.assertEqual(layout.state_ring_entries, 4)
        self.assertEqual(layout.entries_per_kernel_block(128), 32)

    def test_fp8_pool_view_flattens_framework_mla_storage_without_copy(self):
        base = torch.arange(2 * 64 * 33, dtype=torch.int64).to(torch.uint8)
        base = base.view(2, 64, 33)
        actual, entries_per_block = compressor.fp8_pool_view(base, 132)
        self.assertEqual(tuple(actual.shape), (2, 16, 132))
        self.assertEqual(entries_per_block, 16)
        self.assertEqual(actual.data_ptr(), base.data_ptr())

    def test_state_pool_view_flattens_framework_mla_storage_without_copy(self):
        base = torch.arange(2 * 64 * 16, dtype=torch.float32).view(2, 64, 16)
        actual, entries_per_block = compressor.fp32_state_pool_view(base, 256)
        self.assertEqual(tuple(actual.shape), (8, 256))
        self.assertEqual(entries_per_block, 4)
        self.assertEqual(actual.data_ptr(), base.data_ptr())

    def test_pool_view_rejects_partial_entries(self):
        base = torch.empty(2, 64, 32, dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "exact multiple"):
            compressor.fp8_pool_view(base, 132)

    def test_pool_view_rejects_noncontiguous_storage(self):
        base = torch.empty(2, 132, 4, dtype=torch.uint8).transpose(1, 2)
        with self.assertRaisesRegex(RuntimeError, "must be contiguous"):
            compressor.fp8_pool_view(base, 132)

    def test_state_ring_rejects_invalid_geometry(self):
        with self.assertRaisesRegex(ValueError, "compress_ratio must be positive"):
            compressor.compressor_state_ring_entries(0, 0)
        with self.assertRaisesRegex(ValueError, "overlap must be 0 or 1"):
            compressor.compressor_state_ring_entries(4, 2)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            compressor.compressor_state_ring_entries(4, 0, -1)

    def test_torch_hadamard_fallback_matches_reference(self):
        original_fast_path = hadamard._fast_hadamard
        hadamard._fast_hadamard = None
        try:
            base = torch.arange(2 * 128, dtype=torch.float32).view(2, 128)
            actual = hadamard.normalized_hadamard_transform(base)
        finally:
            hadamard._fast_hadamard = original_fast_path
        expected = compressor._hadamard_rotate_reference(base)
        torch.testing.assert_close(actual, expected)

    def test_reference_compressor_pools_non_overlapping_groups(self):
        kv = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 2)
        score = torch.zeros_like(kv)
        ape = torch.zeros(4, 2)

        actual, boundaries = compressor.compress_indexer_projection_reference(
            kv, score, ape, compress_ratio=4, overlap=0
        )
        torch.testing.assert_close(boundaries, torch.tensor([3, 7]))
        pooled = kv[:4].mean(dim=0)
        expected_one = torch.tensor(
            [(pooled[0] + pooled[1]) / 2**0.5, (pooled[0] - pooled[1]) / 2**0.5]
        )
        expected = torch.stack([expected_one, expected_one])
        torch.testing.assert_close(actual, expected)

    def test_reference_emits_only_complete_groups(self):
        kv = torch.zeros(3, 128)
        score = torch.zeros_like(kv)
        ape = torch.zeros(4, 128)
        actual, boundaries = compressor.compress_indexer_projection_reference(
            kv, score, ape
        )
        self.assertEqual(tuple(actual.shape), (0, 128))
        self.assertEqual(boundaries.numel(), 0)

    def test_reference_rejects_non_power_of_two_head_dim(self):
        kv = torch.zeros(4, 3)
        with self.assertRaisesRegex(ValueError, "power of two"):
            compressor.compress_indexer_projection_reference(
                kv, torch.zeros_like(kv), torch.zeros(4, 3)
            )


if __name__ == "__main__":
    unittest.main()

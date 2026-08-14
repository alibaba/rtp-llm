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


grouping = _load_sibling_module("glm54_indexer_grouping", "indexer_grouping.py")
compressor = _load_sibling_module(
    "glm54_indexer_compressor", "indexer_compressor.py"
)


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

    def test_glm54_geometry_splits_selection_and_attention_topk(self):
        config = SimpleNamespace(
            indexer_topk=512,
            indexer_compress_ratio=4,
            sparse_attention_topk=2048,
        )
        actual = grouping.IndexerGroupingGeometry.from_attention_config(config)
        self.assertEqual(actual.selection_topk, 512)
        self.assertEqual(actual.group_size, 4)
        self.assertEqual(actual.attention_topk, 2048)

    def test_geometry_rejects_mismatched_attention_width(self):
        config = SimpleNamespace(
            indexer_topk=512,
            indexer_compress_ratio=4,
            sparse_attention_topk=512,
        )
        with self.assertRaisesRegex(ValueError, r"selection_topk \* group_size"):
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


class IndexerCompressorTest(unittest.TestCase):
    def test_state_ring_includes_overlap_and_mtp_slack(self):
        self.assertEqual(compressor.compressor_state_ring_entries(4, 1, 0), 8)
        self.assertEqual(compressor.compressor_state_ring_entries(4, 1, 3), 12)

    def test_cache_layout_matches_fp8_ratio4_contract(self):
        layout = compressor.IndexerCompressorCacheLayout()
        self.assertEqual(layout.kv_entry_bytes, 132)
        self.assertEqual(layout.state_width, 512)
        self.assertEqual(layout.state_ring_entries, 8)
        self.assertEqual(layout.entries_per_kernel_block(128), 32)

    def test_reference_compressor_uses_previous_and_current_projection_branches(self):
        # ratio=4, overlap=1, head_dim=2 => projection width=4.
        # First half is the previous-group branch; second half is current.
        kv = torch.zeros(8, 4, dtype=torch.float32)
        kv[:, :2] = torch.tensor(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 2
        )
        kv[:, 2:] = torch.tensor(
            [[10.0, 20.0], [20.0, 30.0], [30.0, 40.0], [40.0, 50.0]] * 2
        )
        score = torch.zeros_like(kv)
        ape = torch.zeros(4, 4)
        norm = torch.ones(2)

        actual, boundaries = compressor.compress_indexer_projection_reference(
            kv, score, ape, norm, compress_ratio=4, overlap=1, norm_eps=1e-6
        )
        torch.testing.assert_close(boundaries, torch.tensor([3, 7]))

        first_raw = kv[:4, 2:].mean(dim=0)
        second_raw = torch.cat([kv[:4, :2], kv[4:8, 2:]], dim=0).mean(dim=0)

        def rms_norm(value):
            return value * torch.rsqrt(value.square().mean() + 1e-6)

        expected = torch.stack([rms_norm(first_raw), rms_norm(second_raw)])
        torch.testing.assert_close(actual, expected)

    def test_reference_emits_only_complete_groups(self):
        kv = torch.zeros(3, 256)
        score = torch.zeros_like(kv)
        ape = torch.zeros(4, 256)
        norm = torch.ones(128)
        actual, boundaries = compressor.compress_indexer_projection_reference(
            kv, score, ape, norm
        )
        self.assertEqual(tuple(actual.shape), (0, 128))
        self.assertEqual(boundaries.numel(), 0)


if __name__ == "__main__":
    unittest.main()

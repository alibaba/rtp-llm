import math
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _sparse_prefill_fast_path_limit,
)
from rtp_llm.models_py.modules.hybrid.indexer_compressor import (
    IndexerCompressorCacheLayout,
    compress_indexer_projection_reference,
    compressor_state_ring_entries,
    fp32_state_pool_view,
    fp8_pool_view,
)
from rtp_llm.models_py.modules.hybrid.indexer_grouping import (
    IndexerGroupingGeometry,
    append_incomplete_tail_indices,
    expand_indexer_group_indices,
)


class Glm53IndexerGroupingTest(unittest.TestCase):
    def test_prefill_fast_path_uses_expanded_raw_token_width(self) -> None:
        compressed = SimpleNamespace(indexer_topk=512, sparse_attention_topk=2051)
        legacy = SimpleNamespace(indexer_topk=2048, sparse_attention_topk=0)
        self.assertEqual(_sparse_prefill_fast_path_limit(compressed), 2051)
        self.assertEqual(_sparse_prefill_fast_path_limit(legacy), 2048)

    def test_geometry_uses_compressed_selection_space(self) -> None:
        geometry = IndexerGroupingGeometry.from_attention_config(
            SimpleNamespace(
                indexer_topk=512,
                indexer_compress_ratio=4,
                sparse_attention_topk=2051,
            )
        )
        self.assertEqual(geometry.selection_topk, 512)
        self.assertEqual(geometry.group_size, 4)
        self.assertEqual(geometry.attention_topk, 2051)
        self.assertEqual(geometry.tail_size, 3)

    def test_invalid_attention_width_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserve expanded complete groups"):
            IndexerGroupingGeometry(512, 4, 2048).validate()

    def test_group_expansion_rejects_incomplete_pool_id(self) -> None:
        pooled = torch.tensor([[0, 1, 2, -1]], dtype=torch.int32)
        raw_lengths = torch.tensor([10], dtype=torch.int32)
        expanded = expand_indexer_group_indices(
            pooled, 4, raw_sequence_lengths=raw_lengths
        )
        self.assertEqual(
            expanded.tolist(),
            [[0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1]],
        )
        with_tail = append_incomplete_tail_indices(expanded, raw_lengths, 4)
        self.assertEqual(with_tail[0, -3:].tolist(), [8, 9, -1])

    def test_short_sequence_selects_only_raw_tail(self) -> None:
        pooled = torch.tensor([[0, -1]], dtype=torch.int64)
        raw_lengths = torch.tensor([3], dtype=torch.int64)
        expanded = expand_indexer_group_indices(
            pooled, 4, raw_sequence_lengths=raw_lengths
        )
        self.assertTrue(torch.all(expanded == -1))
        with_tail = append_incomplete_tail_indices(expanded, raw_lengths, 4)
        self.assertEqual(with_tail[0, -3:].tolist(), [0, 1, 2])


class Glm53IndexerCacheLayoutTest(unittest.TestCase):
    def test_fp8_pool_view_preserves_block_boundaries(self) -> None:
        base = torch.empty((2, 32 * 132), dtype=torch.uint8)
        view, entries = fp8_pool_view(base, 132)
        self.assertEqual(entries, 32)
        self.assertEqual(tuple(view.shape), (2, 32, 132))
        self.assertEqual(view.data_ptr(), base.data_ptr())

    def test_fp8_pool_view_preserves_planar_bytes_without_repacking(self) -> None:
        base = torch.zeros((1, 2 * 132), dtype=torch.uint8)
        # Physical ABI: two 128-byte K rows followed by two float32 scales.
        base[0, :128].fill_(11)
        base[0, 128:256].fill_(22)
        base[0, 256:260].copy_(
            torch.tensor([1.5], dtype=torch.float32).view(torch.uint8)
        )
        base[0, 260:264].copy_(
            torch.tensor([2.5], dtype=torch.float32).view(torch.uint8)
        )

        view, entries = fp8_pool_view(base, 132)

        self.assertEqual(entries, 2)
        self.assertEqual(view.data_ptr(), base.data_ptr())
        raw = view.view(1, -1)
        self.assertTrue(torch.all(raw[0, :128] == 11))
        self.assertTrue(torch.all(raw[0, 128:256] == 22))
        self.assertEqual(raw[0, 256:260].view(torch.float32).item(), 1.5)
        self.assertEqual(raw[0, 260:264].view(torch.float32).item(), 2.5)

    def test_fp32_state_pool_view_preserves_width_and_ring(self) -> None:
        base = torch.empty((2, 4, 256), dtype=torch.float32)
        view, entries = fp32_state_pool_view(base, 256)
        self.assertEqual(entries, 4)
        self.assertEqual(tuple(view.shape), (8, 256))
        self.assertEqual(view.data_ptr(), base.data_ptr())

    def test_layout_dimensions_cover_mtp_slack(self) -> None:
        layout = IndexerCompressorCacheLayout(gen_num_per_cycle=3)
        self.assertEqual(layout.kv_entry_bytes, 132)
        self.assertEqual(layout.state_width, 256)
        self.assertEqual(layout.state_ring_entries, 8)
        self.assertEqual(layout.entries_per_kernel_block(128), 32)
        self.assertEqual(compressor_state_ring_entries(4, 0, 0), 4)

    def test_misaligned_pool_stride_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not an exact multiple"):
            fp8_pool_view(torch.empty((2, 133), dtype=torch.uint8), 132)


class Glm53KPoolReferenceTest(unittest.TestCase):
    def test_four_token_softmax_pool_and_hadamard(self) -> None:
        key = torch.stack([torch.full((128,), float(value)) for value in (1, 2, 3, 4)])
        score = torch.zeros_like(key)
        ape = torch.zeros((4, 128), dtype=key.dtype)
        compressed, boundaries = compress_indexer_projection_reference(key, score, ape)
        self.assertEqual(boundaries.tolist(), [3])
        self.assertEqual(tuple(compressed.shape), (1, 128))
        expected_dc = torch.tensor(2.5 * math.sqrt(128)).to(torch.bfloat16).item()
        self.assertEqual(compressed[0, 0].item(), expected_dc)
        torch.testing.assert_close(compressed[0, 1:], torch.zeros(127))

    def test_partial_group_is_not_compressed(self) -> None:
        key = torch.randn(3, 128)
        compressed, boundaries = compress_indexer_projection_reference(
            key, torch.zeros_like(key), torch.zeros(4, 128)
        )
        self.assertEqual(tuple(compressed.shape), (0, 128))
        self.assertEqual(boundaries.numel(), 0)


if __name__ == "__main__":
    unittest.main()

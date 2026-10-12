"""V4.1 causal index selection and checkpoint-rounded score equations."""

import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models.test.dsv41_cache_codec_test import (
    codec,
    load_codec_source,
    make_cache,
    reference_rows,
)

with patch.dict(sys.modules, {"rtp_llm.models_py.modules.dsv4.v41_cache": codec}):
    indexer = load_codec_source("_dsv41_indexer_test", "v41_indexer.py")


def sources(layer):
    return V41LayerSources(
        layer,
        2 if layer < 20 else 1,
        layer if layer < 20 else 20,
        layer,
        None if layer < 20 else 20,
    )


def fixture(ratio):
    torch.manual_seed(415)
    cache, _ = make_cache("indexer_kv", gap=128)
    table = torch.tensor([[2, 0, 1]] * 5, dtype=torch.int32)
    if ratio == 1:
        table = table[:, :2].contiguous()
    q = torch.randn(5, 32, 128).bfloat16()
    weights = torch.randn(5, 32).bfloat16()
    positions = torch.tensor(
        [0, ratio - 1, 129 * ratio - 1, 300 * ratio - 1, -999], dtype=torch.int64
    )
    valid = torch.tensor([True, True, True, True, False])
    # Upper halves of ratio-2 indexer pages are deliberately not initialized.
    keys = torch.randn(300, 128).bfloat16()
    slots = torch.tensor(
        [
            int(table[0, i // (256 // ratio)]) * 256 + i % (256 // ratio)
            for i in range(300)
        ]
    )
    cache.write(keys, slots, backend="eager")
    q[-1] = float("nan")
    weights[-1] = float("nan")
    table[-1] = -99
    return q, weights, cache, table, positions, valid


class V41IndexerTest(unittest.TestCase):
    def test_scores_match_scalar_quantization_and_reference_bf16_equations(self):
        for layer in (2, 20):
            ratio = sources(layer).compress_ratio
            q, weights, cache, pages, positions, valid = fixture(ratio)
            op = indexer.V41Indexer(sources(layer), max_positions=300, backend="eager")
            actual, visible = op.score(q, weights, cache, pages, positions, valid)
            expected = torch.full((5, 300), -torch.inf)
            for row in range(4):
                query = codec.dequantize_rows_eager(
                    reference_rows(q[row], "indexer_kv"), "indexer_kv"
                )
                count = (int(positions[row]) + 1) // ratio
                slots = torch.tensor(
                    [
                        int(pages[row, i // (256 // ratio)]) * 256 + i % (256 // ratio)
                        for i in range(count)
                    ],
                    dtype=torch.int64,
                )
                keys = cache.read(slots, backend="eager")
                score = torch.einsum("hd,nd->hn", query, keys)
                expected[row, :count] = (
                    (score.relu() * (weights[row] / 64)[:, None]).sum(0).float()
                )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(visible.tolist(), [1 if ratio == 1 else 0, 1, 129, 300, 0])
            result = op(q, weights, cache, pages, positions, valid_rows=valid)
            for row in range(5):
                count = int(visible[row])
                self.assertEqual(result.topk[row, :count].tolist(), list(range(count)))
                self.assertTrue(bool((result.topk[row, count:] == -1).all()))

    def test_candidates_pin_newest_partial_block_and_filter_later_sources(self):
        width = 17009
        scores = torch.arange(width, dtype=torch.float32)[None, :].repeat(2, 1)
        scores[0, -1] = -100
        scores[1] = -torch.inf
        visible = torch.tensor([width, 0])
        first = indexer._select(scores, visible, candidate_source=True, candidates=None)
        self.assertEqual(first.candidate_counts.tolist(), [2048, 0])
        blocks = first.candidate_blocks[0]
        self.assertIn((width - 1) // 8, blocks.tolist())
        self.assertEqual(blocks.unique().numel(), 2048)
        block_count = (width + 7) // 8
        self.assertEqual(blocks.tolist(), list(range(block_count - 2048, block_count)))
        self.assertTrue(bool((first.candidate_blocks[1] == -1).all()))
        self.assertTrue(bool((first.topk[1] == -1).all()))
        later_scores = -scores
        later_scores[1] = -torch.inf
        later = indexer._select(
            later_scores,
            visible,
            candidate_source=False,
            candidates=first.candidate_blocks,
        )
        selected = later.topk[0].long()
        self.assertTrue(bool(torch.isin(selected // 8, blocks).all()))
        allowed = torch.isin(torch.arange(width) // 8, blocks)
        cutoff = later_scores[0, allowed].topk(512).values.min()
        self.assertTrue(bool((later_scores[0, selected] >= cutoff).all()))
        self.assertTrue(bool((selected[1:] > selected[:-1]).all()))

    def test_candidate_padding_does_not_clear_block_zero(self):
        scores = torch.arange(9).float()[None, :]
        candidates = torch.full((1, 2048), -1, dtype=torch.int32)
        candidates[0, :2] = torch.tensor([0, 1])
        result = indexer._select(
            scores, torch.tensor([9]), candidate_source=False, candidates=candidates
        )
        self.assertEqual(result.topk[0, :9].tolist(), list(range(9)))
        self.assertTrue(bool((result.topk[0, 9:] == -1).all()))

    def test_later_source_requires_complete_causal_candidates(self):
        q, weights, cache, pages, positions, valid = fixture(1)
        producer = indexer.V41Indexer(sources(20), max_positions=300, backend="eager")
        later = indexer.V41Indexer(sources(24), max_positions=300, backend="eager")
        first = producer(q, weights, cache, pages, positions, valid_rows=valid)
        second = later(
            q,
            weights,
            cache,
            pages,
            positions,
            valid_rows=valid,
            candidates=first.candidate_blocks,
        )
        torch.testing.assert_close(first.topk, second.topk, atol=0, rtol=0)
        self.assertIsNone(second.candidate_blocks)
        with self.assertRaisesRegex(ValueError, "candidate input"):
            later(q, weights, cache, pages, positions, valid_rows=valid)
        with self.assertRaisesRegex(ValueError, "candidate input"):
            producer(
                q,
                weights,
                cache,
                pages,
                positions,
                valid_rows=valid,
                candidates=first.candidate_blocks,
            )
        bad = first.candidate_blocks.clone()
        bad[3, 0] = bad[3, 1]
        with self.assertRaisesRegex(ValueError, "coverage"):
            later(q, weights, cache, pages, positions, valid_rows=valid, candidates=bad)

    def test_shape_capacity_and_live_value_rejections(self):
        for source, width in (
            (sources(2), 16385),
            (sources(20), 32769),
            (sources(21), 128),
        ):
            with self.assertRaises(ValueError):
                indexer.V41Indexer(source, max_positions=width)
        q, weights, cache, pages, positions, valid = fixture(2)
        op = indexer.V41Indexer(sources(2), max_positions=300, backend="eager")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            op.score(q, weights.float(), cache, pages, positions, valid)
        pages[3, 2] = -1
        with self.assertRaisesRegex(ValueError, "allocated"):
            op.score(q, weights, cache, pages, positions, valid)
        pages[3, 2] = 1
        positions[3] = 600
        with self.assertRaisesRegex(ValueError, "range"):
            op.score(q, weights, cache, pages, positions, valid)
        positions[3] = 599
        weights[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            op.score(q, weights, cache, pages, positions, valid)

    def test_zero_rows_keep_fixed_frame_shapes(self):
        q, weights, cache, pages, positions, valid = fixture(1)
        op = indexer.V41Indexer(sources(20), max_positions=300, backend="eager")
        result = op(
            q[:0], weights[:0], cache, pages[:0], positions[:0], valid_rows=valid[:0]
        )
        self.assertEqual(result.topk.shape, (0, 512))
        self.assertEqual(result.candidate_blocks.shape, (0, 2048))
        self.assertEqual(result.candidate_counts.shape, (0,))


if __name__ == "__main__":
    unittest.main()

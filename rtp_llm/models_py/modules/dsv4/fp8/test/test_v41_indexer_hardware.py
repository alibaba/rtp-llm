"""SM120 FP4 index scoring, causal candidates and graph qualification."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8._v41_indexer_triton import _query_quant
from rtp_llm.models_py.modules.dsv4.v41_cache import (
    V41PagedCache,
    dequantize_rows_eager,
    quantize_rows_eager,
)
from rtp_llm.models_py.modules.dsv4.v41_indexer import V41Indexer


def sources(layer):
    return V41LayerSources(
        layer,
        2 if layer < 20 else 1,
        layer if layer < 20 else 20,
        layer,
        None if layer < 20 else 20,
    )


class V41IndexerHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "V4.1 hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(416)

    def fixture(self, ratio, width, rows=4):
        used = 256 // ratio
        count = (width + used - 1) // used
        layout = dsv41_page_layouts(1)["indexer_kv"]
        raw = torch.full(
            (count, layout.stride_bytes + 128), 173, dtype=torch.uint8, device="cuda"
        )
        cache = V41PagedCache(raw[:, : layout.stride_bytes], "indexer_kv")
        pages = torch.arange(count - 1, -1, -1, device="cuda", dtype=torch.int32)[
            None, :
        ].repeat(rows, 1)
        logical = torch.arange(count * used, device="cuda")
        slots = pages[0, logical // used].long() * 256 + logical % used
        cache.write(
            torch.randn(count * used, 128, device="cuda", dtype=torch.bfloat16), slots
        )
        q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
        positions = torch.full(
            (rows,), width * ratio - 1, device="cuda", dtype=torch.int64
        )
        positions[0] = 0
        valid = torch.ones(rows, device="cuda", dtype=torch.bool)
        valid[-1] = False
        q[-1] = float("nan")
        weights[-1] = float("nan")
        pages[-1] = -99
        positions[-1] = -999
        return q, weights, cache, pages, positions, valid

    def cpu_scores(self, op, q, weights, cache, pages, positions, valid):
        reference = V41Indexer(
            op.sources, max_positions=op.max_positions, backend="eager"
        )
        return reference.score(
            q.cpu(),
            weights.cpu(),
            V41PagedCache(cache.raw.cpu(), "indexer_kv"),
            pages.cpu(),
            positions.cpu(),
            valid.cpu(),
        )

    def check_selection(self, result, scores, visible, candidates=None):
        topk = result.topk.cpu().long()
        for row, count in enumerate(visible.tolist()):
            selected = topk[row][topk[row] >= 0]
            self.assertEqual(selected.numel(), min(512, count))
            self.assertTrue(bool((selected < count).all()))
            self.assertTrue(bool((selected[1:] > selected[:-1]).all()))
            allowed = scores[row].clone()
            if candidates is not None:
                permitted = torch.isin(
                    torch.arange(scores.shape[1]) // 8, candidates[row].cpu()
                )
                allowed[~permitted] = -torch.inf
            if selected.numel():
                threshold = allowed.topk(min(512, count)).values.min()
                self.assertTrue(bool((allowed[selected] >= threshold).all()))
            self.assertTrue(bool((topk[row, selected.numel() :] == -1).all()))
        if result.candidate_blocks is not None:
            blocks = result.candidate_blocks.cpu().long()
            counts = result.candidate_counts.cpu().tolist()
            for row, count in enumerate(visible.tolist()):
                expected = min(2048, (count + 7) // 8)
                self.assertEqual(counts[row], expected)
                active = blocks[row, :expected]
                self.assertEqual(active.unique().numel(), expected)
                if count:
                    self.assertIn((count - 1) // 8, active.tolist())
                    self.assertTrue(bool((active < (count + 7) // 8).all()))
                    block_scores = (
                        torch.nn.functional.pad(
                            scores[row], (0, -scores.shape[1] % 8), value=-torch.inf
                        )
                        .reshape(-1, 8)
                        .amax(-1)
                    )
                    block_scores[(count - 1) // 8] = torch.inf
                    cutoff = block_scores.topk(expected).values.min()
                    self.assertTrue(bool((block_scores[active] >= cutoff).all()))
                self.assertTrue(bool((blocks[row, expected:] == -1).all()))

    def test_full_prefill_row_capacity(self):
        q, weights, cache, pages, positions, valid = self.fixture(1, 32768, rows=128)
        q.normal_()
        weights.normal_()
        valid.fill_(True)
        pages[-1].copy_(pages[1])
        positions[-1] = 32767
        op = V41Indexer(sources(20), max_positions=32768)
        actual, _ = op.score(q, weights, cache, pages, positions, valid)
        sample = torch.tensor([1, 64, 127], device="cuda")
        expected, _ = self.cpu_scores(
            op,
            q[sample],
            weights[sample],
            cache,
            pages[sample],
            positions[sample],
            valid[sample],
        )
        torch.testing.assert_close(actual[sample].cpu(), expected, atol=0, rtol=0)
        result = op(q, weights, cache, pages, positions, valid_rows=valid)
        self.assertEqual(result.topk.shape, (128, 512))
        self.assertEqual(result.candidate_blocks.shape, (128, 2048))
        self.assertTrue(bool((result.candidate_counts[1:] == 2048).all()))
        self.assertEqual(result.candidate_counts[0].item(), 1)

    def test_query_fp4_rounding_and_masked_nan_rows(self):
        q = torch.randn(3, 32, 128, dtype=torch.bfloat16, device="cuda")
        q[0, :, :16] = torch.tensor(
            [
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5,
                6,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5,
                -6,
            ],
            device="cuda",
        )
        q[2] = float("nan")
        valid = torch.tensor([True, True, False], device="cuda")
        out = torch.empty_like(q)
        _query_quant[(3, 32)](q, valid, out, num_warps=4)
        expected = dequantize_rows_eager(
            quantize_rows_eager(q[:2].cpu().reshape(-1, 128), "indexer_kv"),
            "indexer_kv",
        ).reshape(2, 32, 128)
        torch.testing.assert_close(out[:2].cpu(), expected, atol=0, rtol=0)
        self.assertTrue(bool((out[2] == 0).all()))

    def test_score_and_selection_at_ratio2_pages_and_32k_capacity(self):
        for layer, width in ((2, 300), (14, 16384), (20, 32768)):
            with self.subTest(layer=layer, width=width):
                q, weights, cache, pages, positions, valid = self.fixture(
                    sources(layer).compress_ratio, width
                )
                op = V41Indexer(sources(layer), max_positions=width)
                original = cache.raw.clone()
                actual, visible = op.score(q, weights, cache, pages, positions, valid)
                expected, expected_visible = self.cpu_scores(
                    op, q, weights, cache, pages, positions, valid
                )
                torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    visible.cpu(), expected_visible, atol=0, rtol=0
                )
                result = op(q, weights, cache, pages, positions, valid_rows=valid)
                self.check_selection(result, expected, expected_visible)
                torch.testing.assert_close(cache.raw, original, atol=0, rtol=0)
                if layer == 20:
                    later = V41Indexer(sources(24), max_positions=width)
                    changed_weights = -weights
                    next_scores, _ = self.cpu_scores(
                        later, q, changed_weights, cache, pages, positions, valid
                    )
                    refined = later(
                        q,
                        changed_weights,
                        cache,
                        pages,
                        positions,
                        valid_rows=valid,
                        candidates=result.candidate_blocks,
                    )
                    self.check_selection(
                        refined, next_scores, expected_visible, result.candidate_blocks
                    )

    def test_graph_updates_queries_pages_positions_masks_and_candidates(self):
        width = 17009
        q, weights, cache, pages, positions, valid = self.fixture(1, width)
        producer = V41Indexer(sources(20), max_positions=width)
        later = V41Indexer(sources(24), max_positions=width)
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm"):
                producer(q, weights, cache, pages, positions, valid_rows=valid)
        for _ in range(2):
            first = producer(q, weights, cache, pages, positions, valid_rows=valid)
            later(
                q,
                weights,
                cache,
                pages,
                positions,
                valid_rows=valid,
                candidates=first.candidate_blocks,
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            first = producer(q, weights, cache, pages, positions, valid_rows=valid)
            second = later(
                q,
                -weights,
                cache,
                pages,
                positions,
                valid_rows=valid,
                candidates=first.candidate_blocks,
            )
        for position, live in ((width - 1, True), (0, False), (256, True)):
            q.normal_()
            weights.normal_()
            valid.fill_(live)
            positions.fill_(position)
            pages.copy_(
                torch.arange(pages.shape[1], device="cuda", dtype=torch.int32)[
                    None, :
                ].expand_as(pages)
            )
            graph.replay()
            torch.cuda.synchronize()
            expected, visible = self.cpu_scores(
                producer, q, weights, cache, pages, positions, valid
            )
            next_scores, _ = self.cpu_scores(
                later, q, -weights, cache, pages, positions, valid
            )
            self.check_selection(first, expected, visible)
            self.check_selection(second, next_scores, visible, first.candidate_blocks)


if __name__ == "__main__":
    unittest.main()

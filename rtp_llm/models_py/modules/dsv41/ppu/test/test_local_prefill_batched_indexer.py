"""Geometry and native-score equivalence for request-relative Prefill tiles."""

import os
from types import SimpleNamespace
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_prefill_indexer as sparse
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import PrefillIndexerKeys
from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_indexer as batch


class LocalPrefillIndexerLayoutTest(unittest.TestCase):
    def test_large_score_tile_reuses_the_hierarchical_selector(self):
        logits = SimpleNamespace(shape=(9007, 65536))
        output = SimpleNamespace(shape=(9007, 512))
        self.assertEqual(tuple(batch.dense_selection_tiles(logits, output, True)),
                         (slice(0, 4096), slice(4096, 8192), slice(8192, 9007)))
        self.assertEqual(tuple(batch.dense_selection_tiles(logits, output, False)), (slice(0, 9007),))
        logits.shape = (9007, 16911)
        self.assertEqual(tuple(batch.dense_selection_tiles(logits, output, True)), (slice(0, 9007),))

    def test_ragged_shapes_and_budget(self):
        spans = (slice(0, 3), slice(3, 8), slice(8, 15))
        group, = batch.score_groups(spans, (257, 511, 777), 15, 32, score_bytes=8192)
        self.assertEqual(group.offsets, (0, 512, 1024))
        self.assertEqual(group.bank_rows, 1801)
        self.assertEqual(group.width, 777)
        self.assertEqual(tuple(group.tiles()), (slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 15)))
        self.assertIsNone(batch.score_groups(spans, (257, 511, 777), 15, 32, score_bytes=4096))
        self.assertIsNone(batch.score_groups((slice(0, 3), slice(4, 8)), (10, 20), 8, 32))

    def test_large_banks_partition_without_dropping_requests(self):
        groups = batch.score_groups(
            (slice(0, 3), slice(3, 10), slice(10, 16)), (656001, 394002, 701), 16, 32,
            sparse_width=16384,
        )
        self.assertEqual([(g.first_request, g.rows) for g in groups], [(0, slice(0, 3)), (1, slice(3, 16))])
        self.assertEqual(groups[1].offsets, (0, 394240))


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU test")
class LocalPrefillIndexerDeviceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        self.spans = (slice(0, 3), slice(3, 8), slice(8, 15))
        self.counts = (257, 511, 777)
        self.q = batch._quantized_indexer_query(torch.randn(15, 32, 128, device="cuda", dtype=torch.bfloat16))
        self.weights = torch.rand((15, 32), device="cuda", dtype=torch.float32)
        self.keys = []
        for count in self.counts:
            quant = batch._quantized_indexer_query(torch.randn(count, 128, device="cuda", dtype=torch.bfloat16))
            self.keys.append(PrefillIndexerKeys(quant.values, quant.scales))
        self.visible = torch.tensor(
            (0, 17, 257, 201, 255, 256, 510, 511, 4, 127, 255, 512, 690, 776, 777),
            device="cuda", dtype=torch.int32,
        )
        group, = batch.score_groups(self.spans, self.counts, 15, 32)
        cache = SimpleNamespace(read_indexer_fp4=lambda layer, rid, global_entries:
                                (self.keys[rid].quant, self.keys[rid].scale))
        self.bank = batch._pack_keys(cache, 20, group)
        self.offsets = torch.tensor(
            [start for span, start in zip(self.spans, group.offsets) for _ in range(span.start, span.stop)],
            device="cuda", dtype=torch.int32,
        )
        self.row_counts = torch.tensor(
            [count for span, count in zip(self.spans, self.counts) for _ in range(span.start, span.stop)],
            device="cuda", dtype=torch.int32,
        )

    def _query(self, span):
        return SimpleNamespace(values=self.q.values[span], scales=self.q.scales[span])

    @torch.no_grad()
    def test_dense_scores_match_each_request_with_odd_row_boundaries(self):
        logits = batch.dense_logits(self.q, self.bank, self.weights, self.offsets, self.visible, max(self.counts))
        for span, keys in zip(self.spans, self.keys):
            visible = self.visible[span]
            expected = batch.dense_logits(
                self._query(span), keys, self.weights[span], torch.zeros_like(visible), visible, len(keys),
            )
            torch.testing.assert_close(logits[span, :len(keys)], expected, rtol=0, atol=0)
            self.assertTrue(logits[span, len(keys):].isneginf().all())

    @torch.no_grad()
    def test_sparse_scores_and_relative_remap_match_each_request(self):
        candidates = torch.tensor((2, 0, 5, 2, 14, 19, -1, 53), device="cuda", dtype=torch.int32)
        candidates = candidates.expand(15, -1).contiguous()
        plan = sparse.prepare_plan(
            candidates, self.visible, len(self.bank), ppu=True,
            row_key_offsets=self.offsets, row_key_counts=self.row_counts,
        )
        logits = sparse.score(self.q.values, self.q.scales, self.bank, self.weights, plan)
        columns = torch.arange(plan.sparse_columns, device="cuda").expand(15, -1).contiguous()
        output = torch.empty_like(columns, dtype=torch.int32)
        remapped = sparse.remap(columns, plan, logits=logits, out=output)
        self.assertIs(remapped, output)
        for span, keys in zip(self.spans, self.keys):
            reference = sparse.prepare_plan(candidates[span], self.visible[span], len(keys), ppu=True)
            q = self._query(span)
            expected = sparse.score(q.values, q.scales, keys, self.weights[span], reference)
            live = columns[span] < reference.end[:, None]
            torch.testing.assert_close(logits[span][live], expected[live], rtol=0, atol=0)
            expected_positions = sparse.remap(columns[span], reference, logits=expected)
            torch.testing.assert_close(remapped[span], expected_positions, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

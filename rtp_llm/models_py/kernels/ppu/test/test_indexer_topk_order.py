"""PPU TopK ordering with real selection kernels and changing graph inputs."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda import indexer_op


class IndexerTopkOrderTest(TestCase):
    def make_op(self):
        return indexer_op.IndexerOp(1, 128, 2048, 0, stable_topk=True)

    def assert_selection(self, actual, starts, ends, relative=False):
        actual = actual.cpu().tolist()
        for row, start, end in zip(actual, starts, ends):
            expected = list(range(max(start, end - 2048), end))
            if relative:
                expected = [index - start for index in expected]
            expected += [-1] * (2048 - len(expected))
            self.assertEqual(row, expected)

    def test_prefill_order_and_request_masks(self):
        op = self.make_op()
        self.assertTrue(op._ppu_stable_topk)
        starts = [0, 0, 128, 512]
        ends = [17, 2048, 8192, 4096]
        ks = torch.tensor(starts, dtype=torch.int32, device="cuda")
        ke = torch.tensor(ends, dtype=torch.int32, device="cuda")
        logits = torch.arange(8192, dtype=torch.float32, device="cuda").repeat(4, 1)
        output = torch.empty(4, 2048, dtype=torch.int32, device="cuda")
        for _ in range(12):
            op._run_prefill_topk(logits, ks, ke, output, 8192)
            self.assert_selection(output, starts, ends, relative=True)
        # Identical ties may select a different set from another backend, but
        # within this native backend the same set must have a repeatable order.
        logits.zero_()
        ke.fill_(8192)
        ks.zero_()
        op._run_prefill_topk(logits, ks, ke, output, 8192)
        reference = output.cpu().clone()
        self.assertTrue(torch.equal(reference, torch.arange(2048, dtype=torch.int32).repeat(4, 1)))
        for _ in range(12):
            op._run_prefill_topk(logits, ks, ke, output, 8192)
            self.assertTrue(torch.equal(reference, output.cpu()))

    def test_paged_graph_replay_changes_lengths_and_scores(self):
        op = self.make_op()
        logits = torch.arange(8192, dtype=torch.float32, device="cuda").repeat(4, 1)
        lengths = torch.full((4,), 8192, dtype=torch.int32, device="cuda")
        params = SimpleNamespace(expanded_seq_lens=lengths, kvlen_d=lengths)
        inputs = SimpleNamespace(kv_cache_kernel_block_id_device=torch.zeros(4, 128, dtype=torch.int32, device="cuda"))
        cache = SimpleNamespace(seq_size_per_block=64, kv_cache_base=torch.zeros(1, 64 * 132, dtype=torch.uint8, device="cuda"))
        q = torch.zeros(4, 1, 128, device="cuda")
        weights = torch.ones(4, 1, 1, device="cuda")
        with patch.object(indexer_op, "_fp8_paged_mqa_logits_compat", return_value=logits):
            for _ in range(3):
                op._get_topk_paged(q, weights, cache, params, inputs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = op._get_topk_paged(q, weights, cache, params, inputs)
            for ends in ([17, 2048, 8192, 4096], [4096, 1, 8192, 1024]):
                lengths.copy_(torch.tensor(ends, dtype=torch.int32, device="cuda"))
                for _ in range(6):
                    graph.replay()
                    self.assert_selection(output, [0] * 4, ends)
            logits.neg_()
            lengths.fill_(8192)
            graph.replay()
            expected = torch.arange(2048, dtype=torch.int32).repeat(4, 1)
            self.assertTrue(torch.equal(output.cpu(), expected))

    def test_canonicalization_is_opt_in_on_ppu(self):
        for ppu in (False, True):
            for stable in (False, True):
                with patch.object(indexer_op, "is_ppu", return_value=ppu):
                    op = indexer_op.IndexerOp(1, 128, 2048, 0, stable_topk=stable)
                    self.assertEqual(op._ppu_stable_topk, ppu and stable)

    def test_partial_ties_signed_scores_and_large_row_count(self):
        op = self.make_op()
        scores = torch.zeros(1030, 8192, device="cuda")
        scores[:, :2000] = 1
        starts = torch.zeros(1030, dtype=torch.int32, device="cuda")
        ends = torch.full((1030,), 8192, dtype=torch.int32, device="cuda")
        output = torch.empty(1030, 2048, dtype=torch.int32, device="cuda")
        expected = torch.arange(2048, dtype=torch.int32).repeat(1030, 1)
        for _ in range(4):
            op._run_prefill_topk(scores, starts, ends, output, 8192)
            self.assertTrue(torch.equal(output.cpu(), expected))
        scores = torch.linspace(-1, 1, 8192, device="cuda").repeat(4, 1)
        scores[1] = torch.arange(8192, device="cuda") % 255 - 128
        scores[2].zero_()
        scores[2, ::2] = -0.0
        scores[3] = torch.arange(8192, dtype=torch.int32).view(torch.float32).to(scores.device)
        starts, ends, output = starts[:4], ends[:4], output[:4]
        op._run_prefill_topk(scores, starts, ends, output, 8192)
        reference = torch.argsort(scores.cpu(), descending=True, stable=True)[:, :2048].sort(dim=-1).values.int()
        self.assertTrue(torch.equal(output.cpu(), reference))

    def test_short_and_empty_rows_keep_minus_one_padding(self):
        op = self.make_op()
        for columns in (0, 17):
            scores = torch.zeros(3, columns, device="cuda")
            starts = torch.zeros(3, dtype=torch.int32, device="cuda")
            ends = torch.tensor([0, min(1, columns), columns], dtype=torch.int32, device="cuda")
            output = torch.empty(3, 2048, dtype=torch.int32, device="cuda")
            op._run_prefill_topk(scores, starts, ends, output, columns)
            self.assert_selection(output, [0] * 3, ends.cpu().tolist())


if __name__ == "__main__":
    main()

"""Bounded SM90 UVA async lookup, including changed-input graph replay.

Reuses only the existing small-table fixture (132 KiB registered host table).
No checkpoint/shared production table is created, duplicated, or registered.
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.engram import HostEngramEmbedding, NgramHashState
from rtp_llm.models_py.modules.dsv41.sm90_engram import UvaEngramPrefetch
from rtp_llm.models_py.modules.dsv41.test import test_sm90_engram as fixture_module


class Sm90EngramUvaPrefetchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_module.Sm90EngramTest.setUpClass()

    def setUp(self):
        fixture = fixture_module.Sm90EngramTest()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.fixture = fixture
        self.device = fixture.device
        self.model = fixture.model
        self.embedding = self.model.embed_tokens
        self.hasher = NgramHashState(
            fixture.layout, token_map=[*range(12), 31], device=self.device
        )
        self.prefetch = UvaEngramPrefetch(self.device, (0, 1))
        self.addCleanup(self.prefetch.finish)

    def _lookups(self, hashes):
        return ((0, self.embedding, hashes), (1, self.embedding, hashes))

    def _run_prefetched(self, hidden, windows):
        # Hashing and masks are inside the captured body, so replay must use
        # the latest complete token history rather than a cached first request.
        hashes = self.hasher.hash_token_windows(windows, dead_mask=windows == 12)
        ids = hashes[:, 0].contiguous()
        mask = (windows[:, 0] >= 0) & (windows[:, 0] != 12)
        try:
            self.prefetch.begin(self._lookups(ids))
            output = hidden
            for layer_id in (0, 1):
                output = self.model(
                    output, ids, mask, embedding_rows=self.prefetch.wait(layer_id)
                )
            return output
        finally:
            self.prefetch.finish()

    def _run_eager(self, hidden, windows):
        hashes = self.hasher.hash_token_windows(windows, dead_mask=windows == 12)
        ids = hashes[:, 0].contiguous()
        mask = (windows[:, 0] >= 0) & (windows[:, 0] != 12)
        output = hidden
        for _ in (0, 1):
            output = self.model(output, ids, mask)
        return output

    @torch.inference_mode()
    def test_eager_and_graph_changed_history_mask_and_bucket_are_bitwise_equal(self):
        for capacity in (1, 2, 4):
            windows = torch.zeros(capacity, 4, dtype=torch.int32, device=self.device)
            hidden = torch.randn(capacity, 4, 5120, device=self.device).bfloat16()
            original = hidden.clone()
            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    hidden.copy_(original)
                    self._run_prefetched(hidden, windows)
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                captured = self._run_prefetched(hidden, windows)
            for case, live in enumerate((capacity, 1, 0, capacity)):
                fresh = (
                    torch.arange(capacity * 4).reshape(capacity, 4) + case * 3
                ) % 12
                fresh[live:] = -1
                if live:
                    fresh[0, 2] = -1  # Truncate an ngram at a request boundary.
                if live and case == 3:
                    fresh[0, 0] = 12  # Change which rows are masked.
                windows.copy_(fresh)
                original.normal_()
                hidden.copy_(original)
                graph.replay()
                torch.cuda.synchronize(self.device)
                result = captured.clone()
                expected = self._run_eager(original.clone(), windows)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                actual_eager = self._run_prefetched(original.clone(), windows)
                torch.testing.assert_close(actual_eager, expected, rtol=0, atol=0)
                invalid = (windows[:, 0] < 0) | (windows[:, 0] == 12)
                torch.testing.assert_close(
                    result[invalid], original[invalid], rtol=0, atol=0
                )
                self.assertEqual(self.prefetch.rows, {})
                self.assertIsNone(self.prefetch._origin)

    @torch.inference_mode()
    def test_prefetch_matches_cpu_rows_and_explicit_out(self):
        cpu = HostEngramEmbedding(self.fixture.table, self.fixture.table_scales)
        for count in (0, 1, 7):
            ids = torch.randint(0, 4000, (count, 12), device=self.device)
            if count:
                ids[0, :3] = torch.tensor([-1, 4000, 4007], device=self.device)
            expected = cpu(ids.cpu(), "cpu")
            self.prefetch.begin(self._lookups(ids))
            try:
                for layer_id in (0, 1):
                    rows = self.prefetch.wait(layer_id)
                    torch.testing.assert_close(rows.cpu(), expected, rtol=0, atol=0)
            finally:
                self.prefetch.finish()
        with self.assertRaisesRegex(RuntimeError, "registered layer tables"):
            self.prefetch.begin(((0, cpu, ids),))
        self.assertIsNone(self.prefetch._origin)

    @torch.inference_mode()
    def test_failure_after_first_lookup_joins_stream_and_allows_next_request(self):
        ids = torch.zeros(1, 12, dtype=torch.int64, device=self.device)

        class FailingLookup:
            _uva = self.embedding._uva
            weight = self.embedding.weight

            def __call__(self, *args, **kwargs):
                raise RuntimeError("injected lookup failure")

        with self.assertRaisesRegex(RuntimeError, "injected lookup failure"):
            self.prefetch.begin(((0, self.embedding, ids), (1, FailingLookup(), ids)))
        self.assertEqual(self.prefetch.rows, {})
        self.assertIsNone(self.prefetch._origin)
        self.prefetch.begin(self._lookups(ids))
        # Simulate a consumer failing before waiting on either layer.
        self.prefetch.finish()
        torch.cuda.synchronize(self.device)
        self.assertEqual(self.prefetch.rows, {})


if __name__ == "__main__":
    unittest.main()

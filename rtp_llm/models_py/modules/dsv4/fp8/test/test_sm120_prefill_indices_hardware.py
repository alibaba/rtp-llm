"""CUDA component test for the generic SM120 prefill index normalization.

Exact behavior of the generic ``canonical_topk`` + ``clamp_min_(0)`` chain on
real CUDA tensors, checked against an INDEPENDENT Python list oracle —
padded width, live-prefix filtering, zero tails and clamped counts —
including non-default stream execution, inference-mode tensors, allocator
churn between two forwards with identical shapes but different contents, and
graph replay preserving the caller-supplied input lengths.
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
    SM120_EXTRA_TOPK_WIDTHS,
    canonical_topk,
)

SWA_SUPPORTED = (128, 512, 1024, 2048)
WINDOW = 128


def producer_table(rows, width, lens, device, seed):
    g = torch.Generator().manual_seed(seed)
    table = torch.zeros((rows, width), dtype=torch.int32, device=device)
    for r in range(rows):
        n = int(lens[r])
        if n > 0:
            table[r, :n] = torch.randint(0, 4096, (n,), generator=g).to(
                device=device, dtype=torch.int32
            )
    return table


def normalize_actual(indices, lens, supported):
    """The production consumer chain: canonical_topk then clamp_min_(0)."""
    i, n = canonical_topk(indices, lens, supported)
    return i.clamp_min_(0), n


def expected_rows(host_table, host_lens, supported):
    """Independent list oracle: no call into the kernels under test."""
    e_indices, e_lens = [], []
    for row, length in zip(host_table, host_lens):
        width = len(row)
        padded = next(w for w in supported if w >= width)
        limit = min(max(length, 0), padded)
        live = [v for j, v in enumerate(row) if j < limit and v >= 0]
        e_indices.append(live + [0] * (padded - len(live)))
        e_lens.append(len(live))
    return e_indices, e_lens


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ReadyChunkCudaTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")
        torch.cuda.init()

    def check(self, rows, width, lens, supported, seed):
        table = producer_table(rows, width, lens, self.device, seed)
        lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
        e_i, e_l = expected_rows(table.cpu().tolist(), lens, supported)
        got_i, got_l = normalize_actual(table.clone(), lens_t.clone(), supported)
        self.assertTrue(
            torch.equal(
                got_i,
                torch.tensor(e_i, dtype=torch.int32, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                got_l,
                torch.tensor(e_l, dtype=torch.int32, device=self.device),
            )
        )
        self.assertEqual(got_i.device.type, "cuda")

    def test_widths_on_cuda(self):
        for width, supported in (
            (WINDOW, SWA_SUPPORTED),
            (64, SM120_EXTRA_TOPK_WIDTHS),
            (192, SM120_EXTRA_TOPK_WIDTHS),
            (256, SM120_EXTRA_TOPK_WIDTHS),
            (2048, SM120_EXTRA_TOPK_WIDTHS),
        ):
            for rows in (1, 1024):
                lens = [min(2 * r + 1, width) for r in range(rows)]
                self.check(rows, width, lens, supported, seed=rows + width)

        # Holes inside the limit are filtered; -1 entries never survive.
        table = torch.tensor([[7, -1, 9, 99]], dtype=torch.int32, device=self.device)
        table_copy = table.clone()
        lens_t = torch.tensor([3], dtype=torch.int32, device=self.device)
        lens_copy = lens_t.clone()
        got_i, got_l = normalize_actual(table, lens_t, SWA_SUPPORTED)
        self.assertEqual(got_i[0].tolist()[:3], [7, 9, 0])
        self.assertEqual(got_l.tolist(), [2])
        self.assertTrue(torch.equal(table, table_copy))
        self.assertTrue(torch.equal(lens_t, lens_copy))

        # Negative and oversized lengths clamp into the padded width.
        for length, e_live in ((-3, []), (10**6, [7, 9, 99])):
            got_i, got_l = normalize_actual(
                table,
                torch.tensor([length], dtype=torch.int32, device=self.device),
                SWA_SUPPORTED,
            )
            self.assertEqual(got_i[0].tolist()[: len(e_live) + 1], e_live + [0])
            self.assertEqual(got_l.tolist(), [len(e_live)])

        # Widths beyond every supported bucket keep the existing rejection.
        with self.assertRaises(RuntimeError):
            normalize_actual(
                torch.zeros((1, 8193), dtype=torch.int32, device=self.device),
                torch.ones(1, dtype=torch.int32, device=self.device),
                SWA_SUPPORTED,
            )

    def test_non_default_stream_and_events(self):
        rows, width = 256, WINDOW
        lens = [min(r + 1, width) for r in range(rows)]
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            table = producer_table(rows, width, lens, self.device, 11)
            lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
            got_i, _ = normalize_actual(table, lens_t, SWA_SUPPORTED)
            ev = torch.cuda.Event()
            ev.record(stream)
        torch.cuda.current_stream(self.device).wait_event(ev)
        e_i, _ = expected_rows(table.cpu().tolist(), lens, SWA_SUPPORTED)
        self.assertTrue(
            torch.equal(got_i, torch.tensor(e_i, dtype=torch.int32, device=self.device))
        )

    def test_two_forwards_same_shape_different_contents(self):
        # Allocator churn: forward A's storage must never leak into forward B.
        rows, width = 1024, WINDOW
        lens = [WINDOW] * rows
        for seed in (101, 202):
            table = producer_table(rows, width, lens, self.device, seed)
            lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
            e_i, _ = expected_rows(table.cpu().tolist(), lens, SWA_SUPPORTED)
            got_i, _ = normalize_actual(table.clone(), lens_t.clone(), SWA_SUPPORTED)
            self.assertTrue(
                torch.equal(
                    got_i,
                    torch.tensor(e_i, dtype=torch.int32, device=self.device),
                )
            )
            del table, lens_t, got_i, e_i

    def test_inference_mode(self):
        rows, width = 64, WINDOW
        lens = [min(r + 3, width) for r in range(rows)]
        with torch.inference_mode():
            table = producer_table(rows, width, lens, self.device, 5)
            lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
            got_i, _ = normalize_actual(table, lens_t, SWA_SUPPORTED)
        e_i, _ = expected_rows(table.cpu().tolist(), lens, SWA_SUPPORTED)
        self.assertTrue(
            torch.equal(got_i, torch.tensor(e_i, dtype=torch.int32, device=self.device))
        )

    def test_graph_replay_preserves_input_lengths(self):
        table = torch.arange(WINDOW, dtype=torch.int32, device=self.device).repeat(2, 1)
        lengths = torch.tensor([-1, WINDOW + 5], dtype=torch.int32, device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            normalize_actual(table, lengths, SWA_SUPPORTED)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            norm_i, norm_l = normalize_actual(table, lengths, SWA_SUPPORTED)
        # Capture only; the expected values are computed on the host.
        for values, e_counts in (([-1, WINDOW + 5], [0, 128]), ([17, -3], [17, 0])):
            lengths.copy_(torch.tensor(values, dtype=torch.int32, device=self.device))
            graph.replay()
            self.assertEqual(lengths.tolist(), values)
            self.assertEqual(norm_l.tolist(), e_counts)
            for row, count in enumerate(e_counts):
                live = norm_i[row].tolist()
                self.assertEqual(live[:count], list(range(WINDOW))[:count])
                self.assertTrue(all(v == 0 for v in live[count:]))


if __name__ == "__main__":
    unittest.main()

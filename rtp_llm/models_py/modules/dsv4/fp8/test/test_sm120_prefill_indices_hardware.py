"""G1 CUDA component test for the SM120 prefill index-adapter ready path.

Exact bit-equality of `_ready_chunk` against the incumbent
`canonical_topk` + `clamp_min_(0)` chain on real CUDA tensors, including the
in-place lens-clamp side effect, non-default stream execution, inference-mode
tensors, and allocator churn between two forwards with identical shapes but
different contents (the record carries no cross-forward reuse).
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import _ready_chunk
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


def reference(indices, lens, supported):
    out_i, out_l = canonical_topk(indices, lens, supported)
    out_i.clamp_min_(0)
    return out_i, out_l


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ReadyChunkCudaTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")
        torch.cuda.init()

    def check(self, rows, width, lens, supported, seed):
        table = producer_table(rows, width, lens, self.device, seed)
        lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
        ref_i, ref_l = reference(table.clone(), lens_t.clone(), supported)
        got = _ready_chunk(table, lens_t, 0, rows, supported)
        self.assertIsNotNone(got)
        new_i, new_l = got
        self.assertTrue(torch.equal(ref_i, new_i))
        self.assertTrue(torch.equal(ref_l, new_l))
        self.assertEqual(new_i.device.type, "cuda")

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

    def test_non_default_stream_and_events(self):
        rows, width = 256, WINDOW
        lens = [min(r + 1, width) for r in range(rows)]
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            table = producer_table(rows, width, lens, self.device, 11)
            lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
            got = _ready_chunk(table, lens_t, 0, rows, SWA_SUPPORTED)
            self.assertIsNotNone(got)
            new_i, _ = got
            ev = torch.cuda.Event()
            ev.record(stream)
        torch.cuda.current_stream(self.device).wait_event(ev)
        ref_i, _ = reference(
            table.clone(),
            torch.tensor(lens, dtype=torch.int32, device=self.device),
            SWA_SUPPORTED,
        )
        self.assertTrue(torch.equal(ref_i, new_i))

    def test_two_forwards_same_shape_different_contents(self):
        # Allocator churn: forward A's storage must never leak into forward B.
        rows, width = 1024, WINDOW
        lens = [WINDOW] * rows
        for seed in (101, 202):
            table = producer_table(rows, width, lens, self.device, seed)
            lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
            ref_i, _ = reference(table.clone(), lens_t.clone(), SWA_SUPPORTED)
            got = _ready_chunk(table, lens_t, 0, rows, SWA_SUPPORTED)
            self.assertTrue(torch.equal(ref_i, got[0]))
            del table, lens_t, got, ref_i

    def test_inference_mode(self):
        rows, width = 64, WINDOW
        lens = [min(r + 3, width) for r in range(rows)]
        with torch.inference_mode():
            table = producer_table(rows, width, lens, self.device, 5)
            lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
            got = _ready_chunk(table, lens_t, 0, rows, SWA_SUPPORTED)
            self.assertIsNotNone(got)
            ref_i, _ = reference(
                table.clone(),
                torch.tensor(lens, dtype=torch.int32, device=self.device),
                SWA_SUPPORTED,
            )
            self.assertTrue(torch.equal(ref_i, got[0]))


if __name__ == "__main__":
    unittest.main()

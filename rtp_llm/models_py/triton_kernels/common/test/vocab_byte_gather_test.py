import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.fp8_vocab_gather import (
    dequant_hyper,
    raw_rows,
)
from rtp_llm.models_py.triton_kernels.common.test.vocab_gather_fixture import (
    VocabByteGatherTestFixture,
)

MODULE = "rtp_llm.models_py.modules.base.common.fp8_embedding"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class VocabByteGatherTest(VocabByteGatherTestFixture, unittest.TestCase):

    def test_rank_scales_masks_invalid_tokens_and_bf16_rounding(self):
        for dtype in (torch.int32, torch.int64):
            ids = torch.tensor(
                [-1, 0, 256, 0, 257, 0, 17, 0, 3, 0, 8, 0], device="cuda", dtype=dtype
            )[::2]
            mask = torch.tensor(
                [1, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0, 0], device="cuda", dtype=torch.int32
            )[::2]
            for scales in (
                [0.119713, 0.27351],
                [0.0, 1.0],
                [float("inf"), float("nan")],
            ):
                self.scales.copy_(torch.tensor(scales, device="cuda"))
                for selected_mask in (None, mask):
                    for branches in (1, 4):
                        actual = self.bytes_consumer(ids, selected_mask, branches)
                        expected = self.reference(ids, selected_mask, branches)
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.uint8), expected.view(torch.uint8)
                            )
                        )
        self.assertEqual(self.bytes_consumer(ids[:0]).shape, (0, 10240))

    def test_live_graph_reads_ids_masks_bytes_and_all_rank_scales(self):
        ids = torch.randint(0, 257, (1024,), device="cuda", dtype=torch.int32)
        mask = torch.ones_like(ids, dtype=torch.bool)
        for _ in range(3):
            self.bytes_consumer(ids, mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.bytes_consumer(ids, mask)
        for gain in (0.01, 1.0, 3.0):
            ids.random_(0, 257)
            mask.copy_(ids % 3 != 0)
            self.weights[0].fill_(gain)
            self.weights[1].fill_(gain * 2)
            self.scales.copy_(torch.tensor([gain * 0.17, gain * 0.31], device="cuda"))
            graph.replay()
            self.assertTrue(
                torch.equal(
                    actual.view(torch.uint8),
                    self.reference(ids, mask).view(torch.uint8),
                )
            )
        graph.reset()

    def test_invalid_layouts_fail_before_kernel(self):
        ids = torch.arange(4, device="cuda", dtype=torch.int32)
        for bad in (ids.cpu(), ids.float(), ids.reshape(2, 2)):
            with self.assertRaises(ValueError):
                raw_rows(bad, self.weights[0])
        gathered = torch.cat([raw_rows(ids, w) for w in self.weights])
        for bad in (self.scales.cpu(), self.scales[:1], self.scales.to(torch.bfloat16)):
            with self.assertRaises(ValueError):
                dequant_hyper(gathered, bad, 4, 1280, 2, 4)
        with self.assertRaises(ValueError):
            dequant_hyper(gathered, self.scales, 4, 1280, 2, 4, ids[:1])


if __name__ == "__main__":
    unittest.main()

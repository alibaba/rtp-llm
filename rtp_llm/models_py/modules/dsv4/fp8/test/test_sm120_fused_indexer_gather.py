"""The SM120 gather fusion must preserve every FP32 score bit."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import quantize_indexer_k
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    _fp8_paged_indexer_score_sm120,
)


def make_inputs(width, batch, count, block):
    blocks = (width + block - 1) // block
    pool = torch.zeros(
        (batch * blocks + 1, block, 132), device="cuda", dtype=torch.uint8
    )
    keys = torch.randn(batch * blocks * block, 128, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(keys.shape[0], device="cuda", dtype=torch.int64) + block
    quantize_indexer_k(keys, slots, pool)
    q = torch.randn(batch, count, 64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(batch, count, 64, device="cuda", dtype=torch.bfloat16)
    q, w = indexer_q_fp8_quant_fold(q, w)
    table = torch.arange(1, batch * blocks + 1, device="cuda", dtype=torch.int32)
    table = table.view(batch, blocks)
    lengths = torch.full((batch, count), width, device="cuda", dtype=torch.int32)
    return (
        q,
        w.view(batch * count, 64),
        pool.view(-1, 132),
        table,
        lengths,
        block,
        width,
    )


def call(inputs, enabled):
    with patch.dict(os.environ, DSV4_SM120_INDEXER_FUSED_GATHER=str(int(enabled))):
        return _fp8_paged_indexer_score_sm120(*inputs)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class FusedIndexerGatherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1710)

    def test_exact_logits_across_cache_and_speculation_shapes(self):
        for block in (32, 64):
            for width in (257, 2053, 8192):
                for batch, count in ((1, 1), (1, 3), (2, 3), (2, 4)):
                    with self.subTest(
                        block=block, width=width, batch=batch, count=count
                    ):
                        inputs = make_inputs(width, batch, count, block)
                        self.assertTrue(
                            torch.equal(call(inputs, False), call(inputs, True))
                        )

    def test_invalid_blocks_and_shorter_rows(self):
        inputs = make_inputs(2053, 2, 3, 64)
        inputs[3][:, 0] = -1
        inputs[3][:, 1] = 100000
        inputs[4][0] = torch.tensor([2053, 129, 0], device="cuda")
        self.assertTrue(torch.equal(call(inputs, False), call(inputs, True)))

    def test_noncontiguous_table(self):
        inputs = list(make_inputs(257, 2, 3, 32))
        inputs[3] = inputs[3].T.contiguous().T
        self.assertTrue(torch.equal(call(inputs, False), call(inputs, True)))

    def test_zero_context_and_empty_width(self):
        inputs = list(make_inputs(257, 2, 3, 64))
        inputs[4].zero_()
        self.assertTrue(torch.equal(call(inputs, False), call(inputs, True)))
        inputs[-1] = 0
        self.assertTrue(torch.equal(call(inputs, False), call(inputs, True)))

    def test_cuda_graph_live_metadata(self):
        inputs = make_inputs(2053, 2, 3, 64)
        for _ in range(3):
            call(inputs, True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = call(inputs, True)
        inputs[4][0, -1] = 0
        inputs[3][1, 0] = -1
        graph.replay()
        self.assertTrue(torch.equal(actual, call(inputs, False)))


if __name__ == "__main__":
    unittest.main()

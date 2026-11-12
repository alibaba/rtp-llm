import unittest

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.base.cuda.select_topk import SelectTopk
from rtp_llm.models_py.triton_kernels.moe.qwen4_topk import _moe_topk


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MoeTopkTritonTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(461)
        config = ModelConfig()
        config.expert_num = 512
        config.moe_k = 10
        config.has_moe_norm = True
        self.native = SelectTopk(config)
        self.fused = lambda logits, ids, weights: _moe_topk[(logits.shape[0],)](
            logits,
            ids,
            weights,
            STRIDE=logits.stride(0),
            num_warps=1,
            enable_fp_fusion=False,
        )

    def compare(self, logits, ids, weights):
        expected_ids = torch.empty_like(ids)
        expected_weights = torch.empty_like(weights)
        self.native(logits, expected_ids, expected_weights)
        torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(weights, expected_weights, atol=2e-7, rtol=2e-6)

    def test_native_parity_for_ties_and_small_and_large_batches(self):
        for rows in (1, 7, 17, 32, 127, 4093):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(rows=rows, dtype=dtype):
                    logits = torch.randn(rows, 512, device="cuda", dtype=dtype)
                    logits[0] = 0
                    ids = torch.empty(rows, 10, device="cuda", dtype=torch.int32)
                    weights = torch.empty(rows, 10, device="cuda")
                    self.fused(logits, ids, weights)
                    self.compare(logits, ids, weights)

    def test_graph_replay_refreshes_strided_logits(self):
        logits = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)[:, :512]
        ids = torch.empty(17, 10, device="cuda", dtype=torch.int32)
        weights = torch.empty(17, 10, device="cuda")
        self.fused(logits, ids, weights)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.fused(logits, ids, weights)
        for _ in range(3):
            logits.copy_(torch.randn_like(logits))
            logits[0] = 0
            graph.replay()
            self.compare(logits, ids, weights)

    def test_selection_preserves_softmax_rounding_ties_and_underflow(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                logits = torch.zeros(2, 512, device="cuda", dtype=dtype)
                logits[0, 511] = 1e-10
                logits[1] = torch.linspace(
                    -2000, -1500, 512, device="cuda", dtype=dtype
                )
                logits[1, 511] = 0
                ids = torch.empty(2, 10, device="cuda", dtype=torch.int32)
                weights = torch.empty(2, 10, device="cuda")
                self.fused(logits, ids, weights)
                self.compare(logits, ids, weights)
                self.assertEqual(ids[0].tolist(), list(range(10)))
                self.assertEqual(ids[1].tolist(), [511, *range(9)])

    def test_nonfinite_logits_preserve_native_routing_and_bounded_ids(self):
        logits = torch.randn(7, 512, device="cuda")
        logits[0] = float("nan")
        logits[1, 0] = float("nan")
        logits[2, 0] = float("inf")
        logits[3] = float("-inf")
        ids = torch.empty(7, 10, device="cuda", dtype=torch.int32)
        weights = torch.empty(7, 10, device="cuda")
        self.fused(logits, ids, weights)
        expected_ids = torch.empty_like(ids)
        expected_weights = torch.empty_like(weights)
        self.native(logits, expected_ids, expected_weights)
        torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(
            weights, expected_weights, atol=2e-7, rtol=2e-6, equal_nan=True
        )
        self.assertTrue(((ids >= 0) & (ids < 512)).all())


if __name__ == "__main__":
    unittest.main()

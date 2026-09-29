"""Analytical routing oracles, padding masks, and changed-input Graph replay."""
import unittest
import torch
from rtp_llm.models_py.modules.kimi_k3.routing import grouped_topk

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KimiK3NativeRoutingTest(unittest.TestCase):

    def run_op(self, scores, bias, renormalize=True, scale=1.0):
        return grouped_topk(scores, bias, top_k=16, groups=1, top_groups=1, renormalize=renormalize, scale=scale)

    def test_exact_ties_and_normalization(self):
        bias = torch.zeros(896, device='cuda', dtype=torch.float32)
        for rows in (1, 7, 629, 1025, 8192):
            scores = torch.zeros((rows, 896), device='cuda', dtype=torch.float32)
            weights, ids = self.run_op(scores, bias)
            self.assertTrue(torch.equal(ids, torch.arange(16, device='cuda', dtype=torch.int32).expand(rows, -1)))
            self.assertTrue(torch.equal(weights, torch.full_like(weights, 1 / 16)))
            weights, _ = self.run_op(scores, bias, renormalize=False, scale=2.5)
            self.assertTrue(torch.equal(weights, torch.full_like(weights, 1.25)))

    def test_graph_bias_and_padding_changes(self):
        scores = torch.zeros((9, 896), device='cuda', dtype=torch.float32)
        bias = torch.zeros(896, device='cuda', dtype=torch.float32)
        valid = torch.ones(9, device='cuda', dtype=torch.bool)

        def call():
            weights, ids = self.run_op(scores, bias)
            return (torch.where(valid[:, None], weights, 0), torch.where(valid[:, None], ids, 0))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            weights, ids = call()
        for high, real in ((895, 1), (12, 7), (895, 9), (0, 3)):
            bias.zero_()
            bias[high] = 1
            valid.copy_(torch.arange(9, device='cuda') < real)
            graph.replay()
            torch.cuda.synchronize()
            expected = [high] + [x for x in range(896) if x != high][:15]
            oracle = torch.tensor(expected, device='cuda', dtype=torch.int32).expand(real, -1)
            self.assertTrue(torch.equal(ids[:real], oracle))
            self.assertTrue(torch.equal(weights[:real], torch.full_like(weights[:real], 1 / 16)))
            self.assertEqual(int(torch.count_nonzero(ids[real:])), 0)
            self.assertEqual(int(torch.count_nonzero(weights[real:])), 0)

    def test_empty_and_reject_wrong_dtype(self):
        bias = torch.zeros(896, device='cuda', dtype=torch.float32)
        weights, ids = self.run_op(torch.empty((0, 896), device='cuda'), bias)
        self.assertEqual(weights.shape, (0, 16))
        self.assertEqual(ids.shape, (0, 16))
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(ids.dtype, torch.int32)
        with self.assertRaisesRegex(RuntimeError, 'FP32'):
            self.run_op(torch.zeros((1, 896), device='cuda', dtype=torch.bfloat16), bias)
if __name__ == '__main__':
    unittest.main(verbosity=2)

import unittest

import torch

from rtp_llm.models_py.modules.kimi_k3.norm import KimiK3LatentRMSNorm


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NativeRMSNormTest(unittest.TestCase):
    def test_analytical_output_and_changed_input_graph(self):
        weight = ((torch.arange(3584, device="cuda") % 8) * 0.125).bfloat16()
        norm = KimiK3LatentRMSNorm(weight, 3.0)
        for rows in (1, 2, 3, 7, 8, 9, 255, 256, 257):
            x = torch.ones((rows, 3584), device="cuda", dtype=torch.bfloat16)
            expected = (weight * 0.5).expand(rows, -1)
            self.assertTrue(torch.equal(norm(x), expected))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                norm(x)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = norm(x)
            for sign in (-1, 0, 1, -1):
                x.fill_(sign)
                graph.replay()
                self.assertTrue(torch.equal(y, expected * sign))

    def test_empty_and_invalid_contracts(self):
        weight = torch.ones(3584, device="cuda", dtype=torch.bfloat16)
        norm = KimiK3LatentRMSNorm(weight, 1e-5)
        empty = norm(torch.empty(0, 3584, device="cuda", dtype=torch.bfloat16))
        self.assertEqual(empty.shape, (0, 3584))
        x = torch.ones(2, 3584, device="cuda", dtype=torch.bfloat16)
        for bad in (x.float(), x[:, :-1], x.t()):
            with self.assertRaises(RuntimeError):
                norm(bad)
        with self.assertRaises(RuntimeError):
            KimiK3LatentRMSNorm(weight, -1)(x)


if __name__ == "__main__":
    unittest.main()

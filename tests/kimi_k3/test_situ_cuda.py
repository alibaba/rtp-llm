"""Run against the checkpoint formula; no production extension is needed."""

import importlib.util
from pathlib import Path
import unittest

import torch


class SituCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("This numerical regression requires a CUDA device")
        path = Path(__file__).resolve().parents[2] / (
            "rtp_llm/models_py/triton_kernels/common/situ.py"
        )
        spec = importlib.util.spec_from_file_location("k3_situ_cuda", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.situ = staticmethod(module.situ)

    @staticmethod
    def reference(g, u, beta, up_beta):
        g, u = g.float(), u.float()
        g = beta * torch.tanh(g / beta) * torch.sigmoid(g)
        if up_beta is not None:
            u = up_beta * torch.tanh(u / up_beta)
        return g * u

    def test_dense_and_split_views(self):
        torch.manual_seed(731)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for rows, width in ((1, 17), (9, 1025), (4096, 1024)):
                for packed in (False, True):
                    for up_beta in (None, 25.0):
                        with self.subTest(dtype=dtype, rows=rows, packed=packed,
                                          up_beta=up_beta):
                            g, u = (torch.randn(rows, width * 2, device="cuda",
                                                dtype=dtype) * 12).chunk(2, -1)
                            if not packed:
                                g, u = g.contiguous(), u.contiguous()
                            original_g, original_u = g.clone(), u.clone()
                            actual = self.situ(g, u, 4.0, up_beta)
                            expected = self.reference(g, u, 4.0, up_beta).to(dtype)
                            rtol = {torch.bfloat16: 0.008, torch.float16: 0.001,
                                    torch.float32: 2e-6}[dtype]
                            torch.testing.assert_close(actual, expected, rtol=rtol,
                                                       atol=2e-6)
                            torch.testing.assert_close(g, original_g, rtol=0, atol=0)
                            torch.testing.assert_close(u, original_u, rtol=0, atol=0)

    def test_graph_replay_reads_new_inputs(self):
        g = torch.randn(9, 1025, device="cuda", dtype=torch.bfloat16)
        u = torch.randn_like(g)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.situ(g, u, 4.0, 25.0)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.situ(g, u, 4.0, 25.0)
        for _ in range(3):
            g.normal_()
            u.normal_()
            graph.replay()
            torch.testing.assert_close(
                output, self.reference(g, u, 4.0, 25.0).to(g.dtype),
                rtol=0.008, atol=2e-6,
            )

    def test_no_full_size_fp32_temporary(self):
        g = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
        u = torch.randn_like(g)
        self.situ(g, u, 4.0, 25.0)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = self.situ(g, u, 4.0, 25.0)
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - before
        self.assertLessEqual(extra, result.numel() * result.element_size() + 1048576)


if __name__ == "__main__":
    unittest.main()

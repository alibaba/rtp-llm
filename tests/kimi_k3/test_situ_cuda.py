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

    def test_inplace_consumes_gate_only(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            g = torch.randn(9, 1025, device="cuda", dtype=dtype)
            u = torch.randn_like(g)
            expected = self.reference(g, u, 4.0, 25.0).to(dtype)
            original_u = u.clone()
            actual = self.situ(g, u, 4.0, 25.0, inplace=True)
            self.assertEqual(actual.data_ptr(), g.data_ptr())
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=2e-6)
            torch.testing.assert_close(u, original_u, rtol=0, atol=0)
        packed = torch.randn(9, 2050, device="cuda", dtype=torch.bfloat16)
        g, u = packed.chunk(2, -1)
        with self.assertRaisesRegex(ValueError, "owned contiguous"):
            self.situ(g, u, 4.0, 25.0, inplace=True)

    def test_empty_inplace_projection(self):
        for shape in ((0, 12), (3, 0)):
            gate = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            up = torch.empty_like(gate)
            self.assertIs(self.situ(gate, up, 4.0, 25.0, inplace=True), gate)

    def test_projection_exceeding_int32_element_offsets(self):
        # The 64K prefill regression crossed the signed 32-bit offset boundary.
        # This is intentionally a real allocation, not a smaller shape mock.
        rows, width = 65536, 32769
        n = rows * width
        self.assertGreater(n, 2**31)
        up = torch.full((rows, width), -0.75, device="cuda", dtype=torch.bfloat16)
        probes = torch.tensor(
            [0, width - 1, width, 2**31 - 1, 2**31, n - 1], device="cuda"
        )
        expected = self.reference(
            torch.full((len(probes),), 0.25, device="cuda"),
            torch.full((len(probes),), -0.75, device="cuda"), 4.0, 25.0,
        ).to(up.dtype)
        for inplace in (False, True):
            gate = torch.full_like(up, 0.25)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = self.situ(gate, up, 4.0, 25.0, inplace=inplace)
            torch.cuda.synchronize()
            if inplace:
                self.assertEqual(out.data_ptr(), gate.data_ptr())
                self.assertLessEqual(torch.cuda.max_memory_allocated() - before,
                                     1024 * 1024)
            torch.testing.assert_close(out.view(-1)[probes], expected,
                                       rtol=0.008, atol=2e-6)
            torch.testing.assert_close(up.view(-1)[probes],
                                       torch.full_like(expected, -0.75),
                                       rtol=0, atol=0)
            del out, gate

    def test_inplace_graph_replay(self):
        g = torch.randn(9, 1025, device="cuda", dtype=torch.bfloat16)
        u = torch.randn_like(g)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.situ(g, u, 4.0, 25.0, inplace=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.situ(g, u, 4.0, 25.0, inplace=True)
        for _ in range(3):
            g.normal_()
            u.normal_()
            expected = self.reference(g, u, 4.0, 25.0).to(g.dtype)
            graph.replay()
            torch.testing.assert_close(out, expected, rtol=0.008, atol=2e-6)

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

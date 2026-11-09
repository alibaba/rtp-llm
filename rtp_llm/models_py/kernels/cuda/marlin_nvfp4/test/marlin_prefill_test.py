import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import (
    _MAX_ROUTES_PER_CHUNK,
    Nvfp4MarlinMoE,
)


class MarlinPrefillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Nvfp4MarlinMoE.is_supported():
            raise unittest.SkipTest("a supported NVFP4 Marlin runtime is required")
        torch.manual_seed(1853)
        torch.backends.cuda.matmul.allow_tf32 = False
        # Small matrices keep the 128K routing regression inexpensive. Repeat
        # four distinct matrices to retain Qwen4's 256 local / 512 global IDs.
        cls.k = cls.n = 128
        q1 = torch.randint(256, (4, 256, 64), device="cuda", dtype=torch.uint8)
        q2 = torch.randint(256, (4, 128, 64), device="cuda", dtype=torch.uint8)
        s1 = (2.0 ** torch.randint(-4, 1, (4, 256, 8), device="cuda")).to(
            torch.float8_e4m3fn
        )
        s2 = (2.0 ** torch.randint(-4, 1, (4, 128, 8), device="cuda")).to(
            torch.float8_e4m3fn
        )
        g1 = torch.tensor([0.01, 0.02, 0.005, 0.015], device="cuda")
        g2 = torch.tensor([0.02, 0.01, 0.015, 0.005], device="cuda")
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )

        def dequant(q, scales, global_scale):
            codes = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
            return (
                lut[codes]
                * scales.float().repeat_interleave(16, -1)
                * global_scale[:, None, None]
            )

        cls.w1, cls.w2 = dequant(q1, s1, g1), dequant(q2, s2, g2)
        cls.models = [
            Nvfp4MarlinMoE(
                *(t.repeat_interleave(64, 0) for t in (q1, s1, g1, q2, s2, g2)),
                2,
                rank,
            )
            for rank in (0, 1)
        ]

    def inputs(self, rows, rank):
        x = torch.randn(rows, self.k, device="cuda", dtype=torch.bfloat16) * 0.2
        ids = (
            (
                torch.randint(512, (rows, 1), device="cuda")
                + torch.arange(10, device="cuda") * 37
            )
            .remainder(512)
            .int()
        )
        ids[::113] = torch.arange(10, device="cuda") + (1 - rank) * 256
        weights = torch.randn(rows, 10, device="cuda").softmax(-1)
        return x, ids, weights

    def assert_reference(self, actual, x, ids, weights, rank):
        expected = torch.zeros_like(x, dtype=torch.float32)
        local = ids - rank * 256
        for base in range(4):
            up, gate = F.linear(x.float(), self.w1[base]).chunk(2, -1)
            output = F.linear(up * F.silu(gate), self.w2[base])
            mask = (local >= base * 64) & (local < (base + 1) * 64)
            expected += output * (weights * mask).sum(-1, keepdim=True)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual(torch.count_nonzero(actual[::113]).item(), 0)
        boundary = _MAX_ROUTES_PER_CHUNK // 10
        for rows in (slice(None), slice(-3, None), slice(boundary - 1, boundary + 2)):
            error = (actual[rows].float() - expected[rows]).square().mean().sqrt()
            scale = expected[rows].square().mean().sqrt()
            self.assertLess((error / scale).item(), 0.02)

    def test_128k_prefill_top10_and_partial_tail(self):
        for rank, model in enumerate(self.models):
            with self.subTest(rank=rank):
                args = self.inputs(131073, rank)
                self.assert_reference(model(*args), *args, rank)

    def test_chunk_boundary_graph_refreshes_inputs_and_nonlocal_routes(self):
        rows = _MAX_ROUTES_PER_CHUNK // 10 + 3
        for rank, model in enumerate(self.models):
            with self.subTest(rank=rank):
                args = self.inputs(rows, rank)
                model(*args)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = model(*args)
                for _ in range(2):
                    for dst, src in zip(args, self.inputs(rows, rank)):
                        dst.copy_(src)
                    expected = model(*args)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assert_reference(actual, *args, rank)
                graph.reset()


if __name__ == "__main__":
    unittest.main()

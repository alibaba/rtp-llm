import torch

from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import Nvfp4MarlinMoE


class MarlinNvfp4TestFixture:
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(918)
        cls.n, cls.k, cls.experts = 640, 2560, 4
        cls.q1 = torch.randint(256, (4, 1280, 1280), device="cuda", dtype=torch.uint8)
        cls.q2 = torch.randint(256, (4, 2560, 320), device="cuda", dtype=torch.uint8)
        cls.s1 = (2.0 ** torch.randint(-9, 9, (4, 1280, 160), device="cuda")).to(
            torch.float8_e4m3fn
        )
        cls.s2 = (2.0 ** torch.randint(-9, 9, (4, 2560, 40), device="cuda")).to(
            torch.float8_e4m3fn
        )
        cls.s1[0].zero_()
        cls.g1 = torch.tensor([1e-5, 1e-4, 2e-4, 5e-5], device="cuda")
        cls.g2 = torch.tensor([2e-4, 5e-5, 1e-4, 1e-5], device="cuda")
        cls.models = [
            Nvfp4MarlinMoE(cls.q1, cls.s1, cls.g1, cls.q2, cls.s2, cls.g2, 2, rank)
            for rank in (0, 1)
        ]
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )

        def dequant(q, sf, glob):
            codes = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
            return (
                lut[codes] * sf.float().repeat_interleave(16, -1) * glob[:, None, None]
            )

        cls.w1 = dequant(cls.q1, cls.s1, cls.g1)
        cls.w2 = dequant(cls.q2, cls.s2, cls.g2)

    def inputs(self, rows, rank=0, local=None):
        x = torch.randn(rows, self.k, device="cuda", dtype=torch.bfloat16) * 0.2
        ids = torch.rand(rows, 8, device="cuda").topk(3, -1).indices.int()
        if local is not None:
            ids = (
                torch.tensor(local, device="cuda", dtype=torch.int32)
                .expand(rows, -1)
                .contiguous()
            )
        weights = torch.randn(rows, 3, device="cuda").softmax(-1)
        return x, ids, weights

    def reference(self, x, ids, weights, rank):
        result = torch.zeros_like(x, dtype=torch.float32)
        for row in range(x.shape[0]):
            for slot in range(ids.shape[1]):
                expert = int(ids[row, slot]) - rank * self.experts
                if not 0 <= expert < self.experts:
                    continue
                upgate = self.w1[expert] @ x[row].float()
                up, gate = upgate.chunk(2)
                result[row] += (
                    self.w2[expert] @ (up * torch.nn.functional.silu(gate))
                ) * weights[row, slot]
        return result

    def assert_reference(self, actual, expected):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        if not bool(expected.any()):
            self.assertTrue(bool((actual == 0).all()))
            return
        relative_rmse = (
            ((actual.float() - expected).square().mean() / expected.square().mean())
            .sqrt()
            .item()
        )
        self.assertLess(relative_rmse, 0.02)

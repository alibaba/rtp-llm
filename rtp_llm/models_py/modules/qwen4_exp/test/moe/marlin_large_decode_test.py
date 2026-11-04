import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.moe.marlin_nvfp4 import Nvfp4MarlinMoE


class MarlinLargeDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 is required")

    def test_actual_ep2_expert_count_top10_target_rows_and_live_graph(self):
        torch.manual_seed(1807)
        torch.backends.cuda.matmul.allow_tf32 = False
        q1 = torch.randint(256, (4, 1280, 1280), device="cuda", dtype=torch.uint8)
        q2 = torch.randint(256, (4, 2560, 320), device="cuda", dtype=torch.uint8)
        s1 = (2.0 ** torch.randint(-6, 8, (4, 1280, 160), device="cuda")).to(
            torch.float8_e4m3fn
        )
        s2 = (2.0 ** torch.randint(-6, 8, (4, 2560, 40), device="cuda")).to(
            torch.float8_e4m3fn
        )
        g1 = torch.tensor([1e-5, 1e-4, 2e-4, 5e-5], device="cuda")
        g2 = torch.tensor([2e-4, 5e-5, 1e-4, 1e-5], device="cuda")
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )

        def dequant(q, scale, global_scale):
            ids = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
            return (
                lut[ids]
                * scale.float().repeat_interleave(16, -1)
                * global_scale[:, None, None]
            )

        w1, w2 = dequant(q1, s1, g1), dequant(q2, s2, g2)
        for rank in (0, 1):
            # Replication preserves an independent four-matrix FP32 oracle
            # while exercising all 256 local / 512 global expert identities.
            args = [t.repeat_interleave(64, 0) for t in (q1, s1, g1, q2, s2, g2)]
            model = Nvfp4MarlinMoE(*args, 2, rank)
            del args
            for rows in (16, 64, 256):
                x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16) * 0.2
                ids = torch.rand(rows, 512, device="cuda").topk(10, -1).indices.int()
                weights = torch.randn(rows, 10, device="cuda").softmax(-1)
                actual = model(x, ids, weights)
                expected = torch.zeros(rows, 2560, device="cuda", dtype=torch.float32)
                local = ids - rank * 256
                for base in range(4):
                    gate_up = F.linear(x.float(), w1[base])
                    up, gate = gate_up.chunk(2, -1)
                    output = F.linear(up * F.silu(gate), w2[base])
                    mask = (local >= base * 64) & (local < (base + 1) * 64)
                    expected += output * (weights * mask).sum(-1, keepdim=True)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                error = (
                    actual.float() - expected
                ).square().mean().sqrt() / expected.square().mean().sqrt()
                self.assertLess(error.item(), 0.02)
                if rows in (64, 256):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = model(x, ids, weights)
                    x.mul_(0.8)
                    ids.copy_(
                        torch.rand(rows, 512, device="cuda").topk(10, -1).indices.int()
                    )
                    weights.copy_(torch.randn_like(weights).softmax(-1))
                    eager = model(x, ids, weights)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
                    graph.reset()


if __name__ == "__main__":
    unittest.main()

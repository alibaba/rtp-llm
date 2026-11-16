import unittest

import torch

from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual
from rtp_llm.models_py.modules.qwen4_exp.gated_residual import Qwen4GatedResidual


class GatedResidualBf16Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.capability = torch.cuda.get_device_capability()
        if cls.capability not in ((10, 3), (12, 0)):
            raise unittest.SkipTest("SM103 or SM120 is required")

    def test_bf16_consumer_preserves_residual_norm_and_projection_boundaries(self):
        torch.manual_seed(1779)

        def random(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02

        sm103 = self.capability == (10, 3)
        hidden, rank = (8192, 1024) if sm103 else (2560, 320)
        width = 4 * hidden
        weights = (
            random(width),
            random(rank, width),
            random(width, rank),
            random(4, width),
        )
        kwargs = dict(hc_mult=4, norm_eps=1e-6, role="PREFILL")
        baseline = GatedResidual(*weights, **kwargs)
        candidate = Qwen4GatedResidual(*weights, **kwargs)
        for rows in ((4, 64, 2048) if sm103 else (7, 4093, 8192, 16384)):
            x = random(rows, width)
            s = random(rows, hidden)
            g = random(rows, 4)
            expected = baseline.forward_with_injection(x, s, g)
            actual = candidate.forward_with_injection(x, s, g)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            if sm103 and rows == 4:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = candidate.forward_with_injection(x, s, g)
                x.copy_(random(*x.shape))
                s.copy_(random(*s.shape))
                g.copy_(random(*g.shape))
                graph.replay()
                expected = baseline.forward_with_injection(x, s, g)
                for a, b in zip(captured, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual
from rtp_llm.models_py.modules.qwen4_exp.gated_residual import Qwen4GatedResidual


class GatedResidualBf16Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 is required")

    def test_bf16_consumer_preserves_residual_norm_and_projection_boundaries(self):
        torch.manual_seed(1779)

        def random(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02

        weights = (
            random(10240),
            random(320, 10240),
            random(10240, 320),
            random(4, 10240),
        )
        kwargs = dict(hc_mult=4, norm_eps=1e-6, role="PREFILL")
        baseline = GatedResidual(*weights, **kwargs)
        candidate = Qwen4GatedResidual(*weights, **kwargs)
        for rows in (7, 4093, 8192, 16384):
            x = random(rows, 10240)
            s = random(rows, 2560)
            g = random(rows, 4)
            expected = baseline.forward_with_injection(x, s, g)
            actual = candidate.forward_with_injection(x, s, g)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

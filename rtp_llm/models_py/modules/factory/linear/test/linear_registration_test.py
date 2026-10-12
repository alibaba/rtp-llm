import unittest

import torch
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear


class LinearRegistrationTest(unittest.TestCase):
    def test_unscaled_fp32_projection_keeps_nonquantized_backend(self):
        self.assertIn(CudaF16Linear, LinearFactory._strategies)
        weight = torch.arange(12, dtype=torch.float32).view(4, 3)
        layer = LinearFactory.create_linear(
            weight, None, None, Fp8BlockWiseQuantConfig()
        )
        self.assertIsInstance(layer, CudaF16Linear)
        actual = layer(torch.ones(2, 4))
        torch.testing.assert_close(actual, weight.sum(0).expand(2, 3), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

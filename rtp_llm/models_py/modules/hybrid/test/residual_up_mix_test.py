import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.test import residual_fixtures

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.up_mix"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualUpMixTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 fast path")
        torch.manual_seed(607)

    def test_complete_hyperconnection(self):
        for rows in (1, 2, 3, 4):
            for inject in (False, True):
                module = residual_fixtures.ResidualFp8Fixture().module(inject)
                module.down_projection.role = module.up_projection.role = "DECODE"
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                with patch(MODULE + ".try_fp8_up_mix", return_value=None):
                    expected = module(x)
                residual_fixtures.ResidualFp8Fixture().assert_outputs(
                    module(x), expected
                )


if __name__ == "__main__":
    unittest.main()

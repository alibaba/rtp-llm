import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.test.residual_fixtures import ResidualFp8Fixture

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.norm"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualNormFp8Test(unittest.TestCase, ResidualFp8Fixture):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 fast path")
        torch.manual_seed(703)

    def test_complete_hyperconnection_consumer(self):
        for rows in (8192, 12288):
            for inject in (False, True):
                module = self.module(inject)
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                with patch(
                    MODULE + ".try_grouped_rms_norm_with_fp8_scale", return_value=None
                ):
                    expected = module(x)
                self.assert_outputs(module(x), expected)

    def test_graph_refreshes_norm_gamma_and_global_scale(self):
        module = self.module()
        x = torch.randn(8192, 10240, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            module(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = module(x)
        for gain in (0.01, 0.2, 2.0):
            x.normal_()
            module.norm_gamma.normal_(0, gain)
            graph.replay()
            with patch(
                MODULE + ".try_grouped_rms_norm_with_fp8_scale", return_value=None
            ):
                expected = module(x)
            self.assert_outputs(actual, expected)


if __name__ == "__main__":
    unittest.main()

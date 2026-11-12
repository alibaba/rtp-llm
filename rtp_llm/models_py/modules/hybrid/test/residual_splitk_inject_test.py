import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.test.residual_fixtures import ResidualFp8Fixture

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.splitk_inject"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualSplitKInjectTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 native numerical contract")
        torch.manual_seed(83001)

    def test_complete_consumer_and_graph_live_weight(self):
        for rows in (4, 8, 16, 17, 24, 32, 4093, 4094, 4095, 4096):
            model = ResidualFp8Fixture().module(True)
            model.up_projection.role = "DECODE"
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                model(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model(x)
            for gain in (0.01, 0.3, 2.0):
                x.normal_(0, gain)
                model.norm_gamma.normal_(0, gain)
                model.inject_projection.compute_weight.normal_(0, gain)
                graph.replay()
                with patch(MODULE + ".try_splitk_down_inject", return_value=None):
                    expected = model(x)
                ResidualFp8Fixture().assert_outputs(actual, expected)


if __name__ == "__main__":
    unittest.main()

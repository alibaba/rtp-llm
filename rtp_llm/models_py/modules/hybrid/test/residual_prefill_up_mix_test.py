import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.test.residual_fixtures import ResidualFp8Fixture

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.prefill_up_mix"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualPrefillUpMixTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 measured prefill consumer")
        torch.manual_seed(111201)

    def test_complete_consumer_with_and_without_write_gate(self):
        oracle = ResidualFp8Fixture()
        for rows in (
            4093,
            4094,
            4095,
            4096,
            6144,
            8192,
            12288,
            16384,
            24576,
            32768,
            49152,
        ):
            for inject in (False, True):
                model = oracle.module(inject)
                model.up_projection.role = "PREFILL"
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                with patch(MODULE + ".try_fp8_prefill_up_mix", return_value=None):
                    expected = model(x)
                oracle.assert_outputs(model(x), expected)

    def test_complete_graph_refreshes_input_raw_gamma_and_up_scale(self):
        oracle = ResidualFp8Fixture()
        for rows in (4093, 4096, 12288):
            model = oracle.module(True)
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                model(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model(x)
            for scale in (0.013475, 0.119713, 0.2719):
                x.normal_()
                model.norm_gamma.normal_(0, 0.2)
                model.up_projection.weight.copy_(
                    (torch.randn_like(model.up_projection.weight.float()) * 0.1).to(
                        torch.float8_e4m3fn
                    )
                )
                model.up_projection.scale.fill_(scale)
                graph.replay()
                with patch(MODULE + ".try_fp8_prefill_up_mix", return_value=None):
                    expected = model(x)
                oracle.assert_outputs(actual, expected)
            graph.reset()


if __name__ == "__main__":
    unittest.main()

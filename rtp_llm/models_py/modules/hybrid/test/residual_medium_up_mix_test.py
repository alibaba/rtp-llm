"""Full HC consumer: medium GEMM epilogue, fresh Graph inputs and gains."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.test import residual_fixtures as oracle

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.medium_up_mix"


class MediumHCConsumerTest(unittest.TestCase):
    def test_complete_consumer_with_and_without_inject(self):
        reference = oracle.ResidualFp8Fixture()
        for rows in (8, 16, 20, 24, 28, 32):
            for inject in (False, True):
                torch.manual_seed(28100 + rows)
                model = reference.module(inject)
                model.up_projection.role = model.down_projection.role = "DECODE"
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                with patch(MODULE + ".try_fp8_hc_medium_up_mix", return_value=None):
                    expected = model(x)
                reference.assert_outputs(model(x), expected)

    def test_graph_refreshes_input_gamma_fp8_weight_and_scale(self):
        reference = oracle.ResidualFp8Fixture()
        for rows in (8, 16, 28):
            model = reference.module(True)
            model.up_projection.role = model.down_projection.role = "DECODE"
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                model(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model(x)
            for scale in (0.013475, 0.119713, 0.2719):
                x.normal_()
                model.norm_gamma.normal_(0, 0.07)
                model.up_projection.weight.copy_(
                    (torch.randn_like(model.up_projection.weight.float()) * 0.1).to(
                        torch.float8_e4m3fn
                    )
                )
                model.up_projection.scale.fill_(scale)
                with patch(MODULE + ".try_fp8_hc_medium_up_mix", return_value=None):
                    expected = model(x)
                graph.replay()
                torch.cuda.synchronize()
                reference.assert_outputs(actual, expected)


if __name__ == "__main__":
    unittest.main()

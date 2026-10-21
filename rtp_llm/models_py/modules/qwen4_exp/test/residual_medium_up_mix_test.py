"""Full HC consumer: medium GEMM epilogue, fresh Graph inputs and gains."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.qwen4_exp.test import residual_norm_fp8_test as oracle
from rtp_llm.models_py.triton_kernels.common.gated_residual_medium_up_mix_triton import (
    try_fp8_hc_medium_up_mix,
)
from rtp_llm.models_py.triton_kernels.common.gated_residual_mix_triton import (
    fused_mix_reduce,
)

MODULE = "rtp_llm.models_py.triton_kernels.common.gated_residual_medium_up_mix_triton"


class MediumHCConsumerTest(unittest.TestCase):
    def test_up_consumer_non_power_of_two_scales(self):
        torch.manual_seed(28173)
        for rows in (8, 16, 20, 24, 28, 32):
            for magnitude, scale in ((0.01, 0.013475), (1.0, 0.119713), (10.0, 0.2719)):
                with self.subTest(rows=rows, magnitude=magnitude, scale=scale):
                    x = (
                        torch.randn(rows, 320, device="cuda", dtype=torch.bfloat16)
                        * magnitude
                    )
                    w = (torch.randn(10240, 320, device="cuda") * 0.1).to(
                        torch.float8_e4m3fn
                    )
                    sw = torch.tensor([scale], device="cuda")
                    norm = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                    expected = fused_mix_reduce(
                        Fp8Linear(w, sw, role="DECODE")(x), norm, 4
                    )
                    actual = try_fp8_hc_medium_up_mix(x, w, sw, norm, 4)
                    self.assertIsNotNone(actual)
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.int16), expected.view(torch.int16)
                        ),
                        int((actual != expected).sum()),
                    )

    def test_complete_consumer_with_and_without_inject(self):
        reference = oracle.ResidualNormFp8Test()
        for rows in (1, 3, 4, 8, 16, 20, 24, 28, 32):
            for inject in (False, True):
                torch.manual_seed(28100 + rows)
                model = reference.module(inject)
                model.up_projection.role = model.down_projection.role = "DECODE"
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                with patch(MODULE + ".try_fp8_hc_medium_up_mix", return_value=None):
                    expected = model(x)
                reference.assert_outputs(model(x), expected)

    def test_graph_refreshes_input_gamma_fp8_weight_and_scale(self):
        reference = oracle.ResidualNormFp8Test()
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

    def test_unsupported_small_and_large_shapes_fallback(self):
        w = torch.empty((10240, 320), device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.ones(1, device="cuda")
        for rows in (1, 3, 4, 5, 7, 33, 4096):
            x = torch.empty(rows, 320, device="cuda", dtype=torch.bfloat16)
            norm = torch.empty(rows, 10240, device="cuda", dtype=x.dtype)
            self.assertIsNone(try_fp8_hc_medium_up_mix(x, w, s, norm, 4))


if __name__ == "__main__":
    unittest.main()

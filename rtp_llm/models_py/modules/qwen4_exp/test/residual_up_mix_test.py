import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.fp8_dense_gemv_triton import fp8_hc_gemv
from rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_mix_triton import (
    fused_mix_reduce,
)
from rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_up_mix_triton import (
    try_fp8_up_mix,
)
from rtp_llm.models_py.modules.qwen4_exp.test import residual_norm_fp8_test

MODULE = "rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_up_mix_triton"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualUpMixTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 fast path")
        torch.manual_seed(607)

    def test_bf16_logits_gate_product_and_branch_order(self):
        w = torch.randn(10240, 320, device="cuda").to(torch.float8_e4m3fn)
        for rows in (1, 3, 4):
            for gain in (0.03125, 0.25, 0.0):
                x = torch.randn(rows, 320, device="cuda", dtype=torch.bfloat16)
                normed = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                scale = torch.tensor([gain], device="cuda")
                for factor in (1.0, 1000.0):
                    values = x * factor
                    expected = fused_mix_reduce(
                        fp8_hc_gemv(values, w, scale), normed, 4
                    )
                    actual = try_fp8_up_mix(values, w, scale, normed, 4)
                    self.assertIsNotNone(actual)
                    self.assertTrue(torch.equal(actual, expected))

    def test_complete_hyperconnection(self):
        for rows in (1, 3, 4):
            for inject in (False, True):
                module = residual_norm_fp8_test.ResidualNormFp8Test().module(inject)
                module.down_projection.role = module.up_projection.role = "DECODE"
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                with patch(MODULE + ".try_fp8_up_mix", return_value=None):
                    expected = module(x)
                residual_norm_fp8_test.ResidualNormFp8Test().assert_outputs(
                    module(x), expected
                )

    def test_graph_reads_live_input_weights_scale_and_norm(self):
        x = torch.randn(4, 320, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(10240, 320, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.tensor([0.03125], device="cuda")
        normed = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            try_fp8_up_mix(x, w, scale, normed, 4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_fp8_up_mix(x, w, scale, normed, 4)
        for gain in (0.25, 0.03125):
            x.normal_()
            normed.normal_()
            w.copy_(torch.randn(w.shape, device="cuda").to(w.dtype))
            scale.fill_(gain)
            graph.replay()
            expected = fused_mix_reduce(fp8_hc_gemv(x, w, scale), normed, 4)
            self.assertTrue(torch.equal(actual, expected))

    def test_other_shapes_and_gradients_fall_back(self):
        w = torch.empty(10240, 320, device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, device="cuda")
        for rows in (0, 5, 8, 16):
            x = torch.empty(rows, 320, device="cuda", dtype=torch.bfloat16)
            normed = torch.empty(rows, 10240, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(try_fp8_up_mix(x, w, scale, normed, 4))
        x = torch.empty(4, 320, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        normed = torch.empty(4, 10240, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(try_fp8_up_mix(x, w, scale, normed, 4))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.fp8_dense import Qwen4Fp8Linear
from rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_mix_triton import (
    fused_mix_reduce,
)
from rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_prefill_up_mix_triton import (
    try_fp8_prefill_up_mix,
)
from rtp_llm.models_py.modules.qwen4_exp.test.residual_norm_fp8_test import (
    ResidualNormFp8Test,
)

MODULE = (
    "rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_prefill_up_mix_triton"
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualPrefillUpMixTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 measured prefill consumer")
        torch.manual_seed(111201)

    def test_native_projection_and_bf16_boundaries_with_independent_scales(self):
        for rows in (4093, 4094, 4095, 4096, 12288):
            for magnitude, scale in ((0.01, 0.013475), (1.0, 0.119713), (10.0, 0.2719)):
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
                    Qwen4Fp8Linear(w, sw, role="PREFILL")(x), norm, 4
                )
                actual = try_fp8_prefill_up_mix(x, w, sw, norm, 4)
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_complete_consumer_with_and_without_write_gate(self):
        oracle = ResidualNormFp8Test()
        for rows in (4093, 4094, 4095, 4096, 12288):
            for inject in (False, True):
                model = oracle.module(inject)
                model.up_projection.role = "PREFILL"
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                with patch(MODULE + ".try_fp8_prefill_up_mix", return_value=None):
                    expected = model(x)
                oracle.assert_outputs(model(x), expected)

    def test_complete_graph_refreshes_input_raw_gamma_and_up_scale(self):
        oracle = ResidualNormFp8Test()
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

    def test_unmeasured_shapes_and_gradients_decline_before_writes(self):
        w = torch.empty(10240, 320, device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, device="cuda")
        for rows in (1, 3, 4, 8, 16, 32, 33, 4092, 4097, 8192, 12287, 12289):
            x = torch.empty(rows, 320, device="cuda", dtype=torch.bfloat16)
            norm = torch.empty(rows, 10240, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(try_fp8_prefill_up_mix(x, w, scale, norm, 4))
        x = torch.empty(
            4096, 320, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        norm = torch.empty(4096, 10240, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(try_fp8_prefill_up_mix(x, w, scale, norm, 4))


if __name__ == "__main__":
    unittest.main()

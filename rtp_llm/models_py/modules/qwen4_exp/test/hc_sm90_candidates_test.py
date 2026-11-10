"""SM90 correctness and graph checks for opt-in HC candidates."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.qwen4_exp.test import residual_norm_fp8_test as fixture
from rtp_llm.models_py.triton_kernels.common.gated_residual_norm_triton import (
    grouped_rms_norm_triton,
    try_grouped_rms_norm_with_fp8_scale,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class HcSm90CandidatesTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 is required")
        torch.manual_seed(5245)


    def test_prefill_norm_scale_matches_native_and_complete_consumer(self):
        flag = "RTP_LLM_QWEN4_HC_SM90_PREFILL_NORM_SCALE"
        for rows in (33, 257, 2155, 4096):
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            gamma = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07
            expected = grouped_rms_norm_triton(x, gamma, 2560, 1e-6)
            q_ref, s_ref = scaled_fp8_per_tensor_quant(expected)
            actual, scale = try_grouped_rms_norm_with_fp8_scale(x, gamma, 2560, 1e-6)
            q, _ = scaled_fp8_per_tensor_quant(actual, scale=scale)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(scale, s_ref, rtol=0, atol=0)
            self.assertTrue(torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)))
            model = fixture.ResidualNormFp8Test().module()
            with patch.dict(os.environ, {flag: "0"}):
                ref = model(x)
            with patch.dict(os.environ, {flag: "1"}):
                out = model(x)
            fixture.ResidualNormFp8Test().assert_outputs(out, ref)

    def test_prefill_norm_graph_refreshes_input(self):
        flag = "RTP_LLM_QWEN4_HC_SM90_PREFILL_NORM_SCALE"
        model = fixture.ResidualNormFp8Test().module()
        x = torch.randn(257, 10240, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {flag: "1"}):
            for _ in range(3):
                model(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model(x)
            for _ in range(3):
                x.normal_()
                with patch.dict(os.environ, {flag: "0"}):
                    expected = model(x)
                graph.replay()
                fixture.ResidualNormFp8Test().assert_outputs(actual, expected)


if __name__ == "__main__":
    unittest.main()

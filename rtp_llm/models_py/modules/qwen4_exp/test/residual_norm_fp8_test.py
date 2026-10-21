import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.hybrid.gated_residual import (
    GatedResidual,
)
from rtp_llm.models_py.triton_kernels.common.gated_residual_norm_triton import (
    grouped_rms_norm_triton,
    try_grouped_rms_norm_with_fp8_scale,
)

MODULE = "rtp_llm.models_py.triton_kernels.common.gated_residual_norm_triton"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualNormFp8Test(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 fast path")
        torch.manual_seed(703)

    def test_norm_scale_and_fp8_bytes(self):
        for rows in (256, 257, 4093, 4096, 12288):
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            gamma = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07
            for kind in ("random", "wide", "zero", "nonfinite"):
                with self.subTest(rows=rows, kind=kind):
                    if kind == "wide":
                        x.mul_(1000.0)
                    if kind == "zero":
                        x.zero_()
                    if kind == "nonfinite":
                        x.normal_()
                        x[0, 0] = float("inf")
                        x[1, 2560] = float("nan")
                    expected = grouped_rms_norm_triton(x, gamma, 2560, 1.0e-6)
                    reference, reference_scale = scaled_fp8_per_tensor_quant(expected)
                    actual, scale = try_grouped_rms_norm_with_fp8_scale(
                        x, gamma, 2560, 1.0e-6
                    )
                    torch.testing.assert_close(
                        actual, expected, rtol=0, atol=0, equal_nan=True
                    )
                    self.assertTrue(torch.equal(scale, reference_scale))
                    quantized, _ = scaled_fp8_per_tensor_quant(actual, scale=scale)
                    self.assertTrue(
                        torch.equal(
                            quantized.view(torch.uint8), reference.view(torch.uint8)
                        )
                    )

    def module(self, inject=True):
        def weight(shape):
            return (torch.randn(shape, device="cuda") * 0.1).to(torch.float8_e4m3fn)

        return GatedResidual(
            torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07,
            weight((320, 10240)),
            weight((10240, 320)),
            weight((4, 10240)) if inject else None,
            hc_mult=4,
            norm_eps=1.0e-6,
            weight_scales=[torch.tensor([0.03125], device="cuda")] * 3,
            role="PREFILL",
        )

    def assert_outputs(self, actual, expected):
        for a, b in zip(actual, expected):
            if b is None:
                self.assertIsNone(a)
            else:
                self.assertTrue(torch.equal(a, b))

    def test_complete_hyperconnection_consumer(self):
        for rows in (257, 4093, 8192, 12288):
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

    def test_unsupported_shapes_and_training_fall_back(self):
        gamma = torch.zeros(10240, device="cuda", dtype=torch.bfloat16)
        for rows in (0, 1, 32, 255, 12289):
            x = torch.empty(rows, 10240, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(
                try_grouped_rms_norm_with_fp8_scale(x, gamma, 2560, 1.0e-6)
            )
        x = torch.empty(257, 10240, device="cuda", dtype=torch.bfloat16)
        for eps in (-1.0, float("inf")):
            self.assertIsNone(try_grouped_rms_norm_with_fp8_scale(x, gamma, 2560, eps))
        x.requires_grad_(True)
        self.assertIsNone(try_grouped_rms_norm_with_fp8_scale(x, gamma, 2560, 1.0e-6))


if __name__ == "__main__":
    unittest.main()

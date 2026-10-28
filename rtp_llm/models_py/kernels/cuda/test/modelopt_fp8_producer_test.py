"""Producer fusion must preserve the original BF16-to-FP8 rounding boundary."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda import modelopt_fp8_quant as quant
from rtp_llm.models_py.modules.factory.linear.impl.cuda.modelopt_fp8_linear import (
    CudaModelOptFp8Linear,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class ModelOptFp8ProducerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261028)
        self.enabled = patch.multiple(
            quant, _FUSED_PRODUCER_ENABLED=True, _FUSED_QUANT_ENABLED=True
        )
        self.enabled.start()
        self.addCleanup(self.enabled.stop)

    def test_byte_equivalence(self):
        for width in (3072, 5120, 6144):
            for rows in (1, 3, 17, 64, 257):
                for rms in (False, True):
                    with self.subTest(width=width, rows=rows, rms=rms):
                        x = torch.randn(
                            rows, width, device="cuda", dtype=torch.bfloat16
                        )
                        z = torch.randn_like(x.repeat(1, 2))[:, :width]
                        w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
                        for scale_value in (1e-10, 0.03125, 2.0):
                            scale = torch.tensor(scale_value, device="cuda")
                            if rms:
                                y = RmsNormGated(w)(
                                    x.reshape(-1, 128), z.reshape(-1, 128)
                                ).reshape(rows, width)
                            else:
                                y = x * torch.sigmoid(z)
                            expected = quant.static_fp8_quant_rows(y, scale)
                            actual = quant.gated_fp8_quant_rows(
                                x, z, scale, norm_weight=w if rms else None
                            )
                            self.assertTrue(
                                torch.equal(
                                    expected[0].view(torch.uint8),
                                    actual[0].view(torch.uint8),
                                )
                            )
                            self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_sigmoid_bf16_values(self):
        # Include saturation, signed zero, subnormals and NaN/Inf gate inputs.
        z = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        z = z.view(torch.bfloat16).repeat(3).reshape(64, 3072)
        x = torch.randn_like(z)
        scale = torch.tensor(0.03125, device="cuda")
        expected = quant.static_fp8_quant_rows(x * torch.sigmoid(z), scale)
        actual = quant.gated_fp8_quant_rows(x, z, scale)
        self.assertTrue(
            torch.equal(expected[0].view(torch.uint8), actual[0].view(torch.uint8))
        )

    def test_graph_replay_reads_current_inputs(self):
        x = torch.randn(3, 3072, device="cuda", dtype=torch.bfloat16)
        z = torch.randn_like(x)
        scale = torch.tensor(0.03125, device="cuda")
        quant.gated_fp8_quant_rows(x, z, scale)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = quant.gated_fp8_quant_rows(x, z, scale)
        for factor in (0.5, -2.0, 4.0):
            x.mul_(factor)
            z.add_(0.125)
            scale.mul_(2)
            graph.replay()
            expected = quant.static_fp8_quant_rows(x * torch.sigmoid(z), scale)
            self.assertTrue(
                torch.equal(expected[0].view(torch.uint8), actual[0].view(torch.uint8))
            )
            self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_fallback_and_empty(self):
        x = torch.zeros(0, 3072, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor(0.03125, device="cuda")
        actual = quant.gated_fp8_quant_rows(x, x, scale)
        self.assertEqual(actual[0].shape, (0, 3072))
        with patch.object(quant, "_FUSED_PRODUCER_ENABLED", False):
            self.assertIsNone(quant.gated_fp8_quant_rows(x, x, scale))
        x = torch.zeros(3, 256, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(quant.gated_fp8_quant_rows(x, x, scale))
        x = torch.zeros(3, 3072, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(quant.gated_fp8_quant_rows(x, x[:, :-1], scale))
        self.assertIsNone(quant.gated_fp8_quant_rows(x, x, scale.cpu()))
        self.assertIsNone(quant.gated_fp8_quant_rows(x.float(), x.float(), scale))

    def test_projection_and_multiple_scales(self):
        weight = (torch.randn(5120, 3072, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        scales = torch.stack(
            (
                torch.linspace(0.01, 0.03, 5120, device="cuda"),
                torch.full((5120,), 0.03125, device="cuda"),
            )
        )
        layer = CudaModelOptFp8Linear(weight, scales)
        x = torch.randn(3, 3072, device="cuda", dtype=torch.bfloat16)
        z = torch.randn_like(x)
        expected = layer(x * torch.sigmoid(z))
        actual = layer.forward_with_gate(x, z)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        scales[1, 2560:] = 0.0625
        layer = CudaModelOptFp8Linear(weight, scales)
        self.assertIsNone(layer.forward_with_gate(x, z))


if __name__ == "__main__":
    unittest.main()

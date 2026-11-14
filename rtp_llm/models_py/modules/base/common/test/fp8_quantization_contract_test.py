import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.base.common.fp8_linear import (
    Fp8Linear,
    _quantize_projection_input,
    project_fp8_pair,
)
from rtp_llm.models_py.triton_kernels.common.fp8_small_quant import try_small_fp8_quant

MODULE = "rtp_llm.models_py.triton_kernels.common.fp8_small_quant"


class Fp8SmallQuantTest(unittest.TestCase):
    def setUp(self):
        # fail-loud rather than skip: this target is pinned to RTX_5000_PRO_CU13.
        self.assertTrue(torch.cuda.is_available(), "This target requires SM120")
        self.assertEqual(
            torch.cuda.get_device_capability(),
            (12, 0),
            "target pool is RTX_5000_PRO_CU13 (SM120)",
        )
        torch.manual_seed(20261019)

    def compare(self, x, padded=True):
        shape = (((x.shape[0] + 15) // 16) * 16, x.shape[1]) if padded else x.shape
        actual = try_small_fp8_quant(
            x, output=torch.empty(shape, device=x.device, dtype=torch.float8_e4m3fn)
        )
        self.assertIsNotNone(actual)
        expected = scaled_fp8_per_tensor_quant(
            x, output=torch.empty(shape, device=x.device, dtype=torch.float8_e4m3fn)
        )
        self.assertTrue(
            torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(actual[1].view(torch.int32), expected[1].view(torch.int32))
        )

    def test_fused_shared_projection_graph_refreshes_quantized_values(self):
        import flashinfer

        layer = Fp8Linear(
            (torch.randn(2560, 640, device="cuda") * 0.1).to(torch.float8_e4m3fn),
            torch.tensor([0.11743], device="cuda"),
            role="DECODE",
        )
        for rows in (4, 8, 12, 16):
            gate_up = torch.randn(rows, 1280, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                layer.forward_silu_and_mul(gate_up)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = layer.forward_silu_and_mul(gate_up)
            pointer = actual.data_ptr()
            for magnitude in (0.0, 1.0e-12, 0.01, 1.0, 100.0):
                gate_up.normal_().mul_(magnitude)
                layer.scale.fill_(0.013475 + magnitude * 0.001)
                activated = flashinfer.activation.silu_and_mul(gate_up)
                expected = layer(activated)
                graph.replay()
                self.assertEqual(actual.data_ptr(), pointer)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                )
            graph.reset()

    def test_static_scale_and_prefill_keep_native_contract(self):
        x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
        with patch(
            MODULE + ".try_small_fp8_quant",
            side_effect=AssertionError("unexpected candidate"),
        ):
            for role, scale in (
                ("PREFILL", None),
                ("DECODE", torch.tensor([0.125], device="cuda")),
            ):
                actual = _quantize_projection_input(x, role=role, scale=scale)
                expected = scaled_fp8_per_tensor_quant(x, scale=scale)
                for a, b in zip(actual, expected):
                    self.assertTrue(
                        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    )

    def test_full_projection_and_pair_keep_weight_scale_and_bf16_output(self):
        for rows in (1, 3, 4, 8, 16, 32):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            layers = [
                Fp8Linear(
                    (torch.randn(512, 2560, device="cuda") * 0.1).to(
                        torch.float8_e4m3fn
                    ),
                    torch.tensor([s], device="cuda"),
                    role="DECODE",
                )
                for s in (0.013475, 0.2719)
            ]
            with patch(MODULE + ".try_small_fp8_quant", return_value=None):
                expected = [layer(x) for layer in layers]
                pair = project_fp8_pair(x, *layers)
            for a, b in zip([layer(x) for layer in layers], expected):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
            for a, b in zip(project_fp8_pair(x, *layers), pair):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    def test_shared_down_full_projection_dynamic_graph_matches_native(self):
        layer = Fp8Linear(
            (torch.randn(2560, 640, device="cuda") * 0.1).to(torch.float8_e4m3fn),
            torch.tensor([0.11743], device="cuda"),
            role="PDFUSION",
        )
        for rows in (1, 3, 4, 8, 12, 16, 20, 24, 28, 32):
            x = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
            with patch(MODULE + ".try_small_fp8_quant", return_value=None):
                expected = layer(x)
            self.assertTrue(
                torch.equal(layer(x).view(torch.uint8), expected.view(torch.uint8))
            )
            for _ in range(3):
                layer(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = layer(x)
            pointer = actual.data_ptr()
            for generation in range(32):
                x.normal_().mul_((0.0, 1.0e-12, 0.01, 1.0, 1.0e4)[generation % 5])
                with patch(MODULE + ".try_small_fp8_quant", return_value=None):
                    expected = layer(x)
                graph.replay()
                self.assertEqual(actual.data_ptr(), pointer)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                )
            graph.reset()


if __name__ == "__main__":
    unittest.main()

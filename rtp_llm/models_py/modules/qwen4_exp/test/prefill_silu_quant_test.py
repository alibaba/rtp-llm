import unittest
from types import SimpleNamespace
from unittest.mock import patch

import flashinfer
import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.models_py.modules.qwen4_exp.prefill_silu_quant_triton import (
    try_silu_mul_quant,
)
from rtp_llm.ops import ActivationType

MODULE = "rtp_llm.models_py.modules.qwen4_exp.prefill_silu_quant_triton"


class PrefillSiluQuantTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(717)

    def test_exact_codes_scales_padding_and_nonfinite_values(self):
        for width in (320, 640):
            for rows in (4093, 4096, 5003, 8192, 12288, 16384):
                gate_up = torch.randn(
                    rows, width * 2, device="cuda", dtype=torch.bfloat16
                )
                for kind in ("random", "zero", "subnormal", "nonfinite"):
                    if kind == "zero":
                        gate_up.zero_()
                    elif kind == "subnormal":
                        gate_up[:, :width].fill_(1.0e-38)
                        gate_up[:, width:].fill_(0.75)
                    elif kind == "nonfinite":
                        gate_up.normal_()
                        gate_up[0, 0] = float("nan")
                        gate_up[1, 0] = float("inf")
                    activated = flashinfer.activation.silu_and_mul(gate_up)
                    padded = torch.empty(
                        ((rows + 15) // 16 * 16, width),
                        device="cuda",
                        dtype=torch.float8_e4m3fn,
                    )
                    expected = scaled_fp8_per_tensor_quant(activated, output=padded)
                    actual = try_silu_mul_quant(gate_up)
                    for a, b in zip(actual, expected):
                        self.assertTrue(
                            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                        )

    def test_full_projection_graph_refreshes_input_and_weight_scale(self):
        for width in (320, 640):
            layer = Fp8Linear(
                (torch.randn(2560, width, device="cuda") * 0.1).to(torch.float8_e4m3fn),
                torch.tensor([0.11743], device="cuda"),
                role="PREFILL",
            )
            for rows in (4093, 8192, 16384):
                gate_up = torch.randn(
                    rows, width * 2, device="cuda", dtype=torch.bfloat16
                )
                for _ in range(3):
                    layer.forward_silu_and_mul(gate_up)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = layer.forward_silu_and_mul(gate_up)
                pointer = actual.data_ptr()
                for magnitude in (0.0, 0.01, 1.0, 100.0):
                    gate_up.normal_().mul_(magnitude)
                    layer.scale.fill_(0.013475 + magnitude * 0.001)
                    expected = layer(flashinfer.activation.silu_and_mul(gate_up))
                    graph.replay()
                    self.assertEqual(actual.data_ptr(), pointer)
                    self.assertTrue(torch.equal(actual, expected))
                graph.reset()

    def test_complete_mlp_and_activation_hooks(self):
        for rows, width in ((4093, 320), (8192, 640)):
            model = DenseMLP.__new__(DenseMLP)
            torch.nn.Module.__init__(model)
            model.is_gated = True
            model.activation_type = ActivationType.Swiglu
            model.parallelism_config = SimpleNamespace(get_ffn_tp_size=lambda: 1)
            model.act_fn = FusedSiluAndMul()
            model.up_proj = Fp8Linear(
                (torch.randn(width * 2, 2560, device="cuda") * 0.1).to(
                    torch.float8_e4m3fn
                ),
                torch.tensor([0.013475], device="cuda"),
                role="PREFILL",
            )
            model.down_proj = Fp8Linear(
                (torch.randn(2560, width, device="cuda") * 0.1).to(torch.float8_e4m3fn),
                torch.tensor([0.11743], device="cuda"),
                role="PREFILL",
            )
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            with patch(MODULE + ".try_silu_mul_quant", return_value=None):
                expected = model(x)
            self.assertTrue(torch.equal(model(x), expected))
            calls = []

            def hook(_module, _inputs, output):
                calls.append(True)
                return output * 0.5

            handle = model.act_fn.register_forward_hook(hook)
            try:
                expected = model.down_proj(
                    flashinfer.activation.silu_and_mul(model.up_proj(x)) * 0.5
                )
                self.assertTrue(torch.equal(model(x), expected))
                self.assertEqual(len(calls), 1)
            finally:
                handle.remove()

    def test_unsupported_inputs_and_output_aliases_are_untouched(self):
        good = torch.ones(4093, 640, device="cuda", dtype=torch.bfloat16)
        for x in (
            good.cpu(),
            good.float(),
            good.t(),
            good[:, ::2],
            good.clone().requires_grad_(),
            torch.empty(4092, 640, device="cuda", dtype=torch.bfloat16),
            torch.empty(16385, 640, device="cuda", dtype=torch.bfloat16),
            torch.empty(4093, 638, device="cuda", dtype=torch.bfloat16),
        ):
            self.assertIsNone(try_silu_mul_quant(x))
        for output in (
            torch.ones(4093, 320, device="cuda", dtype=torch.bfloat16),
            torch.ones(4093, 321, device="cuda", dtype=torch.float8_e4m3fn),
            good.view(torch.float8_e4m3fn).flatten()[: 4093 * 320].view(4093, 320),
        ):
            before = output.view(torch.uint8).clone()
            self.assertIsNone(try_silu_mul_quant(good, output=output))
            self.assertTrue(torch.equal(output.view(torch.uint8), before))

    def test_decode_native_codes_scale_and_graph(self):
        for rows in (20, 24, 28, 32):
            layer = Fp8Linear(
                (torch.randn(2560, 640, device="cuda") * 0.1).to(torch.float8_e4m3fn),
                torch.tensor([0.11743], device="cuda"),
                role="DECODE",
            )
            x = torch.randn(rows, 1280, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                layer.forward_silu_and_mul(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = layer.forward_silu_and_mul(x)
            for magnitude in (0.0, 1.0e-38, 0.01, 1.0, 100.0):
                x.normal_().mul_(magnitude)
                quant, scale = try_silu_mul_quant(x)
                activated = flashinfer.activation.silu_and_mul(x)
                reference = scaled_fp8_per_tensor_quant(
                    activated, output=torch.empty_like(quant)
                )
                for a, b in zip((quant, scale), reference):
                    self.assertTrue(
                        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    )
                layer.scale.fill_(0.013475 + magnitude * 0.001)
                expected = layer(activated)
                graph.replay()
                self.assertTrue(torch.equal(actual, expected))
            graph.reset()
        for rows, width in ((16, 1280), (19, 1280), (25, 1280), (33, 1280), (24, 640)):
            self.assertIsNone(
                try_silu_mul_quant(
                    torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
                )
            )


if __name__ == "__main__":
    unittest.main()

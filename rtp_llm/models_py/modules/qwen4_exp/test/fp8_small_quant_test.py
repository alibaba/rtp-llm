import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.base.common.fp8_linear import (
    Fp8Linear,
    _quantize_projection_input,
    project_fp8_pair,
)
from rtp_llm.models_py.triton_kernels.common.fp8_small_quant import (
    try_small_fp8_quant,
    try_silu_mul_fp8_quant,
)

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

    def test_native_bytes_all_measured_widths_and_prime_rows(self):
        for rows in (1, 3, 4, 7, 8, 11, 12, 16, 20, 24, 25, 32):
            for width in (320, 640, 2560, 10240):
                if width == 640 and rows not in (4, 8, 12, 16):
                    continue
                if rows * width > 65536:
                    continue
                for magnitude in (0.0, 1.0e-12, 0.01, 1.0, 1.0e10):
                    x = (
                        torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                        * magnitude
                    )
                    for padded in (False, True):
                        self.compare(x, padded)

    def test_every_bf16_bit_pattern_and_nonfinite_reduction(self):
        bits = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        for start in range(0, bits.numel(), 2560):
            x = torch.zeros(1, 2560, device="cuda", dtype=torch.bfloat16)
            part = bits[start : start + 2560]
            x.flatten()[: part.numel()].copy_(part)
            self.compare(x)
        for value in (float("nan"), float("inf"), -float("inf"), -0.0):
            self.compare(
                torch.full((4, 2560), value, device="cuda", dtype=torch.bfloat16)
            )

    def test_uniform_subnormal_scale_flush_matches_native(self):
        for bits in (
            0,
            1,
            2,
            0x7F,
            0x80,
            0xFF,
            0x100,
            0x17F,
            0x180,
            0x200,
            0x400,
            0x480,
            0x500,
            0x8001,
            0x807F,
            0x8080,
            0x7F7F,
            0xFF7F,
        ):
            x = (
                torch.full((4, 2560), bits, device="cuda", dtype=torch.int32)
                .to(torch.int16)
                .view(torch.bfloat16)
            )
            self.compare(x)
            x.flatten()[1:].zero_()
            self.compare(x)

    def test_dynamic_graph_preserves_addresses_and_refreshes_scale_and_padding(self):
        for rows, width in (
            (1, 320),
            (4, 640),
            (16, 640),
            (4, 2560),
            (3, 10240),
            (25, 2560),
        ):
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            shape = (((rows + 15) // 16) * 16, width)
            output = torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)
            for _ in range(3):
                try_small_fp8_quant(x, output=output)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = try_small_fp8_quant(x, output=output)
            pointers = [v.data_ptr() for v in actual]
            for magnitude in (0.0, 1.0e-12, 0.1, 10.0):
                x.normal_().mul_(magnitude)
                output.fill_(float("nan"))
                graph.replay()
                expected = scaled_fp8_per_tensor_quant(
                    x, output=torch.empty_like(output)
                )
                self.assertEqual([v.data_ptr() for v in actual], pointers)
                self.assertTrue(
                    torch.equal(
                        actual[0].view(torch.uint8), expected[0].view(torch.uint8)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        actual[1].view(torch.int32), expected[1].view(torch.int32)
                    )
                )
            graph.reset()

    def test_shared_silu_mul_quant_matches_two_kernel_contract(self):
        import flashinfer

        for rows in (4, 8, 12, 16):
            for magnitude in (0.0, 0.01, 0.5, 5.0, 100.0):
                gate_up = (
                    torch.randn(rows, 1280, device="cuda", dtype=torch.bfloat16)
                    * magnitude
                )
                activated = flashinfer.activation.silu_and_mul(gate_up)
                expected = try_small_fp8_quant(activated)
                actual = try_silu_mul_fp8_quant(gate_up)
                self.assertIsNotNone(actual)
                for a, b in zip(actual, expected):
                    self.assertTrue(
                        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    )
        bits = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        for offset in range(0, bits.numel(), 4 * 640):
            gate_up = torch.zeros(4, 1280, device="cuda", dtype=torch.bfloat16)
            chunk = bits[offset : offset + 4 * 640]
            gate_up[:, :640].flatten()[: chunk.numel()].copy_(chunk)
            gate_up[:, 640:].fill_(0.75)
            activated = flashinfer.activation.silu_and_mul(gate_up)
            expected = try_small_fp8_quant(activated)
            actual = try_silu_mul_fp8_quant(gate_up)
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    def test_unsupported_domains_return_without_writing(self):
        good = torch.ones(4, 2560, device="cuda", dtype=torch.bfloat16)
        for x in (
            good.cpu(),
            good.float(),
            good.t(),
            good[:, ::2],
            torch.ones(32, 10240, device="cuda", dtype=torch.bfloat16),
        ):
            self.assertIsNone(try_small_fp8_quant(x))
        for output in (
            torch.ones(4, 2560, device="cuda", dtype=torch.bfloat16),
            torch.ones(17, 2560, device="cuda", dtype=torch.float8_e4m3fn),
            good.view(torch.float8_e4m3fn)
            .flatten()[: good.numel()]
            .reshape(good.shape),
        ):
            before = output.view(torch.uint8).clone()
            self.assertIsNone(try_small_fp8_quant(good, output=output))
            self.assertTrue(torch.equal(output.view(torch.uint8), before))

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

    def test_shared_down_quant_all_bf16_patterns(self):
        bits = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        for start in range(0, bits.numel(), 2560):
            x = torch.zeros(4, 640, device="cuda", dtype=torch.bfloat16)
            part = bits[start : start + 2560]
            x.flatten()[: part.numel()].copy_(part)
            self.compare(x)
        x = torch.randn(3, 640, device="cuda", dtype=torch.bfloat16)
        output = torch.full(
            (16, 640), float("nan"), device="cuda", dtype=torch.float8_e4m3fn
        )
        before = output.view(torch.uint8).clone()
        self.assertIsNone(try_small_fp8_quant(x, output=output))
        self.assertTrue(torch.equal(before, output.view(torch.uint8)))

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

    def test_shared_down_measured_domain_and_fallback_does_not_write(self):
        for rows in range(1, 33):
            x = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
            output = torch.full(
                (((rows + 15) // 16) * 16, 640),
                float("nan"),
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            before = output.view(torch.uint8).clone()
            actual = try_small_fp8_quant(x, output=output)
            if rows in (4, 8, 12, 16):
                self.assertIsNotNone(actual)
                self.compare(x)
            else:
                self.assertIsNone(actual)
                self.assertTrue(torch.equal(before, output.view(torch.uint8)))

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.compare(torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16))
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == "__main__":
    unittest.main()
